"""
Section 3.5 (Human-in-the-Loop Review & Memory) and Section 3.6 (Cover Letter).

Designed to consume Dale's process_top_job outputs (change_log.json paths) and
Jennifer's memory helpers shape: {"skills": [], "facts": []}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from datetime import datetime, timezone
import json
import re

from compile_resume import compile_resume
import pipeline_tracing as tracing
import top3_resume_pipeline as t3p


# Default to repo-root memory.json (callers usually pass an explicit path).
MEMORY_PATH_DEFAULT = str(Path(__file__).resolve().parents[1] / "memory.json")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def load_memory(path: str | Path = MEMORY_PATH_DEFAULT) -> dict[str, Any]:
    memory_path = Path(path)
    if memory_path.exists():
        return json.loads(memory_path.read_text(encoding="utf-8"))
    return {"skills": [], "facts": []}


def save_memory(
    memory: dict[str, Any],
    path: str | Path = MEMORY_PATH_DEFAULT,
) -> Path:
    memory_path = Path(path)
    memory_path.write_text(
        json.dumps(memory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return memory_path


def memory_evidence_strings(memory: dict[str, Any]) -> list[str]:
    """Flatten memory into strings the tailoring prompt already accepts."""
    evidence: list[str] = []
    for skill in memory.get("skills", []) or []:
        evidence.append(f"skill: {skill}")
    for fact in memory.get("facts", []) or []:
        if isinstance(fact, str):
            evidence.append(fact)
            continue
        if not isinstance(fact, dict):
            continue
        text = fact.get("text") or fact.get("skill") or ""
        provenance = fact.get("provenance", "")
        if text and provenance:
            evidence.append(f"{text} [{provenance}]")
        elif text:
            evidence.append(str(text))
    # Preserve order, drop empties/dupes
    seen: set[str] = set()
    unique: list[str] = []
    for item in evidence:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item.strip())
    return unique


def refresh_pipeline_memory(memory: dict[str, Any]) -> None:
    """Push latest memory into Dale's module globals for same-run reworks."""
    evidence = memory_evidence_strings(memory)
    t3p._PIPELINE_MEMORY_EVIDENCE = list(evidence)  # noqa: SLF001


def append_memory_facts(
    memory: dict[str, Any],
    facts: list[dict[str, Any]],
    *,
    path: str | Path = MEMORY_PATH_DEFAULT,
) -> dict[str, Any]:
    """
    Persist candidate skills/facts only. Emits a nested `memory_update` span.
    """
    with tracing.trace_span(
        "memory_update",
        input={"incoming_facts": facts, "path": str(path)},
        metadata={"span_role": "memory_write"},
    ) as span:
        skills = list(memory.get("skills") or [])
        skill_lower = {s.lower() for s in skills}
        stored_facts = list(memory.get("facts") or [])

        written: list[dict[str, Any]] = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            text = str(fact.get("text") or "").strip()
            skill = str(fact.get("skill") or "").strip()
            if not text and not skill:
                continue

            provenance = str(
                fact.get("provenance")
                or "stated by candidate, review round unknown"
            ).strip()

            record = {
                "text": text or skill,
                "skill": skill or None,
                "provenance": provenance,
                "rank": fact.get("rank"),
                "job_title": fact.get("job_title"),
                "company": fact.get("company"),
                "round": fact.get("round"),
                "added_at": datetime.now(timezone.utc).isoformat(),
            }
            stored_facts.append(record)
            written.append(record)

            if skill and skill.lower() not in skill_lower:
                skills.append(skill)
                skill_lower.add(skill.lower())

        memory["skills"] = skills
        memory["facts"] = stored_facts
        saved = save_memory(memory, path=path)
        refresh_pipeline_memory(memory)

        span.output = {
            "written": written,
            "memory_path": str(saved),
            "skills_count": len(skills),
            "facts_count": len(stored_facts),
        }
        span.metadata["config"] = {"path": str(path)}

    return memory


# ---------------------------------------------------------------------------
# Change-log presentation / parsing
# ---------------------------------------------------------------------------

def load_change_log(result: dict[str, Any]) -> dict[str, Any]:
    path = result.get("change_log_path")
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    if "change_log" in result and isinstance(result["change_log"], dict):
        return result["change_log"]
    raise FileNotFoundError(
        f"No change_log found for rank={result.get('rank')} "
        f"(change_log_path={path!r})"
    )


def format_change_log_for_console(change_log: dict[str, Any]) -> str:
    job = change_log.get("job") or {}
    lines = [
        "=" * 72,
        f"RESUME #{change_log.get('rank')} — "
        f"{job.get('title', '?')} at {job.get('company', '?')}",
        f"PDF: {change_log.get('pdf_file')} | "
        f"one_page_verified={change_log.get('one_page_verified')}",
        "-" * 72,
    ]

    summary = change_log.get("summary_edit") or {}
    lines.append("SUMMARY EDIT")
    lines.append(f"  before: {summary.get('before', '')}")
    lines.append(f"  after:  {summary.get('after', '')}")
    lines.append(f"  reason: {summary.get('reason', '')}")
    lines.append(f"  evidence: {json.dumps(summary.get('evidence', []), ensure_ascii=False)}")
    lines.append("")

    lines.append("EXPERIENCE BULLET EDITS")
    for bullet in change_log.get("experience_bullet_edits") or []:
        lines.append(f"  [{bullet.get('target')}]")
        lines.append(f"    before: {bullet.get('before', '')}")
        lines.append(f"    after:  {bullet.get('after', '')}")
        lines.append(f"    reason: {bullet.get('reason', '')}")
        lines.append(
            f"    evidence: {json.dumps(bullet.get('evidence', []), ensure_ascii=False)}"
        )
    lines.append("")

    skills = change_log.get("skill_edits") or {}
    lines.append("SKILL EDITS")
    for change in skills.get("changes") or []:
        lines.append(
            f"  - {change.get('action')}: {change.get('skill')} "
            f"({change.get('reason')})"
        )
        lines.append(
            f"    evidence: {json.dumps(change.get('evidence', []), ensure_ascii=False)}"
        )
    if not skills.get("changes"):
        lines.append("  (none)")
    lines.append("")

    lines.append("PROJECT SWAPS")
    swaps = change_log.get("project_swaps") or []
    if not swaps:
        lines.append("  (none)")
    for swap in swaps:
        lines.append(
            f"  - slot {swap.get('slot')}: remove {swap.get('remove_project_id')} "
            f"→ add {swap.get('add_project_id')} ({swap.get('reason')})"
        )
    lines.append("")

    gaps = change_log.get("genuine_gaps") or []
    lines.append("GENUINE GAPS")
    if not gaps:
        lines.append("  (none)")
    for gap in gaps:
        if isinstance(gap, dict):
            lines.append(f"  - {gap.get('requirement')}: {gap.get('reason')}")
        else:
            lines.append(f"  - {gap}")
    lines.append("=" * 72)
    return "\n".join(lines)


def parse_review_decision(raw: str) -> tuple[str, str]:
    """
    Return (decision, comments) where decision is 'approve' or 'reject'.
    Accepts: approve | a | yes | y | reject: comments | reject comments
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty review input. Type 'approve' or 'reject: <comments>'.")

    lower = text.lower()
    if lower in {"approve", "a", "yes", "y", "ok"}:
        return "approve", ""

    if lower.startswith("reject"):
        rest = text[6:].lstrip(" :,-")
        return "reject", rest

    # Free text without keyword → treat as reject with comments
    return "reject", text


# ---------------------------------------------------------------------------
# LLM helpers for fact extraction and cover-letter drafting
# ---------------------------------------------------------------------------

def _call_json_model(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2500,
) -> dict[str, Any]:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]
    raw = "".join(parts).strip()
    if not raw:
        raise RuntimeError("Model returned empty response.")
    return t3p.parse_json_response(raw)


def extract_candidate_facts_from_comments(
    client: Any,
    *,
    model: str,
    comments: str,
    rank: int,
    job: dict[str, Any],
    round_number: int,
) -> tuple[list[dict[str, Any]], str]:
    """
    LLM interprets reviewer comments and returns only NEW candidate skills/facts
    plus concrete rework instructions. Never returns job data or scores.
    """
    with tracing.trace_span(
        "extract_review_facts",
        input={"comments": comments, "rank": rank, "round": round_number},
        metadata={"model": model},
        as_type="generation",
    ) as span:
        schema = {
            "new_facts": [
                {
                    "text": "Short factual claim about the candidate",
                    "skill": "Canonical skill name if this is a skill, else null",
                }
            ],
            "rework_instructions": (
                "Concrete instructions for the resume rework, incorporating "
                "any new facts and other feedback."
            ),
        }
        result = _call_json_model(
            client,
            model=model,
            system=(
                "You extract ONLY candidate skills/facts from human review "
                "comments for a job-search agent. Ignore job requirements that "
                "are not stated as things the candidate knows. Return JSON only."
            ),
            user=(
                f"REVIEW COMMENTS:\n{comments}\n\n"
                f"RESUME RANK: {rank}\n"
                f"JOB: {job.get('job_title') or job.get('title')} at "
                f"{job.get('company')}\n"
                f"REVIEW ROUND: {round_number}\n\n"
                "Rules:\n"
                "- Include a fact only when the candidate asserts something "
                "about themselves (e.g. 'add GraphQL, I know it').\n"
                "- Do not invent facts.\n"
                "- Do not include job data, scores, or company info as facts.\n"
                "- rework_instructions must tell the tailor how to act on the "
                "comments, including newly stated skills.\n\n"
                f"Return exactly this JSON shape:\n{json.dumps(schema, indent=2)}"
            ),
        )

        provenance = (
            f"stated by candidate, review round {round_number}, resume {rank}"
        )
        facts: list[dict[str, Any]] = []
        for item in result.get("new_facts") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            skill = item.get("skill")
            skill_text = str(skill).strip() if skill else ""
            if not text and not skill_text:
                continue
            facts.append(
                {
                    "text": text or skill_text,
                    "skill": skill_text or None,
                    "provenance": provenance,
                    "rank": rank,
                    "job_title": job.get("job_title") or job.get("title"),
                    "company": job.get("company"),
                    "round": round_number,
                }
            )

        span.output = {
            "facts": facts,
            "rework_instructions": result.get("rework_instructions"),
        }
        return facts, str(result.get("rework_instructions") or comments)


# ---------------------------------------------------------------------------
# Section 3.5 — Human review (interleaved with tailor per rank)
# ---------------------------------------------------------------------------

def _resolve_job(
    result: dict[str, Any],
    jobs: list[dict[str, Any]],
    change_log: dict[str, Any] | None = None,
) -> dict[str, Any]:
    title = (result.get("title") or "").strip()
    company = (result.get("company") or "").strip()
    job_index = {
        (
            (j.get("job_title") or j.get("title") or "").strip().lower(),
            (j.get("company") or "").strip().lower(),
        ): j
        for j in jobs
    }
    job = job_index.get((title.lower(), company.lower()))
    if job is not None:
        return job

    clog = change_log or {}
    return {
        "job_title": (clog.get("job") or {}).get("title", title),
        "company": (clog.get("job") or {}).get("company", company),
        **{k: v for k, v in result.items() if k not in {"rank"}},
    }


def review_one_resume(
    result: dict[str, Any],
    *,
    job: dict[str, Any],
    master_resume_tex: str,
    portfolio: dict[str, Any],
    resume_summary: dict[str, Any],
    master_skills: list[str],
    memory: dict[str, Any],
    client: Any,
    model: str,
    output_dir: str | Path,
    memory_path: str | Path = MEMORY_PATH_DEFAULT,
    max_rounds: int = 2,
    input_fn: Callable[[str], str] | None = None,
    fit_projects: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Human gate for a single already-tailored resume.
    Returns (approved_result, round_log_entries).
    """
    ask = input_fn or input
    output_dir = Path(output_dir)
    rank = int(result["rank"])
    title = result.get("title") or job.get("job_title") or ""
    company = result.get("company") or job.get("company") or ""
    current = dict(result)
    change_log = load_change_log(current)
    round_log: list[dict[str, Any]] = []
    unresolved = False
    round_number = 0

    print(format_change_log_for_console(change_log))

    while True:
        prompt = (
            f"\nResume #{rank} ({title} @ {company})\n"
            "Enter 'approve' or 'reject: <comments>': "
        )
        raw = ask(prompt)
        decision, comments = parse_review_decision(raw)

        if decision == "approve":
            current["review_status"] = "approved"
            current["unresolved"] = False
            current["review_rounds"] = round_number
            break

        round_number += 1
        print(f"\n→ Rejected (round {round_number}/{max_rounds}): {comments}")

        facts, rework_instructions = extract_candidate_facts_from_comments(
            client,
            model=model,
            comments=comments,
            rank=rank,
            job=job,
            round_number=round_number,
        )

        if facts:
            append_memory_facts(memory, facts, path=memory_path)
            print(
                f"  Memory updated with {len(facts)} fact(s); "
                "available to later ranks' tailor passes in this run."
            )
        else:
            refresh_pipeline_memory(memory)

        round_record: dict[str, Any] = {
            "rank": rank,
            "round": round_number,
            "feedback": comments,
            "facts_added": facts,
            "rework_instructions": rework_instructions,
            "actions_taken": [],
        }

        if round_number > max_rounds:
            unresolved = True
            round_record["actions_taken"].append(
                "max_rounds_reached_proceeding_with_best_version"
            )
            round_log.append(round_record)
            print(
                f"  Cap of {max_rounds} revision rounds reached. "
                "Proceeding with best current version (unresolved)."
            )
            current["review_status"] = "approved_unresolved"
            current["unresolved"] = True
            current["review_rounds"] = round_number - 1
            break

        t3p.configure_pipeline(
            client=client,
            resume_summary=resume_summary,
            master_skills=master_skills,
            memory_evidence=memory_evidence_strings(memory),
        )

        with tracing.trace_span(
            "resume_rework",
            input={
                "rank": rank,
                "round": round_number,
                "feedback": comments,
                "rework_instructions": rework_instructions,
            },
            metadata={"job_title": title, "company": company},
        ) as rework_span:
            reworked = t3p.process_top_job(
                job=job,
                rank=rank,
                master_resume_tex=master_resume_tex,
                portfolio=portfolio,
                output_dir=output_dir,
                model=model,
                revision_feedback=rework_instructions,
                fit_projects=fit_projects,
            )
            rework_span.output = {
                "pdf_path": reworked.get("pdf_path"),
                "change_log_path": reworked.get("change_log_path"),
            }
            round_record["actions_taken"].append("re-ran_tailoring_tool")
            round_record["rework_result"] = {
                "pdf_path": reworked.get("pdf_path"),
                "change_log_path": reworked.get("change_log_path"),
            }
            current = dict(reworked)

        round_log.append(round_record)
        change_log = load_change_log(current)
        print("\nUpdated change log after rework:")
        print(format_change_log_for_console(change_log))

    current["change_log"] = change_log
    current["job"] = job
    status = "UNRESOLVED (best effort)" if unresolved else "APPROVED"
    print(f"\n✓ Resume #{rank} {status}\n")
    return current, round_log


def interleaved_tailor_and_review(
    jobs: list[dict[str, Any]],
    *,
    master_resume_tex: str,
    portfolio: dict[str, Any],
    resume_summary: dict[str, Any],
    master_skills: list[str],
    memory: dict[str, Any],
    client: Any,
    model: str,
    output_dir: str | Path,
    memory_path: str | Path = MEMORY_PATH_DEFAULT,
    max_rounds: int = 2,
    input_fn: Callable[[str], str] | None = None,
    fit_structured_by_rank: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Interleaved Top-3 path: for each rank in order,
      configure_pipeline(current memory) → tailor → present change log → review
    then proceed to the next rank.

    A fact learned while reviewing rank N is in memory before rank N+1's
    tailor prompt is built.

    fit_structured_by_rank: optional map rank → fit_analysis structured result
    (must include projects.*) so project swaps recommended at fit time are
    forced into the edit plan.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_structured_by_rank = fit_structured_by_rank or {}

    with tracing.trace_span(
        "interleaved_tailor_and_review",
        input={
            "job_count": len(jobs),
            "max_rounds": max_rounds,
            "ranks": list(range(1, len(jobs) + 1)),
        },
        metadata={"memory_path": str(memory_path)},
    ) as batch_span:
        approved: list[dict[str, Any]] = []
        all_rounds: list[dict[str, Any]] = []

        for rank, job in enumerate(jobs, start=1):
            title = job.get("job_title") or job.get("title") or ""
            company = job.get("company") or ""

            # Always refresh tailor evidence from the latest memory before
            # building this rank's edit plan.
            t3p.configure_pipeline(
                client=client,
                resume_summary=resume_summary,
                master_skills=master_skills,
                memory_evidence=memory_evidence_strings(memory),
            )

            print("\n" + "#" * 72)
            print(
                f"RANK {rank}/{len(jobs)} — tailor then review: "
                f"{title} @ {company}"
            )
            print(
                f"  memory skills entering tailor: {list(memory.get('skills') or [])}"
            )
            print("#" * 72)

            fit_structured = fit_structured_by_rank.get(rank) or {}
            fit_projects = (
                fit_structured.get("projects")
                if isinstance(fit_structured, dict)
                else None
            )

            with tracing.trace_span(
                "process_top_job",
                input={
                    "rank": rank,
                    "job_title": title,
                    "company": company,
                    "memory_skills": list(memory.get("skills") or []),
                    "memory_evidence": memory_evidence_strings(memory),
                    "fit_projects_verdict": (
                        fit_projects.get("verdict")
                        if isinstance(fit_projects, dict)
                        else None
                    ),
                },
                metadata={"phase": "initial_tailor_for_rank"},
            ) as tailor_span:
                last_error: Exception | None = None
                result = None
                for attempt in range(1, 4):
                    try:
                        result = t3p.process_top_job(
                            job=job,
                            rank=rank,
                            master_resume_tex=master_resume_tex,
                            portfolio=portfolio,
                            output_dir=output_dir,
                            model=model,
                            fit_projects=fit_projects
                            if isinstance(fit_projects, dict)
                            else None,
                        )
                        last_error = None
                        break
                    except ValueError as exc:
                        # Flaky LLM edit plans (missing evidence, etc.) — retry.
                        last_error = exc
                        print(
                            f"  Tailor attempt {attempt}/3 failed validation: {exc}"
                        )
                if result is None:
                    raise RuntimeError(
                        f"Tailor failed for rank {rank} after retries"
                    ) from last_error
                tailor_span.output = {
                    "pdf_path": result.get("pdf_path"),
                    "change_log_path": result.get("change_log_path"),
                    "folder": result.get("folder"),
                }

            with tracing.trace_span(
                "human_review_rank",
                input={"rank": rank, "job_title": title, "company": company},
            ) as review_span:
                approved_one, rounds = review_one_resume(
                    result,
                    job=job,
                    master_resume_tex=master_resume_tex,
                    portfolio=portfolio,
                    resume_summary=resume_summary,
                    master_skills=master_skills,
                    memory=memory,
                    client=client,
                    model=model,
                    output_dir=output_dir,
                    memory_path=memory_path,
                    max_rounds=max_rounds,
                    input_fn=input_fn,
                    fit_projects=fit_projects
                    if isinstance(fit_projects, dict)
                    else None,
                )
                review_span.output = {
                    "review_status": approved_one.get("review_status"),
                    "rounds": rounds,
                    "memory_skills": list(memory.get("skills") or []),
                }

            approved.append(approved_one)
            all_rounds.extend(rounds)

        batch_span.output = {
            "approved_count": len(approved),
            "rounds": all_rounds,
            "memory_skills": list(memory.get("skills") or []),
        }
        return approved


def human_review(
    results: list[dict[str, Any]],
    *,
    jobs: list[dict[str, Any]] | None = None,
    master_resume_tex: str,
    portfolio: dict[str, Any],
    resume_summary: dict[str, Any],
    master_skills: list[str],
    memory: dict[str, Any],
    client: Any,
    model: str,
    output_dir: str | Path,
    memory_path: str | Path = MEMORY_PATH_DEFAULT,
    max_rounds: int = 2,
    input_fn: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    """
    Backward-compatible wrapper: review already-tailored results in rank order.

    Prefer interleaved_tailor_and_review() so later ranks' tailor passes see
    memory facts from earlier reviews in the same run.
    """
    jobs = jobs or []
    with tracing.trace_span(
        "human_review",
        input={
            "resume_count": len(results),
            "max_rounds": max_rounds,
            "ranks": [r.get("rank") for r in results],
            "note": "legacy_batch_review_after_all_tailors",
        },
        metadata={"memory_path": str(memory_path)},
    ) as review_span:
        approved: list[dict[str, Any]] = []
        all_rounds: list[dict[str, Any]] = []
        ordered = sorted(results, key=lambda r: int(r.get("rank") or 0))

        print("\n" + "#" * 72)
        print("HUMAN REVIEW — per-resume gate (legacy batch mode)")
        print("#" * 72)

        for result in ordered:
            clog = load_change_log(result)
            job = _resolve_job(result, jobs, clog)
            one, rounds = review_one_resume(
                result,
                job=job,
                master_resume_tex=master_resume_tex,
                portfolio=portfolio,
                resume_summary=resume_summary,
                master_skills=master_skills,
                memory=memory,
                client=client,
                model=model,
                output_dir=output_dir,
                memory_path=memory_path,
                max_rounds=max_rounds,
                input_fn=input_fn,
            )
            approved.append(one)
            all_rounds.extend(rounds)

        review_span.output = {
            "approved_count": len(approved),
            "rounds": all_rounds,
            "memory_skills": list(memory.get("skills") or []),
        }
        return approved


# ---------------------------------------------------------------------------
# Section 3.6 — Cover letter
# ---------------------------------------------------------------------------

def extract_contact_header(resume_tex: str) -> dict[str, str]:
    """Pull name/location/phone/email/github from the resume header block."""
    header_match = re.search(
        r"\\begin\{center\}(.*?)\\end\{center\}",
        resume_tex,
        flags=re.DOTALL,
    )
    block = header_match.group(1) if header_match else resume_tex[:800]

    name_match = re.search(
        r"\\scshape\s+([A-Za-z][A-Za-z .'-]+)",
        block,
    )
    email_match = re.search(
        r"mailto:([^}]+)",
        block,
    )
    github_match = re.search(
        r"\{(https://github\.com/[^}]+)\}|\{(github\.com/[^}]+)\}",
        block,
    )
    phone_match = re.search(
        r"(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})",
        block,
    )
    # Location is typically the first plain segment before a phone/email
    location = ""
    loc_match = re.search(
        r"([A-Za-z .]+,\s*[A-Z]{2})\s*\$\|\$",
        block,
    )
    if loc_match:
        location = loc_match.group(1).strip()

    github = ""
    if github_match:
        github = next(g for g in github_match.groups() if g)

    return {
        "name": (name_match.group(1).strip() if name_match else "Candidate"),
        "location": location,
        "phone": phone_match.group(1) if phone_match else "",
        "email": email_match.group(1) if email_match else "",
        "github": github,
    }


def _evidence_bundle(
    resume_summary: dict[str, Any],
    portfolio: dict[str, Any],
    master_skills: list[str],
    memory: dict[str, Any],
    tailored_tex: str | None,
) -> dict[str, Any]:
    return {
        "resume_summary": resume_summary,
        "portfolio_project_names": [
            p.get("project_name") for p in portfolio.get("projects", [])
        ],
        "portfolio_projects": portfolio.get("projects", []),
        "master_skills": master_skills,
        "memory": memory,
        "tailored_resume_tex_excerpt": (tailored_tex or "")[:6000],
    }


def build_cover_letter_tex(
    contact: dict[str, str],
    letter: dict[str, Any],
) -> str:
    def esc(value: Any) -> str:
        return t3p.latex_escape(str(value or ""))

    skills = letter.get("skills_line") or []
    if isinstance(skills, list):
        skills_text = ", ".join(esc(s) for s in skills)
    else:
        skills_text = esc(skills)

    body_paras = letter.get("body_paragraphs") or []
    body_tex = "\n\n".join(esc(p) for p in body_paras if str(p).strip())

    header_bits = [
        esc(contact.get("location", "")),
        esc(contact.get("phone", "")),
    ]
    email = contact.get("email") or ""
    github = contact.get("github") or ""

    email_tex = (
        f"\\href{{mailto:{email}}}{{{esc(email)}}}" if email else ""
    )
    github_tex = ""
    if github:
        url = github if github.startswith("http") else f"https://{github}"
        github_tex = f"\\href{{{url}}}{{{esc(github.replace('https://', ''))}}}"

    contact_line = " $|$ ".join(
        part for part in [
            *[b for b in header_bits if b],
            email_tex,
            github_tex,
        ]
        if part
    )

    return f"""% Auto-generated cover letter — evidence-grounded, one page
\\documentclass[letterpaper,11pt]{{article}}

\\usepackage[empty]{{fullpage}}
\\usepackage[hidelinks]{{hyperref}}
\\usepackage[english]{{babel}}

\\pagestyle{{empty}}
\\raggedright
\\setlength{{\\parindent}}{{0pt}}
\\setlength{{\\parskip}}{{10pt}}

\\addtolength{{\\oddsidemargin}}{{-0.5in}}
\\addtolength{{\\evensidemargin}}{{-0.5in}}
\\addtolength{{\\textwidth}}{{1in}}
\\addtolength{{\\topmargin}}{{-0.5in}}
\\addtolength{{\\textheight}}{{1.0in}}

\\begin{{document}}

\\begin{{center}}
    {{\\Huge \\scshape {esc(contact.get("name", ""))}}} \\\\ \\vspace{{2pt}}
    {contact_line}
\\end{{center}}

\\vspace{{8pt}}
{esc(letter.get("greeting", "Dear Hiring Manager,"))}

{esc(letter.get("opening", ""))}

{body_tex}

\\textbf{{Skills:}} {skills_text}

{esc(letter.get("closing", "Sincerely,"))} \\\\
\\vspace{{8pt}}
{esc(contact.get("name", ""))}

\\end{{document}}
"""


def generate_cover_letter(
    job: dict[str, Any],
    resume_summary: dict[str, Any],
    output_path: str | Path,
    *,
    portfolio: dict[str, Any] | None = None,
    master_skills: list[str] | None = None,
    memory: dict[str, Any] | None = None,
    tailored_tex_path: str | Path | None = None,
    master_resume_tex: str | None = None,
    client: Any = None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Autonomous cover-letter tool. LLM drafts from evidence only; LaTeX + pdflatex
    produce a one-page PDF matching the resume header style.
    """
    if client is None:
        raise RuntimeError("generate_cover_letter requires an Anthropic client.")
    if model is None:
        model = t3p.RESUME_EDIT_MODEL

    portfolio = portfolio or {"projects": []}
    master_skills = master_skills or []
    memory = memory or load_memory()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tailored_tex = ""
    if tailored_tex_path and Path(tailored_tex_path).exists():
        tailored_tex = Path(tailored_tex_path).read_text(encoding="utf-8")
    contact_source = tailored_tex or master_resume_tex or ""
    if not contact_source:
        raise ValueError(
            "Need tailored_tex_path or master_resume_tex to build the contact header."
        )
    contact = extract_contact_header(contact_source)

    with tracing.trace_span(
        "generate_cover_letter",
        input={
            "job_title": job.get("job_title") or job.get("title"),
            "company": job.get("company"),
            "output_path": str(output_path),
        },
        metadata={"model": model},
    ) as span:
        evidence = _evidence_bundle(
            resume_summary,
            portfolio,
            master_skills,
            memory,
            tailored_tex,
        )

        schema = {
            "greeting": "Dear Hiring Team,",
            "opening": (
                "1-2 sentences naming the specific role and company, with a "
                "hook drawn from company_details."
            ),
            "body_paragraphs": [
                "1-2 paragraphs mapping real experience to the job description"
            ],
            "skills_line": ["skill1", "skill2"],
            "closing": "Sincerely,",
            "evidence_used": [
                {"source": "resume|portfolio|master_skills|memory", "reference": "..."}
            ],
        }

        letter = _call_json_model(
            client,
            model=model,
            system=(
                "You write evidence-grounded one-page cover letters. Never invent "
                "employers, titles, dates, metrics, skills, or projects. Every "
                "claim must cite resume, portfolio, master_skills, or memory. "
                "Return JSON only."
            ),
            user=(
                f"JOB:\n{json.dumps(job, indent=2, default=str)}\n\n"
                f"CANDIDATE EVIDENCE (only sources you may use):\n"
                f"{json.dumps(evidence, indent=2, default=str)}\n\n"
                "Write a cover letter for THIS job with:\n"
                "- greeting\n"
                "- opening that names the role + company and hooks from "
                "job['company_details']\n"
                "- 1-2 body paragraphs mapping real experience to the job "
                "description\n"
                "- skills_line (only evidenced skills)\n"
                "- closing\n"
                "- evidence_used citations\n\n"
                "If evidence is missing for a claim, omit the claim.\n\n"
                f"Return exactly this JSON shape:\n{json.dumps(schema, indent=2)}"
            ),
            max_tokens=3000,
        )

        # Drop skills not present in evidence skill pools
        allowed_skills = {
            s.lower()
            for s in (
                list(master_skills)
                + list(resume_summary.get("skills") or [])
                + list(memory.get("skills") or [])
            )
        }
        for project in portfolio.get("projects", []):
            for field in ("skills", "tech_stack", "keywords"):
                for item in project.get(field, []) or []:
                    allowed_skills.add(str(item).lower())

        filtered_skills = []
        for skill in letter.get("skills_line") or []:
            if str(skill).lower() in allowed_skills:
                filtered_skills.append(skill)
        letter["skills_line"] = filtered_skills

        tex = build_cover_letter_tex(contact, letter)
        tex_path = output_path.with_suffix(".tex")
        if output_path.suffix.lower() == ".pdf":
            tex_path = output_path.with_suffix(".tex")
        else:
            tex_path = output_path if output_path.suffix.lower() == ".tex" else Path(str(output_path) + ".tex")

        tex_path.write_text(tex, encoding="utf-8")
        pdf_path = compile_resume(str(tex_path))
        pdf_path = t3p.verify_one_page(pdf_path)

        # If caller asked for a specific PDF name/location, copy/rename
        final_pdf = Path(output_path)
        if final_pdf.suffix.lower() != ".pdf":
            final_pdf = final_pdf.with_suffix(".pdf")
        if pdf_path.resolve() != final_pdf.resolve():
            final_pdf.write_bytes(pdf_path.read_bytes())
            pdf_path = final_pdf

        plan_path = tex_path.with_name("cover_letter_plan.json")
        plan_path.write_text(
            json.dumps(letter, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        result = {
            "job_title": job.get("job_title") or job.get("title"),
            "company": job.get("company"),
            "tex_path": str(tex_path),
            "pdf_path": str(pdf_path),
            "plan_path": str(plan_path),
            "one_page_verified": True,
            "letter": letter,
        }
        span.output = {
            "pdf_path": result["pdf_path"],
            "skills_line": letter.get("skills_line"),
            "evidence_used": letter.get("evidence_used"),
        }
        return result


def generate_cover_letters_for_approved(
    approved_results: list[dict[str, Any]],
    *,
    resume_summary: dict[str, Any],
    portfolio: dict[str, Any],
    master_skills: list[str],
    memory: dict[str, Any],
    client: Any,
    model: str,
    master_resume_tex: str | None = None,
) -> list[dict[str, Any]]:
    """Generate one cover letter PDF per approved tailored resume."""
    outputs: list[dict[str, Any]] = []
    with tracing.trace_span(
        "cover_letter_batch",
        input={"count": len(approved_results)},
    ) as batch_span:
        for item in approved_results:
            job = item.get("job") or {}
            folder = Path(item.get("folder") or ".")
            out_pdf = folder / "cover_letter.pdf"
            letter = generate_cover_letter(
                job=job,
                resume_summary=resume_summary,
                output_path=out_pdf,
                portfolio=portfolio,
                master_skills=master_skills,
                memory=memory,
                tailored_tex_path=item.get("tex_path"),
                master_resume_tex=master_resume_tex,
                client=client,
                model=model,
            )
            outputs.append(letter)
            print(
                f"Cover letter saved: {letter['pdf_path']} "
                f"({letter['job_title']} @ {letter['company']})"
            )
        batch_span.output = {
            "pdfs": [o["pdf_path"] for o in outputs],
        }
    return outputs
