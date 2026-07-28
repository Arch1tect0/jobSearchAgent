#!/usr/bin/env python3
"""
End-to-end agent-loop run against REAL data:
  ai_ml_jobs.csv + candidate_persona.json + portfolio.json

LLM selects every stage tool. Scripted human review:
  rank 1 (expected: Fervo Energy) → reject: add LangChain, I know it → approve
  rank 2 → approve
  rank 3 → approve

Verifies:
  - filter drops Oxy (both) + Baylor; keeps the 6 §3.1 survivors
  - score Top-3 is Fervo Energy, CHS, Lone Star College
  - HITL skill fact lands in resume.tex / PDF (materialize_skill_edits)
  - full nested span tree for the run
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import paths  # noqa: F401  — adds src/ to sys.path
from paths import (
    REPO_ROOT,
    ENV_FILE,
    JOBS_CSV,
    CANDIDATE_PERSONA,
    PORTFOLIO_JSON,
    RESUME_TEX,
    NOTEBOOK_PATH,
)

from llm_client import build_llm_client
from agent_loop import AgentContext, TOOLS, run_agent
import hitl_cover as hc
import top3_resume_pipeline as t3p
from pipeline_tracing import (
    root_trace,
    format_trace_summary,
    langfuse_enabled,
    reset_langfuse_client,
)
from run_audit_full_pipeline import load_jobs


ROOT = REPO_ROOT
OUT = ROOT / "tailored_resumes" / "_agent_loop_real"
MEMORY_PATH = OUT / "memory.json"

EXPECTED_SURVIVORS = {
    "Fervo Energy",
    "Herrman & Herrman, P.L.L.C.",
    "Deloitte",
    "Community Health Systems (CHS)",
    "MD Anderson Cancer Center (University of Texas)",
    "Lone Star College (LSC-University Park)",
}
EXPECTED_TOP3_COMPANIES = [
    "Fervo Energy",
    "Community Health Systems (CHS)",
    "Lone Star College (LSC-University Park)",
]
HITL_SKILL = "LangChain"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def make_filter_score_fns(resume_summary, portfolio, master_skills, memory_ref):
    def filter_jobs(
        jobs_list,
        preferred_locations=None,
        min_experience_years=0,
        max_experience_years=None,
        excluded_companies=None,
        remote_only=False,
    ):
        preferred_locations = preferred_locations or []
        excluded_companies = [c.lower() for c in (excluded_companies or [])]
        kept, rejected = [], []
        for job in jobs_list:
            reasons = []
            if remote_only and not job.get("remote", False):
                reasons.append("remote-only preference set, job is not remote")
            if not remote_only and preferred_locations:
                loc_ok = (
                    job.get("location") in preferred_locations
                    or job.get("remote", False)
                )
                if not loc_ok:
                    reasons.append(
                        f"location '{job.get('location')}' not in preferred list"
                    )
            req_years = job.get("years_experience_required", 0)
            if req_years < min_experience_years:
                reasons.append(
                    f"requires {req_years}y, below candidate minimum "
                    f"{min_experience_years}y"
                )
            if (
                max_experience_years is not None
                and req_years > max_experience_years
            ):
                reasons.append(
                    f"requires {req_years}y, above candidate's "
                    f"{max_experience_years}y ceiling"
                )
            if job.get("company", "").lower() in excluded_companies:
                reasons.append(
                    f"company '{job.get('company')}' is on the exclusion list"
                )
            if reasons:
                rejected.append(
                    {
                        "job_title": job.get("job_title"),
                        "company": job.get("company"),
                        "reasons": reasons,
                    }
                )
            else:
                kept.append(job)
        return kept, rejected

    weights = {
        "skill_match": 0.5,
        "experience_alignment": 0.25,
        "domain_alignment": 0.25,
    }

    def score_jobs(jobs_list, candidate_years=None):
        from domain_scoring import score_domain_alignment

        mem = memory_ref["memory"]
        candidate_years = (
            candidate_years
            if candidate_years is not None
            else resume_summary["years_experience"]
        )
        candidate_skills = {
            s.lower()
            for s in (
                master_skills
                + resume_summary["skills"]
                + list(mem.get("skills") or [])
            )
        }
        scored = []
        for job in jobs_list:
            required = set(s.lower() for s in job.get("required_skills", []))
            if not required:
                skill_score, matched = 1.0, []
            else:
                matched = sorted(required & candidate_skills)
                skill_score = len(matched) / len(required)
            req = job.get("years_experience_required", 0)
            exp = (
                1.0
                if req == 0
                else max(0.0, 1.0 - abs(candidate_years - req) / max(req, 1))
            )
            dom = score_domain_alignment(job, portfolio["projects"])
            total = (
                skill_score * weights["skill_match"]
                + exp * weights["experience_alignment"]
                + dom * weights["domain_alignment"]
            )
            scored.append(
                {
                    **job,
                    "score": round(total, 4),
                    "score_breakdown": {
                        "skill_match": round(skill_score, 3),
                        "matched_skills": matched,
                        "experience_alignment": round(exp, 3),
                        "domain_alignment": dom,
                    },
                }
            )
        return sorted(scored, key=lambda x: x["score"], reverse=True)

    return filter_jobs, score_jobs


def _find_spans(spans, name):
    found = []
    for s in spans:
        if s.get("name") == name:
            found.append(s)
        found.extend(_find_spans(s.get("children") or [], name))
    return found


def _pdf_text(pdf_path: Path) -> str:
    try:
        return subprocess.check_output(
            ["pdftotext", str(pdf_path), "-"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def main() -> int:
    load_dotenv(ENV_FILE)
    reset_langfuse_client()
    client, model = build_llm_client()
    model = os.environ.get("OPENAI_MODEL") or model
    print(f"Using model: {model}")
    print(f"Langfuse enabled: {langfuse_enabled()}")
    if not langfuse_enabled():
        print(
            "ERROR: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set in .env. "
            "Cannot produce the required public submission trace URL."
        )
        return 1
    print("TOOLS registered:")
    for tool in TOOLS:
        print(f"  - {tool['name']}")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    MEMORY_PATH.write_text(json.dumps({"skills": [], "facts": []}, indent=2) + "\n")

    persona = json.loads(CANDIDATE_PERSONA.read_text())
    preferences = dict(persona["preferences"])
    master_skills = persona["master_skills"]
    portfolio = json.loads(PORTFOLIO_JSON.read_text())
    resume_tex = RESUME_TEX.read_text()
    memory = hc.load_memory(MEMORY_PATH)
    memory_ref = {"memory": memory}

    resume_summary = {
        "titles_held": [
            "Data Science Research Assistant",
            "Software Engineering Intern",
        ],
        "years_experience": 1.0,
        "skills": [
            "python",
            "pytorch",
            "sql",
            "fastapi",
            "docker",
            "mlflow",
            "scikit-learn",
            "pandas",
            "postgresql",
        ],
        "resume_projects": [
            "Resume–Job Matching Agent",
            "Manufacturing Defect Detection System",
            "Document Question-Answering System",
        ],
        "education": [
            {
                "institution": "Example University",
                "degree": "M.S. in Data Science",
            }
        ],
    }

    jobs = load_jobs(JOBS_CSV)
    print(f"Loaded {len(jobs)} real jobs from {JOBS_CSV.name}")

    nb = json.loads(NOTEBOOK_PATH.read_text())
    fit_ns = {
        "t3p": t3p,
        "json": json,
        "MODEL": model,
        "FIT_ANALYSIS_MODEL": model,
    }
    exec("".join(nb["cells"][26]["source"]), fit_ns)

    filter_jobs, score_jobs = make_filter_score_fns(
        resume_summary, portfolio, master_skills, memory_ref
    )

    # Extra approve tokens in case the LLM re-enters review after a nudge.
    inputs = iter(
        [
            f"reject: add {HITL_SKILL}, I know it",
            "approve",
            "approve",
            "approve",
            "approve",
            "approve",
        ]
    )

    def scripted_input(prompt: str = "") -> str:
        val = next(inputs)
        print(f"\n>>> SCRIPTED INPUT: {val!r}")
        return val

    ctx = AgentContext(
        client=client,
        model=model,
        resume_summary=resume_summary,
        portfolio=portfolio,
        master_skills=master_skills,
        memory=memory,
        resume_tex=resume_tex,
        output_dir=OUT,
        memory_path=MEMORY_PATH,
        preferences=preferences,
        filter_jobs_fn=filter_jobs,
        score_jobs_fn=score_jobs,
        fit_analysis_fn=fit_ns["fit_analysis"],
        input_fn=scripted_input,
        max_review_rounds=2,
    )

    user_message = (
        "Run the full job-search pipeline for this candidate against the real "
        "job board (ai_ml_jobs.csv).\n"
        f"Preferences: {json.dumps(preferences)}\n"
        "Filter with those preferences, score the survivors, fit-analyze each "
        "Top-3 rank, tailor ranks 1→2→3 in order (human review pauses after each "
        "tailor — do not ask the user questions; the review console handles "
        "feedback), then generate a cover letter for each approved rank. "
        "Complete every stage end-to-end without confirmation prompts."
    )

    with root_trace(
        "job_search_pipeline",
        metadata={
            "model": model,
            "mode": "llm_tool_selection",
            "data": "ai_ml_jobs.csv + candidate_persona.json",
        },
    ) as root:
        answer, state, agent_trace = run_agent(
            user_message,
            jobs,
            ctx=ctx,
            max_turns=40,
            verbose=True,
        )
        memory_ref["memory"] = ctx.memory

    tree = json.loads(format_trace_summary(root))
    (OUT / "agent_trace.json").write_text(
        json.dumps(agent_trace, indent=2, default=str) + "\n"
    )
    (OUT / "span_tree.json").write_text(format_trace_summary(root) + "\n")
    public_url = root.public_url
    (OUT / "langfuse_public_url.txt").write_text((public_url or "") + "\n")
    print("\n" + "#" * 72)
    print("LANGFUSE PUBLIC TRACE URL")
    print("#" * 72)
    print(public_url or "(missing — check Langfuse keys / SDK)")
    assert public_url, "Langfuse public_url was not captured on the root trace"
    assert "langfuse.com" in public_url or "langfuse" in public_url.lower()
    assert "traces/" in public_url or "/trace/" in public_url

    print("\n" + "#" * 72)
    print("LLM TOOL SELECTION TRACE")
    print("#" * 72)
    selected = [s["tool"] for s in agent_trace if s.get("type") == "tool_call"]
    for step in agent_trace:
        if step.get("type") == "reasoning":
            print(f"\n[turn {step['turn']}] reasoning: {step['content'][:250]}")
        elif step.get("type") == "tool_call":
            print(
                f"[turn {step['turn']}] LLM → {step['tool']} "
                f"{json.dumps(step['input'], default=str)[:200]}"
            )
        elif step.get("type") == "tool_result":
            out = step.get("output") or {}
            brief = {
                k: out.get(k)
                for k in (
                    "kept_count",
                    "rejected_count",
                    "scored_count",
                    "top_3",
                    "rank",
                    "review_status",
                    "job_title",
                    "company",
                    "memory_skills_after_review",
                    "pdf_path",
                    "error",
                )
                if k in out
            }
            print(f"[turn {step['turn']}] result ← {step['tool']} {brief}")
    print("\nLLM-selected tool sequence:", selected)

    # ---- Assertions ----
    print("\n" + "#" * 72)
    print("REAL-DATA ASSERTIONS")
    print("#" * 72)

    filter_results = [
        s
        for s in agent_trace
        if s.get("type") == "tool_result" and s.get("tool") == "filter_jobs"
    ]
    assert filter_results, "filter_jobs never called"
    filt = filter_results[0]["output"]
    print(f"filter kept={filt.get('kept_count')} rejected={filt.get('rejected_count')}")
    assert filt.get("kept_count") == 6, filt
    assert filt.get("rejected_count") == 15, filt

    rejected = filt.get("rejected") or []
    oxy = [r for r in rejected if "Occidental Petroleum (Oxy)" in (r.get("company") or "")]
    baylor = [r for r in rejected if "Baylor" in (r.get("company") or "")]
    assert len(oxy) == 2, f"expected 2 Oxy rejects, got {oxy}"
    assert len(baylor) == 1, f"expected 1 Baylor reject, got {baylor}"
    print(f"Oxy rejected: {len(oxy)}; Baylor rejected: {len(baylor)}")
    for r in oxy + baylor:
        print(f"  ✗ {r['company']} — {r['job_title']}: {r['reasons']}")

    kept_preview = filt.get("kept_preview") or []
    # kept_preview may be truncated to 10; also check state after filter via score
    score_results = [
        s
        for s in agent_trace
        if s.get("type") == "tool_result" and s.get("tool") == "score_jobs"
    ]
    assert score_results, "score_jobs never called"
    scored_out = score_results[0]["output"]
    top3 = scored_out.get("top_3") or []
    print(f"score_jobs scored_count={scored_out.get('scored_count')}")
    assert scored_out.get("scored_count") == 6, scored_out

    top3_companies = [t.get("company") for t in top3]
    print("Top-3 from score_jobs:")
    for t in top3:
        print(
            f"  #{t.get('rank')} score={t.get('score')} "
            f"{t.get('company')} — {t.get('job_title')}"
        )
        print(f"      breakdown={t.get('score_breakdown')}")
    assert top3_companies == EXPECTED_TOP3_COMPANIES, (
        f"Top-3 companies {top3_companies} != {EXPECTED_TOP3_COMPANIES}"
    )

    # Approved tailored artifacts for the three real companies
    approved_companies = {
        (a.get("company") or a.get("job", {}).get("company"))
        for a in state.approved.values()
    }
    print(f"\nApproved companies: {sorted(approved_companies)}")
    for company in EXPECTED_TOP3_COMPANIES:
        assert company in approved_companies, f"missing approved artifact for {company}"

    # HITL skill landed in rank-1 tex + PDF
    rank1 = state.approved[1]
    tex_path = Path(rank1["tex_path"])
    pdf_path = Path(rank1["pdf_path"])
    clog = json.loads(Path(rank1["change_log_path"]).read_text())
    tex = tex_path.read_text()
    pdf_txt = _pdf_text(pdf_path)
    skill_edits = clog.get("skill_edits") or {}
    after = skill_edits.get("after") or {}
    changes = skill_edits.get("changes") or []

    print(f"\nHITL skill={HITL_SKILL!r} on rank 1 ({rank1.get('company')}):")
    print(f"  in resume.tex Skills? {HITL_SKILL in tex}")
    print(f"  in resume.pdf text? {HITL_SKILL in pdf_txt}")
    print(f"  change_log changes: {changes}")
    print(f"  change_log after Systems: {after.get('Systems & Tools') or after}")
    assert HITL_SKILL in tex, "LangChain missing from resume.tex after rework"
    assert HITL_SKILL in pdf_txt, "LangChain missing from resume.pdf after rework"
    assert any(
        isinstance(c, dict) and c.get("skill") == HITL_SKILL and c.get("action") == "add"
        for c in changes
    ), changes
    assert any(
        HITL_SKILL in (v if isinstance(v, list) else [])
        for v in after.values()
    ), after

    memory_spans = _find_spans(tree.get("spans") or [], "memory_update")
    rework_spans = _find_spans(tree.get("spans") or [], "resume_rework")
    assert memory_spans, "no memory_update span"
    assert rework_spans, "no resume_rework span"
    print(f"\nmemory_update spans: {len(memory_spans)}")
    print(f"resume_rework spans: {len(rework_spans)}")
    print(f"  rework feedback: {(rework_spans[0].get('input') or {}).get('feedback')!r}")

    required = {
        "filter_jobs",
        "score_jobs",
        "fit_analysis",
        "tailor_resume",
        "generate_cover_letter",
    }
    assert required.issubset(set(selected)), (
        f"LLM missed tools: {required - set(selected)}"
    )
    assert selected.count("fit_analysis") >= 3
    assert selected.count("tailor_resume") >= 3
    assert selected.count("generate_cover_letter") >= 3

    # Artifact inventory for submission
    print("\n" + "#" * 72)
    print("SUBMISSION ARTIFACTS (real Top-3)")
    print("#" * 72)
    for rank in (1, 2, 3):
        a = state.approved[rank]
        folder = Path(a["folder"])
        print(f"\n#{rank} {a.get('company')} — {a.get('title')}")
        for name in (
            "resume.tex",
            "resume.pdf",
            "change_log.json",
            "cover_letter.tex",
            "cover_letter.pdf",
        ):
            p = folder / name
            print(f"  {'OK' if p.exists() else 'MISSING'}  {p}  ({p.stat().st_size if p.exists() else 0} bytes)")

    print(f"\nFull nested trace: {OUT / 'span_tree.json'}")
    print(f"Agent tool trace: {OUT / 'agent_trace.json'}")
    print("\nASSERTIONS PASSED")
    print("Final answer preview:", (answer or "")[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
