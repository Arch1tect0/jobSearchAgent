"""
Single-agent tool loop for the job search pipeline.

Tools (LLM-selected):
  filter_jobs, score_jobs, fit_analysis, tailor_resume, generate_cover_letter

Human review is NOT a tool — after each successful tailor_resume call the loop
hard-interrupts for console approve/reject (with memory writes + optional rework)
before the tool result is returned to the LLM. That preserves interleaved
memory propagation: rank N is reviewed before rank N+1 is tailored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import json

import hitl_cover as hc
import pipeline_tracing as tracing
import top3_resume_pipeline as t3p


TOOLS: list[dict[str, Any]] = [
    {
        "name": "filter_jobs",
        "description": (
            "Filter the job list using the candidate's preferences: location, "
            "experience range, excluded companies, remote-only. Returns kept "
            "jobs and rejected jobs with reasons."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "preferred_locations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "min_experience_years": {"type": "number"},
                "max_experience_years": {"type": "number"},
                "excluded_companies": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "remote_only": {"type": "boolean"},
            },
            "required": [],
        },
    },
    {
        "name": "score_jobs",
        "description": (
            "Deterministically score the currently kept jobs against the "
            "candidate's full profile (resume, portfolio, master skills, memory). "
            "Returns jobs sorted by score descending with a breakdown per job. "
            "After scoring, work the Top 3 by calling fit_analysis for each rank."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_years": {
                    "type": "number",
                    "description": (
                        "Override years of experience if not using the resume default."
                    ),
                }
            },
            "required": [],
        },
    },
    {
        "name": "fit_analysis",
        "description": (
            "Run evidence-grounded fit analysis for one Top-3 job by rank "
            "(1, 2, or 3). Call once per Top-3 job after scoring."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rank": {
                    "type": "integer",
                    "description": "1-based rank within the current Top-3 list.",
                    "minimum": 1,
                    "maximum": 3,
                }
            },
            "required": ["rank"],
        },
    },
    {
        "name": "tailor_resume",
        "description": (
            "Tailor the master resume for one Top-3 job by rank (1, 2, or 3). "
            "Uses current memory evidence. After the tool runs, the pipeline "
            "hard-pauses for human review of that resume before you receive "
            "the tool result — so call ranks in order (1 then 2 then 3) and do "
            "not tailor the next rank until the previous review completes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rank": {
                    "type": "integer",
                    "description": "1-based rank within the current Top-3 list.",
                    "minimum": 1,
                    "maximum": 3,
                }
            },
            "required": ["rank"],
        },
    },
    {
        "name": "generate_cover_letter",
        "description": (
            "Generate a one-page PDF cover letter for one already-approved "
            "Top-3 resume by rank. Only call after that rank has been approved "
            "via the human-review pause that follows tailor_resume."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rank": {
                    "type": "integer",
                    "description": "1-based rank of an approved tailored resume.",
                    "minimum": 1,
                    "maximum": 3,
                }
            },
            "required": ["rank"],
        },
    },
]


DEFAULT_SYSTEM_PROMPT = """
You are a single job-search agent with tools. Drive the full pipeline yourself
by choosing tools — do not ask the user to run steps for you.

Required order:
1) filter_jobs using the candidate preferences provided in the user message
2) score_jobs on the filtered list
3) fit_analysis once for each Top-3 rank (1, then 2, then 3)
4) tailor_resume for rank 1, then 2, then 3 IN ORDER
   (after each tailor_resume the runtime hard-pauses for human review and may
   update memory before your next tool call — later ranks must see that memory)
5) generate_cover_letter for each approved rank (1, then 2, then 3)
6) Briefly summarize deliverables and stop

Rules:
- Never ask clarifying or confirmation questions (e.g. "Should I proceed?").
- After any successful tailor_resume / fit_analysis / cover letter, immediately
  call the next required tool until the full pipeline is complete.
- If a tool returns an error, retry once if sensible, otherwise continue with
  the next required stage.

Before each tool call, state one short sentence of reasoning.
""".strip()


@dataclass
class AgentContext:
    client: Any
    model: str
    resume_summary: dict[str, Any]
    portfolio: dict[str, Any]
    master_skills: list[str]
    memory: dict[str, Any]
    resume_tex: str
    output_dir: Path
    memory_path: str | Path
    preferences: dict[str, Any]
    filter_jobs_fn: Callable[..., Any]
    score_jobs_fn: Callable[..., Any]
    fit_analysis_fn: Callable[..., Any]
    input_fn: Callable[[str], str] | None = None
    max_review_rounds: int = 2


@dataclass
class AgentState:
    jobs: list[dict[str, Any]]
    fit_analyses: dict[int, str] = field(default_factory=dict)
    fit_structured: dict[int, dict[str, Any]] = field(default_factory=dict)
    tailored: dict[int, dict[str, Any]] = field(default_factory=dict)
    approved: dict[int, dict[str, Any]] = field(default_factory=dict)
    cover_letters: dict[int, dict[str, Any]] = field(default_factory=dict)
    rejected_log: list[dict[str, Any]] = field(default_factory=list)


def top3_jobs(state: AgentState) -> list[dict[str, Any]]:
    return list(state.jobs[:3])


def _require_rank(rank: Any) -> int:
    value = int(rank)
    if value < 1 or value > 3:
        raise ValueError(f"rank must be 1..3, got {value}")
    return value


def _job_for_rank(state: AgentState, rank: int) -> dict[str, Any]:
    jobs = top3_jobs(state)
    if len(jobs) < rank:
        raise ValueError(
            f"Top-3 only has {len(jobs)} job(s); cannot address rank={rank}."
        )
    return jobs[rank - 1]


def dispatch_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    ctx: AgentContext,
    state: AgentState,
) -> dict[str, Any]:
    with tracing.trace_span(
        tool_name,
        input={"tool_input": tool_input},
        metadata={"tool": tool_name},
    ) as span:
        if tool_name == "filter_jobs":
            params = {
                "preferred_locations": tool_input.get(
                    "preferred_locations",
                    ctx.preferences.get("preferred_locations"),
                ),
                "min_experience_years": tool_input.get(
                    "min_experience_years",
                    ctx.preferences.get("min_experience_years", 0),
                ),
                "max_experience_years": tool_input.get(
                    "max_experience_years",
                    ctx.preferences.get("max_experience_years"),
                ),
                "excluded_companies": tool_input.get(
                    "excluded_companies",
                    ctx.preferences.get("excluded_companies"),
                ),
                "remote_only": tool_input.get(
                    "remote_only",
                    ctx.preferences.get("remote_only", False),
                ),
            }
            kept, rejected = ctx.filter_jobs_fn(state.jobs, **params)
            state.jobs = kept
            state.rejected_log = rejected
            payload = {
                "kept_count": len(kept),
                "rejected_count": len(rejected),
                "rejected": rejected,
                "kept_preview": [
                    {
                        "job_title": j.get("job_title"),
                        "company": j.get("company"),
                        "location": j.get("location"),
                    }
                    for j in kept[:10]
                ],
            }

        elif tool_name == "score_jobs":
            kwargs = {}
            if "candidate_years" in tool_input:
                kwargs["candidate_years"] = tool_input["candidate_years"]
            scored = ctx.score_jobs_fn(state.jobs, **kwargs)
            state.jobs = scored
            payload = {
                "scored_count": len(scored),
                "top_3": [
                    {
                        "rank": i,
                        "job_title": j.get("job_title"),
                        "company": j.get("company"),
                        "score": j.get("score"),
                        "score_breakdown": j.get("score_breakdown"),
                    }
                    for i, j in enumerate(scored[:3], start=1)
                ],
            }

        elif tool_name == "fit_analysis":
            rank = _require_rank(tool_input.get("rank"))
            job = _job_for_rank(state, rank)
            raw = ctx.fit_analysis_fn(
                job,
                ctx.resume_summary,
                ctx.portfolio,
                ctx.master_skills,
                ctx.memory,
                client=ctx.client,
                model=ctx.model,
            )
            if isinstance(raw, dict):
                text = str(raw.get("text") or "")
                structured = raw.get("structured")
                if not isinstance(structured, dict):
                    # Fall back to notebook cache shape if present.
                    structured = raw
            else:
                text = str(raw)
                structured = None
                # Notebook FIT_ANALYSIS_CACHE (same globals as the bound fn).
                cache = getattr(ctx.fit_analysis_fn, "__globals__", {}).get(
                    "FIT_ANALYSIS_CACHE"
                )
                if isinstance(cache, dict):
                    cached = cache.get(
                        (job.get("job_title"), job.get("company"))
                    )
                    if isinstance(cached, dict):
                        structured = cached.get("structured")
            state.fit_analyses[rank] = text
            if isinstance(structured, dict):
                state.fit_structured[rank] = structured
            payload = {
                "rank": rank,
                "job_title": job.get("job_title"),
                "company": job.get("company"),
                "fit_analysis_text": text,
                "fit_projects_verdict": (
                    (structured or {}).get("projects", {}) or {}
                ).get("verdict")
                if isinstance(structured, dict)
                else None,
            }

        elif tool_name == "tailor_resume":
            rank = _require_rank(tool_input.get("rank"))
            job = _job_for_rank(state, rank)

            # Enforce interleaved order: prior ranks must already be approved.
            for prior in range(1, rank):
                if prior not in state.approved:
                    raise ValueError(
                        f"Cannot tailor rank {rank} before rank {prior} is "
                        "approved via the human-review pause."
                    )

            # Refresh memory into Dale's pipeline before building the edit plan.
            t3p.configure_pipeline(
                client=ctx.client,
                resume_summary=ctx.resume_summary,
                master_skills=ctx.master_skills,
                memory_evidence=hc.memory_evidence_strings(ctx.memory),
            )
            ctx.output_dir.mkdir(parents=True, exist_ok=True)

            fit_structured = state.fit_structured.get(rank) or {}
            fit_projects = (
                fit_structured.get("projects")
                if isinstance(fit_structured, dict)
                else None
            )

            with tracing.trace_span(
                "process_top_job",
                input={
                    "rank": rank,
                    "job_title": job.get("job_title"),
                    "company": job.get("company"),
                    "memory_skills": list(ctx.memory.get("skills") or []),
                    "memory_evidence": hc.memory_evidence_strings(ctx.memory),
                    "fit_projects_verdict": (
                        fit_projects.get("verdict")
                        if isinstance(fit_projects, dict)
                        else None
                    ),
                },
            ) as tailor_span:
                result = t3p.process_top_job(
                    job=job,
                    rank=rank,
                    master_resume_tex=ctx.resume_tex,
                    portfolio=ctx.portfolio,
                    output_dir=ctx.output_dir,
                    model=ctx.model,
                    fit_projects=fit_projects
                    if isinstance(fit_projects, dict)
                    else None,
                )
                tailor_span.output = {
                    "pdf_path": result.get("pdf_path"),
                    "change_log_path": result.get("change_log_path"),
                }
            state.tailored[rank] = result

            # ---- HARD INTERRUPT (not an LLM tool) ----
            print("\n" + "!" * 72)
            print(
                f"HUMAN REVIEW INTERRUPT after tailor_resume(rank={rank}) "
                f"— not an LLM tool call"
            )
            print("!" * 72)
            approved_one, rounds = hc.review_one_resume(
                result,
                job=job,
                master_resume_tex=ctx.resume_tex,
                portfolio=ctx.portfolio,
                resume_summary=ctx.resume_summary,
                master_skills=ctx.master_skills,
                memory=ctx.memory,
                client=ctx.client,
                model=ctx.model,
                output_dir=ctx.output_dir,
                memory_path=ctx.memory_path,
                max_rounds=ctx.max_review_rounds,
                input_fn=ctx.input_fn,
                fit_projects=fit_projects
                if isinstance(fit_projects, dict)
                else None,
            )
            # Reload memory from disk in case review wrote facts.
            ctx.memory = hc.load_memory(ctx.memory_path)
            hc.refresh_pipeline_memory(ctx.memory)
            state.approved[rank] = approved_one

            change_log = approved_one.get("change_log") or hc.load_change_log(
                approved_one
            )
            payload = {
                "rank": rank,
                "job_title": job.get("job_title"),
                "company": job.get("company"),
                "review_status": approved_one.get("review_status"),
                "pdf_path": approved_one.get("pdf_path"),
                "change_log_path": approved_one.get("change_log_path"),
                "change_log_skill_changes": (
                    (change_log.get("skill_edits") or {}).get("changes") or []
                ),
                "memory_skills_after_review": list(ctx.memory.get("skills") or []),
                "review_rounds": rounds,
                "note": (
                    "Human review interrupt completed before this tool result "
                    "was returned to the LLM."
                ),
            }

        elif tool_name == "generate_cover_letter":
            rank = _require_rank(tool_input.get("rank"))
            if rank not in state.approved:
                raise ValueError(
                    f"Rank {rank} is not approved yet; cannot generate cover letter."
                )
            approved = state.approved[rank]
            job = approved.get("job") or _job_for_rank(state, rank)
            folder = Path(approved.get("folder") or ctx.output_dir)
            out_pdf = folder / "cover_letter.pdf"
            letter = hc.generate_cover_letter(
                job=job,
                resume_summary=ctx.resume_summary,
                output_path=out_pdf,
                portfolio=ctx.portfolio,
                master_skills=ctx.master_skills,
                memory=ctx.memory,
                tailored_tex_path=approved.get("tex_path"),
                master_resume_tex=ctx.resume_tex,
                client=ctx.client,
                model=ctx.model,
            )
            state.cover_letters[rank] = letter
            payload = {
                "rank": rank,
                "pdf_path": letter.get("pdf_path"),
                "job_title": letter.get("job_title"),
                "company": letter.get("company"),
                "skills_line": letter.get("letter", {}).get("skills_line")
                if isinstance(letter.get("letter"), dict)
                else letter.get("skills_line"),
            }

        else:
            payload = {"error": f"unknown tool {tool_name}"}

        span.output = payload
        return payload


def run_agent(
    user_message: str,
    jobs_list: list[dict[str, Any]],
    *,
    ctx: AgentContext,
    system_prompt: str | None = None,
    max_turns: int = 24,
    verbose: bool = True,
    state: AgentState | None = None,
    messages: list[dict[str, Any]] | None = None,
    trace: list[dict[str, Any]] | None = None,
) -> tuple[str, AgentState, list[dict[str, Any]]]:
    """
    Single LLM reasoning loop. The model selects tools; human review is injected
    as a hard interrupt inside tailor_resume dispatch.
    """
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    if messages is None:
        messages = [{"role": "user", "content": user_message}]
    if state is None:
        state = AgentState(jobs=list(jobs_list))
    if trace is None:
        trace = []
    nudge_budget = 4

    with tracing.trace_span(
        "run_agent",
        input={"message_preview": user_message[:500], "job_count": len(jobs_list)},
        metadata={"model": ctx.model},
    ) as root_span:
        for turn in range(max_turns):
            response = ctx.client.messages.create(
                model=ctx.model,
                max_tokens=2000,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            for block in response.content:
                if getattr(block, "type", None) == "text" and (
                    getattr(block, "text", "") or ""
                ).strip():
                    trace.append(
                        {
                            "turn": turn,
                            "type": "reasoning",
                            "content": block.text,
                        }
                    )
                    if verbose:
                        print(f"[turn {turn}] reasoning: {block.text}\n")

            if getattr(response, "stop_reason", None) != "tool_use":
                incomplete = (
                    len(state.approved) < min(3, max(len(top3_jobs(state)), 1))
                    or len(state.cover_letters) < len(state.approved)
                )
                if incomplete and nudge_budget > 0 and turn < max_turns - 1:
                    nudge_budget -= 1
                    next_steps: list[str] = []
                    for rank in (1, 2, 3):
                        if rank not in state.fit_analyses:
                            next_steps.append(f"fit_analysis(rank={rank})")
                    for rank in (1, 2, 3):
                        if rank not in state.approved:
                            next_steps.append(f"tailor_resume(rank={rank})")
                            break
                    for rank in sorted(state.approved):
                        if rank not in state.cover_letters:
                            next_steps.append(f"generate_cover_letter(rank={rank})")
                    nudge = (
                        "Do not ask questions. Continue the required pipeline now. "
                        "Next required tool call(s): "
                        + (
                            ", ".join(next_steps[:3])
                            if next_steps
                            else "generate remaining cover letters"
                        )
                        + "."
                    )
                    messages.append({"role": "user", "content": nudge})
                    if verbose:
                        print(f"[turn {turn}] NUDGE (pipeline incomplete): {nudge}\n")
                    continue

                final_text = "".join(
                    getattr(b, "text", "")
                    for b in response.content
                    if getattr(b, "type", None) == "text"
                )
                root_span.output = {
                    "final_text_preview": final_text[:500],
                    "approved_ranks": sorted(state.approved),
                    "cover_letter_ranks": sorted(state.cover_letters),
                    "tool_trace_summary": [
                        step
                        for step in trace
                        if step.get("type") in {"tool_call", "tool_result"}
                    ],
                }
                return final_text, state, trace

            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue

                tool_name = block.name
                tool_input = dict(block.input or {})
                trace.append(
                    {
                        "turn": turn,
                        "type": "tool_call",
                        "tool": tool_name,
                        "input": tool_input,
                    }
                )
                if verbose:
                    print(f"[turn {turn}] LLM selected tool: {tool_name}({tool_input})")

                try:
                    result_payload = dispatch_tool(
                        tool_name,
                        tool_input,
                        ctx=ctx,
                        state=state,
                    )
                except Exception as exc:  # noqa: BLE001 — return to LLM
                    result_payload = {"error": str(exc)}

                trace.append(
                    {
                        "turn": turn,
                        "type": "tool_result",
                        "tool": tool_name,
                        "output": result_payload,
                    }
                )
                if verbose:
                    preview = json.dumps(result_payload, default=str)
                    print(
                        f"[turn {turn}] tool_result {tool_name}: "
                        f"{preview[:500]}{'...' if len(preview) > 500 else ''}\n"
                    )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result_payload, default=str),
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        root_span.output = {
            "error": "max_turns_exceeded",
            "approved_ranks": sorted(state.approved),
            "cover_letter_ranks": sorted(state.cover_letters),
        }
        return "Max turns reached without a final answer.", state, trace
