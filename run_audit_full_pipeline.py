#!/usr/bin/env python3
"""
Full-pipeline audit run under ONE root trace.

Mirrors notebook run_full_pipeline stages:
  filter → score → fit(top3) → interleaved tailor+review → cover letters

Uses deterministic filter/score (same functions as the agent tools) because
OpenAI compat client does not implement Anthropic tool-use, and Anthropic
API credits are unavailable. Fit/tailor/review/cover use real LLM calls.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

from llm_client import build_llm_client
import hitl_cover as hc
import top3_resume_pipeline as t3p
from pipeline_tracing import root_trace, trace_span, format_trace_summary


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "tailored_resumes" / "_audit_full_pipeline"
MEMORY_PATH = OUT / "memory.json"


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


def load_jobs(csv_path: Path) -> list[dict]:
    _LOCATION_CITY_RE = re.compile(r"^\s*([A-Za-z][A-Za-z .]+?,\s*[A-Z]{2})\b")

    def parse_location(raw: str):
        raw = raw or ""
        m = _LOCATION_CITY_RE.search(raw)
        city = m.group(1).strip() if m else raw.strip()
        is_remote = bool(re.search(r"\bremote\b", raw, re.I))
        return city, is_remote

    def parse_years(raw: str) -> int:
        nums = [int(x) for x in re.findall(r"\d+", raw or "")]
        return nums[0] if nums else 0

    def parse_skills(raw: str) -> list[str]:
        parts = re.split(r"[;,/]|\band\b|\bor\b", raw or "", flags=re.I)
        out = []
        for p in parts:
            p = re.sub(r"\(.*?\)", "", p)
            p = " ".join(p.split()).strip().lower()
            if 1 < len(p) < 40:
                out.append(p)
        return list(dict.fromkeys(out))[:30]

    jobs = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            loc, rem = parse_location(row["Location"])
            jobs.append(
                {
                    "job_title": row["Job Title"].strip(),
                    "company": row["Company"].strip(),
                    "industry_domain": row["Industry/Domain"].strip(),
                    "location": loc,
                    "required_skills": parse_skills(row["Required Skills"]),
                    "years_experience_required": parse_years(
                        row["Years of Experience Required"]
                    ),
                    "remote": rem,
                    "job_description": row["Job Description"].strip(),
                    "company_details": row["Company Details"].strip(),
                    "url": row["URL"].strip(),
                }
            )
    return jobs


def filter_jobs(jobs_list, preferred_locations=None, min_experience_years=0,
                max_experience_years=None, excluded_companies=None, remote_only=False):
    preferred_locations = preferred_locations or []
    excluded_companies = [c.lower() for c in (excluded_companies or [])]
    kept, rejected = [], []
    for job in jobs_list:
        reasons = []
        if remote_only and not job.get("remote", False):
            reasons.append("remote-only preference set, job is not remote")
        if not remote_only and preferred_locations:
            loc_ok = job.get("location") in preferred_locations or job.get("remote", False)
            if not loc_ok:
                reasons.append(f"location '{job.get('location')}' not in preferred list")
        req_years = job.get("years_experience_required", 0)
        if req_years < min_experience_years:
            reasons.append(
                f"requires {req_years}y, below candidate minimum {min_experience_years}y"
            )
        if max_experience_years is not None and req_years > max_experience_years:
            reasons.append(
                f"requires {req_years}y, above candidate's {max_experience_years}y ceiling"
            )
        if job.get("company", "").lower() in excluded_companies:
            reasons.append(f"company '{job.get('company')}' is on the exclusion list")
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


def score_jobs(jobs_list, resume_summary, portfolio, master_skills, memory, weights=None):
    weights = weights or {
        "skill_match": 0.5,
        "experience_alignment": 0.25,
        "domain_alignment": 0.25,
    }
    candidate_skills = set(
        s.lower()
        for s in (
            master_skills
            + resume_summary["skills"]
            + list(memory.get("skills") or [])
        )
    )
    years = resume_summary["years_experience"]
    scored = []
    for job in jobs_list:
        required = set(s.lower() for s in job.get("required_skills", []))
        if not required:
            skill_score, matched = 1.0, []
        else:
            matched = sorted(required & candidate_skills)
            skill_score = len(matched) / len(required)
        req = job.get("years_experience_required", 0)
        exp = 1.0 if req == 0 else max(0.0, 1.0 - abs(years - req) / max(req, 1))
        from domain_scoring import score_domain_alignment

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


def main() -> int:
    load_dotenv(ROOT / ".env")
    client, model = build_llm_client()
    model = os.environ.get("OPENAI_MODEL") or model
    print(f"Using model: {model}")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    MEMORY_PATH.write_text(json.dumps({"skills": [], "facts": []}, indent=2) + "\n")

    persona = json.loads((ROOT / "candidate_persona.json").read_text())
    preferences = persona["preferences"]
    master_skills = persona["master_skills"]
    portfolio = json.loads((ROOT / "portfolio.json").read_text())
    resume_tex = (ROOT / "resume.tex").read_text()
    memory = hc.load_memory(MEMORY_PATH)
    resume_summary = {
        "titles_held": ["Data Science Research Assistant", "Software Engineering Intern"],
        "years_experience": 1.0,
        "skills": [
            "python", "pytorch", "sql", "fastapi", "docker", "mlflow",
            "scikit-learn", "pandas",
        ],
        "resume_projects": [
            "Resume–Job Matching Agent",
            "Manufacturing Defect Detection System",
            "Document Question-Answering System",
        ],
        "education": [
            {"institution": "Example University", "degree": "M.S. in Data Science"}
        ],
    }

    jobs = load_jobs(ROOT / "ai_ml_jobs.csv")

    # Load fit_analysis from notebook cell 26
    nb = json.loads((ROOT / "agent_notebook_v2.ipynb").read_text())
    fit_src = "".join(nb["cells"][26]["source"])
    fit_ns = {"t3p": t3p, "json": json, "MODEL": model, "FIT_ANALYSIS_MODEL": model}
    exec(fit_src, fit_ns)
    fit_analysis = fit_ns["fit_analysis"]

    inputs = iter(
        [
            "approve",
            "approve",
            "approve",
        ]
    )

    def scripted_input(prompt: str = "") -> str:
        val = next(inputs)
        print(f"\n>>> SCRIPTED INPUT: {val!r}")
        return val

    with root_trace(
        "job_search_pipeline",
        metadata={
            "model": model,
            "note": (
                "Audit full run. Filter/score invoked as deterministic tool "
                "functions (same code the agent tools call). Not notebook "
                "run_agent tool-loop — OpenAI compat lacks Anthropic tool-use; "
                "Anthropic credits unavailable."
            ),
        },
    ) as trace:
        with trace_span(
            "filter_jobs",
            input={"preferences": preferences, "job_count": len(jobs)},
        ) as filt_span:
            kept, rejected = filter_jobs(
                jobs,
                preferred_locations=preferences["preferred_locations"],
                min_experience_years=preferences["min_experience_years"],
                max_experience_years=preferences["max_experience_years"],
                excluded_companies=preferences["excluded_companies"],
                remote_only=preferences["remote_only"],
            )
            filt_span.output = {
                "kept_count": len(kept),
                "rejected_count": len(rejected),
                "rejected": rejected,
            }
            print(f"filter: kept={len(kept)} rejected={len(rejected)}")

        with trace_span(
            "score_jobs",
            input={"kept_count": len(kept)},
        ) as score_span:
            scored = score_jobs(
                kept, resume_summary, portfolio, master_skills, memory
            )
            top3 = scored[:3]
            score_span.output = {
                "scored_count": len(scored),
                "top3": [
                    {
                        "title": j["job_title"],
                        "company": j["company"],
                        "score": j["score"],
                        "breakdown": j["score_breakdown"],
                    }
                    for j in top3
                ],
            }
            print("top3:", score_span.output["top3"])

        with trace_span(
            "fit_analysis_batch",
            input={
                "top_3": [
                    {"title": j["job_title"], "company": j["company"]} for j in top3
                ]
            },
        ) as fit_span:
            fits = {}
            for job in top3:
                text = fit_analysis(
                    job,
                    resume_summary,
                    portfolio,
                    master_skills,
                    memory,
                    client=client,
                    model=model,
                )
                if isinstance(text, dict):
                    fits[(job["job_title"], job["company"])] = text.get("text") or ""
                    print("\n--- FIT ---\n", (text.get("text") or "")[:600], "\n...")
                else:
                    fits[(job["job_title"], job["company"])] = text
                    print("\n--- FIT ---\n", text[:600], "\n...")
            fit_span.output = {
                "count": len(fits),
                "titles": [k[0] for k in fits],
            }

        approved = hc.interleaved_tailor_and_review(
            top3,
            master_resume_tex=resume_tex,
            portfolio=portfolio,
            resume_summary=resume_summary,
            master_skills=master_skills,
            memory=memory,
            client=client,
            model=model,
            output_dir=OUT,
            memory_path=MEMORY_PATH,
            max_rounds=2,
            input_fn=scripted_input,
        )

        cover_letters = hc.generate_cover_letters_for_approved(
            approved,
            resume_summary=resume_summary,
            portfolio=portfolio,
            master_skills=master_skills,
            memory=hc.load_memory(MEMORY_PATH),
            client=client,
            model=model,
            master_resume_tex=resume_tex,
        )

        print("\ncover letters:", [c["pdf_path"] for c in cover_letters])

    tree = format_trace_summary(trace)
    (OUT / "full_span_tree.json").write_text(tree + "\n", encoding="utf-8")
    print("\n" + "#" * 72)
    print("FULL NESTED SPAN TREE")
    print("#" * 72)
    print(tree)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
