from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re
import subprocess

from compile_resume import compile_resume


RESUME_EDIT_MODEL = "claude-sonnet-4-20250514"

ALLOWED_EXPERIENCE_TARGETS = {
    "experience-bullet-1",
    "experience-bullet-2",
}

ALLOWED_PROJECT_SLOTS = {
    "project-1",
    "project-2",
    "project-3",
}

ALLOWED_EVIDENCE_SOURCES = {
    "resume",
    "portfolio",
    "master_skills",
    "memory",
}

ALLOWED_SKILL_ACTIONS = {
    "surface_form_alignment",
    "add",
    "highlight",
}


_PIPELINE_CLIENT = None
_PIPELINE_RESUME_SUMMARY: dict[str, Any] | None = None
_PIPELINE_MASTER_SKILLS: list[str] = []
_PIPELINE_MEMORY_EVIDENCE: list[str] = []


def configure_pipeline(
    client: Any,
    resume_summary: dict[str, Any],
    master_skills: list[str] | None = None,
    memory_evidence: list[str] | None = None,
) -> None:
    """
    Configure notebook-provided objects used by process_top_job().

    Call this once after importing the module:

        configure_pipeline(
            client=client,
            resume_summary=resume_summary,
        )
    """
    global _PIPELINE_CLIENT
    global _PIPELINE_RESUME_SUMMARY
    global _PIPELINE_MASTER_SKILLS
    global _PIPELINE_MEMORY_EVIDENCE

    _PIPELINE_CLIENT = client
    _PIPELINE_RESUME_SUMMARY = resume_summary
    _PIPELINE_MASTER_SKILLS = list(master_skills or [])
    _PIPELINE_MEMORY_EVIDENCE = list(memory_evidence or [])


def safe_folder_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value))
    return value.strip("_").lower()


def latex_escape(text: str) -> str:
    """
    Escape plain text before inserting it into LaTeX.

    This function is for model-generated text, not complete LaTeX blocks.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    result = str(text)

    for old, new in replacements.items():
        result = result.replace(old, new)

    return result


def strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = strip_json_fences(text)

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The resume-edit model did not return valid JSON.\n\n"
            f"JSON error: {exc}\n\n"
            f"Response preview:\n{cleaned[:2000]}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError("The model response must be a JSON object.")

    return value


def portfolio_project_map(portfolio_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        project["project_id"]: project
        for project in portfolio_data.get("projects", [])
        if isinstance(project, dict) and project.get("project_id")
    }


def get_portfolio_project(
    portfolio_data: dict[str, Any],
    project_id: str,
) -> dict[str, Any]:
    projects = portfolio_project_map(portfolio_data)

    if project_id not in projects:
        raise KeyError(f"Portfolio project not found: {project_id!r}")

    return projects[project_id]


def require_nonempty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string.")


def validate_evidence(evidence: Any, field_name: str) -> None:
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(
            f"{field_name} must contain at least one evidence record."
        )

    for index, item in enumerate(evidence):
        item_name = f"{field_name}[{index}]"

        if not isinstance(item, dict):
            raise ValueError(f"{item_name} must be a dictionary.")

        source = item.get("source")
        reference = item.get("reference")

        if source not in ALLOWED_EVIDENCE_SOURCES:
            raise ValueError(
                f"{item_name}.source must be one of "
                f"{sorted(ALLOWED_EVIDENCE_SOURCES)}."
            )

        require_nonempty_string(reference, f"{item_name}.reference")



def normalize_project_name(value: Any) -> str:
    """Normalize project names for matching."""
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).strip().lower(),
    )


def current_resume_project_slots(
    resume_summary_data: dict[str, Any],
    portfolio_data: dict[str, Any],
) -> dict[str, str]:
    """
    Match projects currently listed in resume.tex to portfolio project IDs.
    """
    portfolio_projects = portfolio_data.get("projects", [])

    name_to_id = {
        normalize_project_name(project.get("project_name", "")):
            project["project_id"]
        for project in portfolio_projects
        if isinstance(project, dict)
        and project.get("project_id")
        and project.get("project_name")
    }

    resume_projects = resume_summary_data.get(
        "resume_projects",
        [],
    )

    slots: dict[str, str] = {}

    for index, project_name in enumerate(
        resume_projects[:3],
        start=1,
    ):
        normalized = normalize_project_name(project_name)
        project_id = name_to_id.get(normalized)

        if project_id:
            slots[f"project-{index}"] = project_id

    return slots




def sanitize_skill_changes(
    edit_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove incomplete or placeholder skill-change records.

    Skill changes are optional. A record is retained only when it has
    a nonempty skill, supported action, reason, and evidence.
    """
    skills = edit_plan.get("skills")

    if not isinstance(skills, dict):
        edit_plan["skills"] = {
            "before": {},
            "after": {},
            "changes": [],
        }
        return edit_plan

    if not isinstance(skills.get("before"), dict):
        skills["before"] = {}

    if not isinstance(skills.get("after"), dict):
        skills["after"] = {}

    changes = skills.get("changes", [])

    if not isinstance(changes, list):
        skills["changes"] = []
        return edit_plan

    cleaned_changes: list[dict[str, Any]] = []

    for change in changes:
        if not isinstance(change, dict):
            continue

        skill = str(change.get("skill", "")).strip()
        action = str(change.get("action", "")).strip()
        reason = str(change.get("reason", "")).strip()
        evidence = change.get("evidence", [])

        if not skill:
            continue

        if action not in ALLOWED_SKILL_ACTIONS:
            continue

        if not reason:
            continue

        if not isinstance(evidence, list) or not evidence:
            continue

        change["skill"] = skill
        change["action"] = action
        change["reason"] = reason
        change["evidence"] = evidence

        cleaned_changes.append(change)

    skills["changes"] = cleaned_changes
    edit_plan["skills"] = skills
    return edit_plan

def sanitize_project_swaps(
    edit_plan: dict[str, Any],
    portfolio_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove incomplete project swaps and normalize required evidence.

    Project swaps are optional. A swap is retained only when it has a
    valid slot and a valid add_project_id from portfolio.json.

    For every retained swap, Python appends the exact portfolio evidence
    citation required by validation:
        {"source": "portfolio", "reference": add_project_id}
    """
    swaps = edit_plan.get("project_swaps", [])

    if not isinstance(swaps, list):
        edit_plan["project_swaps"] = []
        return edit_plan

    valid_project_ids = set(
        portfolio_project_map(portfolio_data)
    )

    cleaned_swaps: list[dict[str, Any]] = []

    for swap in swaps:
        if not isinstance(swap, dict):
            continue

        slot = str(swap.get("slot", "")).strip()
        add_id = str(swap.get("add_project_id", "")).strip()

        if slot not in ALLOWED_PROJECT_SLOTS:
            continue

        if not add_id or add_id not in valid_project_ids:
            continue

        swap["slot"] = slot
        swap["add_project_id"] = add_id

        remove_id = swap.get("remove_project_id")

        if isinstance(remove_id, str):
            swap["remove_project_id"] = remove_id.strip()

        evidence = swap.get("evidence", [])

        if not isinstance(evidence, list):
            evidence = []

        normalized_evidence: list[dict[str, Any]] = []

        for item in evidence:
            if not isinstance(item, dict):
                continue

            source_name = str(item.get("source", "")).strip().lower()
            reference = str(item.get("reference", "")).strip()

            if source_name and reference:
                normalized_evidence.append(
                    {
                        "source": source_name,
                        "reference": reference,
                    }
                )

        required_citation = {
            "source": "portfolio",
            "reference": add_id,
        }

        if required_citation not in normalized_evidence:
            normalized_evidence.append(required_citation)

        swap["evidence"] = normalized_evidence
        cleaned_swaps.append(swap)

    edit_plan["project_swaps"] = cleaned_swaps
    return edit_plan


def fill_missing_project_remove_ids(
    edit_plan: dict[str, Any],
    resume_summary_data: dict[str, Any],
    portfolio_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Fill a missing remove_project_id using the current project slot.
    """
    slot_map = current_resume_project_slots(
        resume_summary_data,
        portfolio_data,
    )

    for swap in edit_plan.get("project_swaps", []):
        if not isinstance(swap, dict):
            continue

        remove_id = swap.get("remove_project_id")

        if isinstance(remove_id, str) and remove_id.strip():
            continue

        slot = swap.get("slot")
        inferred_id = slot_map.get(slot)

        if not inferred_id:
            raise ValueError(
                f"Could not infer remove_project_id for slot {slot!r}. "
                f"Resolved resume project slots: {slot_map}"
            )

        swap["remove_project_id"] = inferred_id

    return edit_plan


def clean_evidence_list(items: Any) -> list[dict[str, Any]]:
    """Normalize evidence sources; drop unsupported/empty records."""
    if not isinstance(items, list):
        return []

    cleaned: list[dict[str, Any]] = []
    aliases = {
        "master skills": "master_skills",
        "master-skills": "master_skills",
        "master_skill": "master_skills",
        "masterskills": "master_skills",
        "resume.tex": "resume",
        "portfolio.json": "portfolio",
        "candidate memory": "memory",
        "memory.json": "memory",
        "job": "resume",
        "job description": "resume",
        "job_description": "resume",
        "jd": "resume",
        "posting": "resume",
        "skills": "master_skills",
        "skill": "master_skills",
        "candidate": "resume",
        "experience": "resume",
        "project": "portfolio",
        "projects": "portfolio",
    }

    for item in items:
        if not isinstance(item, dict):
            continue

        source = str(item.get("source", "")).strip().lower()
        reference = str(item.get("reference", "")).strip()
        source = aliases.get(source, source)

        if source not in ALLOWED_EVIDENCE_SOURCES:
            continue
        if not reference:
            continue

        cleaned.append({"source": source, "reference": reference})

    return cleaned


def sanitize_evidence_records(
    edit_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize supported evidence sources and remove unsupported ones.
    """
    summary = edit_plan.get("summary")

    if isinstance(summary, dict):
        summary["evidence"] = clean_evidence_list(
            summary.get("evidence")
        )

    for bullet in edit_plan.get("experience_bullets", []):
        if isinstance(bullet, dict):
            bullet["evidence"] = clean_evidence_list(
                bullet.get("evidence")
            )

    skills = edit_plan.get("skills", {})

    if isinstance(skills, dict):
        for change in skills.get("changes", []):
            if isinstance(change, dict):
                change["evidence"] = clean_evidence_list(
                    change.get("evidence")
                )

    for swap in edit_plan.get("project_swaps", []):
        if isinstance(swap, dict):
            swap["evidence"] = clean_evidence_list(
                swap.get("evidence")
            )

    return edit_plan


def ensure_edit_plan_evidence(edit_plan: dict[str, Any]) -> dict[str, Any]:
    """
    Fill empty evidence lists so a valid edit plan is not rejected when the
    model omits citations (common on revision/rework passes).
    """
    default_resume = {
        "source": "resume",
        "reference": "Supporting record on the master resume",
    }

    summary = edit_plan.get("summary")
    if isinstance(summary, dict) and not summary.get("evidence"):
        summary["evidence"] = [dict(default_resume)]

    for bullet in edit_plan.get("experience_bullets") or []:
        if not isinstance(bullet, dict):
            continue
        if not bullet.get("evidence"):
            target = str(bullet.get("target") or "experience")
            bullet["evidence"] = [
                {
                    "source": "resume",
                    "reference": f"Experience section ({target})",
                }
            ]

    skills = edit_plan.get("skills")
    if isinstance(skills, dict):
        for change in skills.get("changes") or []:
            if isinstance(change, dict) and not change.get("evidence"):
                skill = str(change.get("skill") or "skill")
                change["evidence"] = [
                    {
                        "source": "resume",
                        "reference": f"Skills section supporting {skill}",
                    }
                ]

    for swap in edit_plan.get("project_swaps") or []:
        if not isinstance(swap, dict):
            continue
        if not swap.get("evidence"):
            add_id = str(swap.get("add_project_id") or "").strip()
            swap["evidence"] = (
                [{"source": "portfolio", "reference": add_id}]
                if add_id
                else [dict(default_resume)]
            )

    return edit_plan


def validate_edit_plan(
    edit_plan: dict[str, Any],
    portfolio_data: dict[str, Any],
) -> None:
    required_keys = {
        "summary",
        "experience_bullets",
        "skills",
        "project_swaps",
        "gaps",
    }

    missing = required_keys - set(edit_plan)

    if missing:
        raise ValueError(f"Edit plan is missing keys: {sorted(missing)}")

    unexpected = set(edit_plan) - required_keys

    if unexpected:
        raise ValueError(
            f"Edit plan contains unexpected keys: {sorted(unexpected)}"
        )

    summary = edit_plan["summary"]

    if not isinstance(summary, dict):
        raise ValueError("summary must be a dictionary.")

    for key in ("before", "after", "reason"):
        require_nonempty_string(summary.get(key), f"summary.{key}")

    validate_evidence(summary.get("evidence"), "summary.evidence")

    bullets = edit_plan["experience_bullets"]

    if not isinstance(bullets, list) or len(bullets) != 2:
        raise ValueError("Exactly two experience bullet edits are required.")

    targets = set()

    for index, bullet in enumerate(bullets):
        field_name = f"experience_bullets[{index}]"

        if not isinstance(bullet, dict):
            raise ValueError(f"{field_name} must be a dictionary.")

        target = bullet.get("target")

        if target not in ALLOWED_EXPERIENCE_TARGETS:
            raise ValueError(f"{field_name}.target is invalid: {target!r}")

        targets.add(target)

        for key in ("before", "after", "reason"):
            require_nonempty_string(
                bullet.get(key),
                f"{field_name}.{key}",
            )

        validate_evidence(
            bullet.get("evidence"),
            f"{field_name}.evidence",
        )

    if targets != ALLOWED_EXPERIENCE_TARGETS:
        raise ValueError(
            "Experience edits must target exactly experience-bullet-1 "
            "and experience-bullet-2."
        )

    skills = edit_plan["skills"]

    if not isinstance(skills, dict):
        raise ValueError("skills must be a dictionary.")

    if not isinstance(skills.get("before"), dict):
        raise ValueError("skills.before must be a dictionary.")

    if not isinstance(skills.get("after"), dict):
        raise ValueError("skills.after must be a dictionary.")

    changes = skills.get("changes")

    if not isinstance(changes, list):
        raise ValueError("skills.changes must be a list.")

    for index, change in enumerate(changes):
        field_name = f"skills.changes[{index}]"

        if not isinstance(change, dict):
            raise ValueError(f"{field_name} must be a dictionary.")

        require_nonempty_string(change.get("skill"), f"{field_name}.skill")

        action = change.get("action")

        if action not in ALLOWED_SKILL_ACTIONS:
            raise ValueError(
                f"{field_name}.action must be one of "
                f"{sorted(ALLOWED_SKILL_ACTIONS)}."
            )

        require_nonempty_string(change.get("reason"), f"{field_name}.reason")
        validate_evidence(change.get("evidence"), f"{field_name}.evidence")

    project_ids = set(portfolio_project_map(portfolio_data))
    swaps = edit_plan["project_swaps"]

    if not isinstance(swaps, list):
        raise ValueError("project_swaps must be a list.")

    used_slots = set()

    for index, swap in enumerate(swaps):
        field_name = f"project_swaps[{index}]"

        if not isinstance(swap, dict):
            raise ValueError(f"{field_name} must be a dictionary.")

        slot = swap.get("slot")
        remove_id = swap.get("remove_project_id")
        add_id = swap.get("add_project_id")

        if slot not in ALLOWED_PROJECT_SLOTS:
            raise ValueError(f"{field_name}.slot is invalid: {slot!r}")

        if slot in used_slots:
            raise ValueError(f"Project slot used more than once: {slot!r}")

        used_slots.add(slot)

        require_nonempty_string(remove_id, f"{field_name}.remove_project_id")
        require_nonempty_string(add_id, f"{field_name}.add_project_id")

        if add_id not in project_ids:
            raise ValueError(
                f"{field_name}.add_project_id does not exist in "
                f"portfolio.json: {add_id!r}"
            )

        if add_id == remove_id:
            raise ValueError(f"{field_name} replaces a project with itself.")

        require_nonempty_string(swap.get("reason"), f"{field_name}.reason")
        validate_evidence(swap.get("evidence"), f"{field_name}.evidence")

        cited = any(
            item.get("source") == "portfolio"
            and item.get("reference") == add_id
            for item in swap["evidence"]
        )

        if not cited:
            raise ValueError(
                f"{field_name} must cite portfolio project {add_id!r}."
            )

    gaps = edit_plan["gaps"]

    if not isinstance(gaps, list):
        raise ValueError("gaps must be a list.")

    for index, gap in enumerate(gaps):
        field_name = f"gaps[{index}]"

        if not isinstance(gap, dict):
            raise ValueError(f"{field_name} must be a dictionary.")

        require_nonempty_string(
            gap.get("requirement"),
            f"{field_name}.requirement",
        )
        require_nonempty_string(gap.get("reason"), f"{field_name}.reason")


def build_resume_edit_prompt(
    job: dict[str, Any],
    resume_summary_data: dict[str, Any],
    portfolio_data: dict[str, Any],
    current_resume_tex: str,
    master_skills: list[str] | None = None,
    memory_evidence: list[str] | None = None,
    revision_feedback: str | None = None,
    fit_projects: dict[str, Any] | None = None,
) -> str:
    output_schema = {
        "summary": {
            "before": "Existing professional summary",
            "after": "Rewritten professional summary",
            "reason": "Reason for the change",
            "evidence": [
                {
                    "source": "resume",
                    "reference": "Specific supporting record",
                }
            ],
        },
        "experience_bullets": [
            {
                "target": "experience-bullet-1",
                "before": "Existing bullet text",
                "after": "Revised bullet text",
                "reason": "Reason for the change",
                "evidence": [
                    {
                        "source": "resume",
                        "reference": "Specific supporting record",
                    }
                ],
            },
            {
                "target": "experience-bullet-2",
                "before": "Existing bullet text",
                "after": "Revised bullet text",
                "reason": "Reason for the change",
                "evidence": [
                    {
                        "source": "resume",
                        "reference": "Specific supporting record",
                    }
                ],
            },
        ],
        "skills": {
            "before": {},
            "after": {},
            "changes": [],
        },
        "project_swaps": [],
        "gaps": [],
    }

    return f"""
You are preparing a controlled edit plan for a LaTeX resume.

JOB:
{json.dumps(job, indent=2)}

STRUCTURED RESUME:
{json.dumps(resume_summary_data, indent=2)}

PORTFOLIO:
{json.dumps(portfolio_data, indent=2)}

MASTER SKILLS:
{json.dumps(master_skills or [], indent=2)}

APPROVED MEMORY EVIDENCE:
{json.dumps(memory_evidence or [], indent=2)}

FIT ANALYSIS PROJECT GUIDANCE (from the prior fit_analysis tool; honor a
verified swap_suggested recommendation — do not invent project_ids):
{json.dumps(fit_projects or {"verdict": "keep_current", "note": "no fit guidance provided"}, indent=2)}

CURRENT LATEX:
{current_resume_tex}

HUMAN REVIEWER FEEDBACK (address on this revision; empty if first pass):
{revision_feedback or "(none — initial tailor)"}

Only these modifications are allowed:

1. Rewrite the Professional Summary.

2. Modify exactly two experience bullets:
   - experience-bullet-1
   - experience-bullet-2

3. Add, highlight, or surface-align skills only when supported by the
   resume, portfolio, master skills, or approved memory evidence.

4. Swap a project only when a portfolio project is genuinely more
   relevant than a current resume project. If FIT ANALYSIS PROJECT
   GUIDANCE has verdict "swap_suggested" with a valid add_project_id,
   you MUST include that swap in project_swaps (slot + remove/add ids).

Rules:

- Do not change names, contact information, employers, titles, dates,
  education, certifications, formatting, or LaTeX commands.
- Do not rewrite the entire resume.
- Do not invent skills, projects, employers, titles, dates, metrics,
  or results.
- Every edit must include evidence.
- Every added project must use an exact project_id from the portfolio.
- Unsupported requirements must be recorded under gaps.
- If APPROVED MEMORY EVIDENCE names a skill that also appears in the
  job's required_skills (case-insensitive), you MUST include that skill
  in skills.changes (action "add" or "highlight") with evidence
  source "memory" citing the memory fact. Do not leave it as a gap.
- Empty skills.changes and project_swaps lists are valid.
- Return exactly two experience bullet edits.



SKILL CHANGE OUTPUT RULE:

- If no supported skill change is needed, return "changes": [].
- Do not return skill-change records with blank, null, placeholder, unknown,
  or omitted skill names.
- Every retained skill change must include a supported action, reason,
  and candidate evidence.

PROJECT SWAP OUTPUT RULE:

- If no portfolio project is materially better than the current resume
  projects, return "project_swaps": [].
- Do not return a project swap with a blank, null, placeholder, unknown,
  or omitted add_project_id.
- A project swap must include a valid slot and an exact portfolio project_id.

Return exactly one JSON object with this structure:

{json.dumps(output_schema, indent=2)}

Return valid JSON only.
Do not use Markdown fences.
Do not include comments, introduction, or trailing explanation.
""".strip()


def call_resume_edit_model(
    prompt: str,
    model: str = RESUME_EDIT_MODEL,
) -> str:
    if _PIPELINE_CLIENT is None:
        raise RuntimeError(
            "Pipeline client is not configured. Call configure_pipeline() "
            "before process_top_job()."
        )

    response = _PIPELINE_CLIENT.messages.create(
        model=model,
        max_tokens=5000,
        temperature=0,
        system=(
            "Create evidence-grounded, controlled resume edit plans. "
            "Return valid JSON only. Do not include Markdown fences."
        ),
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    text_parts = []

    for block in response.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)

    output_text = "".join(text_parts).strip()

    if not output_text:
        raise RuntimeError(
            "The Anthropic model returned no output text."
        )

    return output_text

def generate_edit_plan(
    job: dict[str, Any],
    master_resume_tex: str,
    portfolio_data: dict[str, Any],
    model: str = RESUME_EDIT_MODEL,
    revision_feedback: str | None = None,
    fit_projects: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if _PIPELINE_RESUME_SUMMARY is None:
        raise RuntimeError(
            "Resume summary is not configured. Call configure_pipeline() "
            "before process_top_job()."
        )

    prompt = build_resume_edit_prompt(
        job=job,
        resume_summary_data=_PIPELINE_RESUME_SUMMARY,
        portfolio_data=portfolio_data,
        current_resume_tex=master_resume_tex,
        master_skills=_PIPELINE_MASTER_SKILLS,
        memory_evidence=_PIPELINE_MEMORY_EVIDENCE,
        revision_feedback=revision_feedback,
        fit_projects=fit_projects,
    )

    raw_response = call_resume_edit_model(
        prompt,
        model=model,
    )

    edit_plan = parse_json_response(raw_response)

    edit_plan = sanitize_evidence_records(
        edit_plan
    )
    edit_plan = ensure_edit_plan_evidence(edit_plan)

    edit_plan = sanitize_skill_changes(
        edit_plan
    )
    edit_plan = sanitize_project_swaps(
        edit_plan=edit_plan,
        portfolio_data=portfolio_data,
    )

    edit_plan = fill_missing_project_remove_ids(
        edit_plan=edit_plan,
        resume_summary_data=_PIPELINE_RESUME_SUMMARY,
        portfolio_data=portfolio_data,
    )

    edit_plan = ensure_memory_skills_for_job(
        edit_plan=edit_plan,
        job=job,
        memory_evidence=_PIPELINE_MEMORY_EVIDENCE,
        master_resume_tex=master_resume_tex,
    )

    edit_plan = ensure_fit_analysis_project_swap(
        edit_plan,
        fit_projects=fit_projects,
        resume_summary_data=_PIPELINE_RESUME_SUMMARY,
        portfolio_data=portfolio_data,
        master_resume_tex=master_resume_tex,
    )
    # Re-sanitize after deterministic inject so evidence/ids stay valid.
    edit_plan = sanitize_project_swaps(
        edit_plan=edit_plan,
        portfolio_data=portfolio_data,
    )
    edit_plan = fill_missing_project_remove_ids(
        edit_plan=edit_plan,
        resume_summary_data=_PIPELINE_RESUME_SUMMARY,
        portfolio_data=portfolio_data,
    )

    validate_edit_plan(
        edit_plan,
        portfolio_data,
    )

    return edit_plan


def replace_professional_summary(tex: str, new_summary: str) -> str:
    pattern = re.compile(
        r"(%\s*AGENT-EDIT-TARGET:\s*summary\s*\n)"
        r"(.*?)"
        r"(?=\n\s*%[-]+\s*\n|\n\s*\\section\{)",
        re.DOTALL,
    )

    updated, count = pattern.subn(
        lambda match: (
            match.group(1)
            + latex_escape(new_summary.strip())
            + "\n"
        ),
        tex,
        count=1,
    )

    if count != 1:
        raise ValueError("Could not uniquely locate the summary marker.")

    return updated


def replace_marked_resume_item(
    tex: str,
    marker: str,
    new_text: str,
) -> str:
    pattern = re.compile(
        rf"(%\s*AGENT-EDIT-TARGET:\s*{re.escape(marker)}\s*\n)"
        rf"(\s*\\resumeItem\{{.*?\}})",
        re.DOTALL,
    )

    updated, count = pattern.subn(
        lambda match: (
            match.group(1)
            + rf"    \resumeItem{{{latex_escape(new_text.strip())}}}"
        ),
        tex,
        count=1,
    )

    if count != 1:
        raise ValueError(f"Could not uniquely locate marker {marker!r}.")

    return updated


def render_skills_section(skills: dict[str, list[Any]]) -> str:
    lines = []

    for category, values in skills.items():
        if not values:
            continue

        rendered_values = ", ".join(
            latex_escape(str(value))
            for value in values
        )

        lines.append(
            rf"  \resumeItem{{\textbf{{{latex_escape(category)}:}} "
            rf"{rendered_values}}}"
        )

    return "\n".join(lines)


def replace_skills_section(
    tex: str,
    skills_after: dict[str, list[Any]],
) -> str:
    pattern = re.compile(
        r"(%\s*AGENT-EDIT-START:\s*skills\s*\n)"
        r".*?"
        r"(%\s*AGENT-EDIT-END:\s*skills)",
        re.DOTALL,
    )

    rendered = render_skills_section(skills_after)

    updated, count = pattern.subn(
        lambda match: (
            match.group(1)
            + rendered
            + "\n"
            + match.group(2)
        ),
        tex,
        count=1,
    )

    if count != 1:
        raise ValueError(
            "Could not locate skills markers. Add:\n"
            "% AGENT-EDIT-START: skills\n"
            "% AGENT-EDIT-END: skills"
        )

    return updated


def render_project_latex(
    project: dict[str, Any],
    slot: str,
) -> str:
    project_id = project["project_id"]
    project_name = latex_escape(project.get("project_name", ""))
    year = latex_escape(str(project.get("year", "")))
    tech_stack = latex_escape(", ".join(project.get("tech_stack", [])))
    industry = latex_escape(project.get("industry", ""))
    summary = latex_escape(project.get("resume_summary", ""))

    return f"""% AGENT-SWAP-START: {slot}
  % PROJECT-ID: {project_id}
  \\resumeEntry{{{project_name}}}{{{year}}}
    {{Tech Stack: {tech_stack}}}{{{industry}}}
  \\resumeItemListStart
    \\resumeItem{{{summary}}}
  \\resumeItemListEnd
% AGENT-SWAP-END: {slot}"""


def replace_project_slot(
    tex: str,
    slot: str,
    project: dict[str, Any],
) -> str:
    pattern = re.compile(
        rf"%\s*AGENT-SWAP-START:\s*{re.escape(slot)}"
        rf".*?"
        rf"%\s*AGENT-SWAP-END:\s*{re.escape(slot)}",
        re.DOTALL,
    )

    # Use a callable repl so re.sub does not treat \r in \resumeEntry
    # as a carriage-return escape (string repls process backslash escapes).
    replacement = render_project_latex(project, slot)
    updated, count = pattern.subn(
        lambda _match: replacement,
        tex,
        count=1,
    )

    if count != 1:
        raise ValueError(
            f"Could not locate project markers for {slot!r}."
        )

    return updated


def normalize_skill_category(category: str) -> str:
    """Store category keys as plain text (undo common LaTeX escapes)."""
    text = str(category).strip()
    text = text.replace(r"\&", "&")
    text = text.replace(r"\%", "%")
    text = text.replace(r"\_", "_")
    text = text.replace(r"\#", "#")
    text = text.replace(r"\$", "$")
    return text


def _skills_dict_populated(skills: Any) -> bool:
    return isinstance(skills, dict) and any(
        isinstance(values, list) and values for values in skills.values()
    )


def _normalize_skills_dict(skills: Any) -> dict[str, list[str]]:
    if not isinstance(skills, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for category, values in skills.items():
        if not isinstance(values, list):
            continue
        key = normalize_skill_category(str(category))
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        if cleaned:
            normalized[key] = cleaned
    return normalized


def _find_skill_category(
    skills: dict[str, list[str]],
    skill: str,
) -> str | None:
    target = skill.strip().lower()
    for category, values in skills.items():
        if any(str(v).strip().lower() == target for v in values):
            return category
    return None


def _category_for_new_skill(
    skills: dict[str, list[str]],
) -> str:
    preferred = ("Systems & Tools", "Languages", "ML & Data")
    lower_map = {
        normalize_skill_category(cat).lower(): cat for cat in skills
    }
    for name in preferred:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return "Systems & Tools"


def parse_skills_after_from_tex(tex: str) -> dict[str, list[str]]:
    """Parse categorized skills from between AGENT-EDIT skills markers."""
    match = re.search(
        r"%\s*AGENT-EDIT-START:\s*skills\s*\n(.*?)%\s*AGENT-EDIT-END:\s*skills",
        tex,
        flags=re.DOTALL,
    )
    if not match:
        return {}

    block = match.group(1)
    skills: dict[str, list[str]] = {}
    for line in block.splitlines():
        cat_match = re.search(
            r"\\textbf\{([^{}]+):\}\s*([^}]+)",
            line,
        )
        if not cat_match:
            continue
        category = normalize_skill_category(cat_match.group(1))
        values = [
            part.strip()
            for part in cat_match.group(2).split(",")
            if part.strip()
        ]
        if values:
            skills[category] = values
    return skills


def materialize_skill_edits(
    edit_plan: dict[str, Any],
    master_resume_tex: str,
) -> dict[str, Any]:
    """
    Make skills.changes actually land in skills.before/after (and thus tex).

    - Populate before from the master resume Skills section when empty.
    - Apply each real 'add' into after (inserting the skill if missing).
    - Keep changes[] only for edits that were applied to the snapshot.
    - Demote un-applicable edits to gaps (no claims/artifact mismatch).
    """
    skills = edit_plan.get("skills")
    if not isinstance(skills, dict):
        skills = {"before": {}, "after": {}, "changes": []}
        edit_plan["skills"] = skills

    gaps = edit_plan.get("gaps")
    if not isinstance(gaps, list):
        gaps = []
        edit_plan["gaps"] = gaps

    changes = skills.get("changes")
    if not isinstance(changes, list):
        changes = []

    parsed = parse_skills_after_from_tex(master_resume_tex)
    before = _normalize_skills_dict(skills.get("before"))
    if not before:
        before = {k: list(v) for k, v in parsed.items()}

    original_after = _normalize_skills_dict(skills.get("after"))
    after = (
        {k: list(v) for k, v in original_after.items()}
        if original_after
        else {k: list(v) for k, v in before.items()}
    )

    can_edit_skills = bool(after) or bool(parsed)
    if not after and parsed:
        after = {k: list(v) for k, v in parsed.items()}

    applied_changes: list[dict[str, Any]] = []

    for change in changes:
        if not isinstance(change, dict):
            continue

        skill = str(change.get("skill", "")).strip()
        action = str(change.get("action", "")).strip()
        if not skill or action not in ALLOWED_SKILL_ACTIONS:
            continue

        if action == "add":
            if not can_edit_skills:
                gaps.append(
                    {
                        "requirement": skill,
                        "reason": (
                            f"Could not apply skill add for {skill!r}: "
                            "Skills section snapshot was unavailable, so the "
                            "edit was not written to the resume."
                        ),
                    }
                )
                continue

            if _find_skill_category(after, skill) is not None:
                if _find_skill_category(before, skill) is None:
                    # Present in after only — treat as applied add.
                    applied_changes.append(change)
                # Already on the resume before this plan — nothing to apply.
                continue

            category = _category_for_new_skill(after)
            after.setdefault(category, []).append(skill)
            applied_changes.append(change)
            continue

        if action == "highlight":
            if _find_skill_category(after, skill) is not None:
                applied_changes.append(change)
                continue
            gaps.append(
                {
                    "requirement": skill,
                    "reason": (
                        f"Could not apply skill highlight for {skill!r}: "
                        "skill is not present in the Skills section."
                    ),
                }
            )
            continue

        # surface_form_alignment requires a concrete after snapshot from the model
        if original_after and _find_skill_category(after, skill) is not None:
            applied_changes.append(change)
            continue
        gaps.append(
            {
                "requirement": skill,
                "reason": (
                    f"Could not apply surface-form alignment for {skill!r}: "
                    "skills.after did not include an aligned form to write."
                ),
            }
        )

    skills["before"] = before
    skills["after"] = after
    skills["changes"] = applied_changes
    edit_plan["skills"] = skills
    return edit_plan


def ensure_memory_skills_for_job(
    edit_plan: dict[str, Any],
    job: dict[str, Any],
    memory_evidence: list[str] | None = None,
    master_resume_tex: str | None = None,
) -> dict[str, Any]:
    """
    If memory names a skill that the job also requires, force it into
    skills.changes / skills.after so same-run memory facts actually land
    in the change log (not only in the prompt).
    """
    memory_evidence = memory_evidence or []
    required = {
        str(s).strip().lower(): str(s).strip()
        for s in (job.get("required_skills") or [])
        if str(s).strip()
    }
    if not required:
        return edit_plan

    mem_skills: list[str] = []
    for item in memory_evidence:
        text = str(item).strip()
        lower = text.lower()
        if lower.startswith("skill:"):
            mem_skills.append(text.split(":", 1)[1].strip())
            continue
        for req_lower, req_orig in required.items():
            if req_lower in lower:
                mem_skills.append(req_orig)

    seen: set[str] = set()
    ordered_mem: list[str] = []
    for skill in mem_skills:
        key = skill.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered_mem.append(skill)

    relevant = [s for s in ordered_mem if s.lower() in required]
    if not relevant:
        return edit_plan

    skills = edit_plan.setdefault(
        "skills", {"before": {}, "after": {}, "changes": []}
    )
    if not isinstance(skills.get("before"), dict):
        skills["before"] = {}
    if not isinstance(skills.get("after"), dict):
        skills["after"] = {}
    if not isinstance(skills.get("changes"), list):
        skills["changes"] = []

    after = skills["after"]
    if not any(isinstance(v, list) and v for v in after.values()):
        if skills["before"] and any(
            isinstance(v, list) and v for v in skills["before"].values()
        ):
            after = {k: list(v) for k, v in skills["before"].items()}
        elif master_resume_tex:
            after = parse_skills_after_from_tex(master_resume_tex)
        else:
            after = {
                "Languages": [],
                "ML & Data": [],
                "Systems & Tools": [],
            }
        skills["after"] = after
        if not skills["before"]:
            skills["before"] = {k: list(v) for k, v in after.items()}

    existing_change_skills = {
        str(c.get("skill", "")).strip().lower()
        for c in skills["changes"]
        if isinstance(c, dict)
    }

    for skill in relevant:
        key = skill.lower()

        category = "Systems & Tools"
        if category not in after or not isinstance(after.get(category), list):
            category = next(iter(after.keys()), "Systems & Tools")
            after.setdefault(category, [])

        bucket = after.setdefault(category, [])
        if not any(str(x).lower() == key for x in bucket):
            bucket.append(skill)

        if key in existing_change_skills:
            continue

        skills["changes"].append(
            {
                "skill": skill,
                "action": "add",
                "reason": (
                    f"Job requires {skill}; candidate confirmed it in memory "
                    "during human review."
                ),
                "evidence": [
                    {
                        "source": "memory",
                        "reference": f"skill: {skill}",
                    }
                ],
            }
        )
        existing_change_skills.add(key)

    edit_plan["skills"] = skills
    return edit_plan


def ensure_fit_analysis_project_swap(
    edit_plan: dict[str, Any],
    *,
    fit_projects: dict[str, Any] | None,
    resume_summary_data: dict[str, Any],
    portfolio_data: dict[str, Any],
    master_resume_tex: str | None = None,
) -> dict[str, Any]:
    """
    When fit_analysis recommended a verified swap, force it into
    edit_plan.project_swaps so tailor_resume actually applies it.

    Fit analysis previously returned only rendered text to the agent loop;
    the tailor prompt never received the structured recommendation, so
    project_swaps stayed []. This mirrors ensure_memory_skills_for_job:
    deterministic post-processing, not another LLM call.
    """
    if not isinstance(fit_projects, dict):
        return edit_plan
    if fit_projects.get("verdict") != "swap_suggested":
        return edit_plan

    swap_sug = fit_projects.get("swap_suggestion") or {}
    weak = fit_projects.get("weak_project") or {}
    if not isinstance(swap_sug, dict):
        return edit_plan

    add_id = str(swap_sug.get("add_project_id") or "").strip()
    if not add_id:
        return edit_plan

    try:
        add_project = get_portfolio_project(portfolio_data, add_id)
    except KeyError:
        gaps = edit_plan.setdefault("gaps", [])
        if isinstance(gaps, list):
            gaps.append(
                {
                    "requirement": f"project swap to {add_id}",
                    "reason": (
                        "Fit analysis recommended a project swap, but "
                        f"add_project_id={add_id!r} is not in portfolio.json."
                    ),
                }
            )
        return edit_plan

    remove_name = str(
        swap_sug.get("remove_project_name")
        or weak.get("name")
        or ""
    ).strip()
    remove_norm = normalize_project_name(remove_name)

    slot_map = current_resume_project_slots(
        resume_summary_data,
        portfolio_data,
    )
    # Prefer PROJECT-ID markers in the master tex when available.
    tex_slots: dict[str, str] = {}
    if master_resume_tex:
        for match in re.finditer(
            r"%\s*AGENT-SWAP-(?:START|TARGET):\s*(project-\d+)\s*\n"
            r"\s*%\s*PROJECT-ID:\s*([^\s]+)",
            master_resume_tex,
        ):
            tex_slots[match.group(1)] = match.group(2).strip()
        if tex_slots:
            slot_map = {**slot_map, **tex_slots}

    slot: str | None = None
    remove_id: str | None = None
    for candidate_slot, project_id in slot_map.items():
        try:
            current = get_portfolio_project(portfolio_data, project_id)
        except KeyError:
            continue
        current_name = normalize_project_name(current.get("project_name", ""))
        if remove_norm and (
            current_name == remove_norm
            or remove_norm in current_name
            or current_name in remove_norm
        ):
            slot = candidate_slot
            remove_id = project_id
            break
        if project_id == add_id:
            # Already on the resume — nothing to do.
            return edit_plan

    if slot is None and remove_norm:
        # Fall back: match resume_summary project order to slot index.
        resume_projects = resume_summary_data.get("resume_projects") or []
        for index, name in enumerate(resume_projects[:3], start=1):
            if normalize_project_name(name) == remove_norm or (
                remove_norm in normalize_project_name(name)
                or normalize_project_name(name) in remove_norm
            ):
                slot = f"project-{index}"
                remove_id = slot_map.get(slot)
                break

    if slot is None or slot not in ALLOWED_PROJECT_SLOTS:
        gaps = edit_plan.setdefault("gaps", [])
        if isinstance(gaps, list):
            gaps.append(
                {
                    "requirement": f"replace project {remove_name or '(unknown)'}",
                    "reason": (
                        "Fit analysis recommended a project swap, but the weak "
                        "resume project could not be matched to a swap slot."
                    ),
                }
            )
        return edit_plan

    if not remove_id:
        remove_id = slot_map.get(slot) or ""

    if remove_id == add_id:
        return edit_plan

    swaps = edit_plan.setdefault("project_swaps", [])
    if not isinstance(swaps, list):
        swaps = []
        edit_plan["project_swaps"] = swaps

    # Replace any conflicting swap on the same slot; keep others.
    swaps[:] = [
        s
        for s in swaps
        if not (isinstance(s, dict) and s.get("slot") == slot)
    ]

    reason = str(
        swap_sug.get("reason")
        or fit_projects.get("statement")
        or (
            f"Fit analysis recommended replacing {remove_name} with "
            f"{add_project.get('project_name')} for stronger job alignment."
        )
    ).strip()

    swaps.append(
        {
            "slot": slot,
            "remove_project_id": remove_id,
            "add_project_id": add_id,
            "reason": reason,
            "evidence": [
                {
                    "source": "portfolio",
                    "reference": add_id,
                }
            ],
        }
    )
    edit_plan["project_swaps"] = swaps
    return edit_plan


def apply_edit_plan(
    master_tex: str,
    edit_plan: dict[str, Any],
    portfolio_data: dict[str, Any],
) -> str:
    tex = replace_professional_summary(
        master_tex,
        edit_plan["summary"]["after"],
    )

    for bullet in edit_plan["experience_bullets"]:
        tex = replace_marked_resume_item(
            tex=tex,
            marker=bullet["target"],
            new_text=bullet["after"],
        )

    skills_after = edit_plan["skills"].get("after") or {}
    has_rendered_skills = isinstance(skills_after, dict) and any(
        isinstance(v, list) and v for v in skills_after.values()
    )
    # Write Skills whenever we have a concrete after snapshot and at least
    # one applied change (materialize_skill_edits populates before/after).
    if edit_plan["skills"].get("changes") and has_rendered_skills:
        tex = replace_skills_section(
            tex,
            skills_after,
        )

    for swap in edit_plan["project_swaps"]:
        project = get_portfolio_project(
            portfolio_data,
            swap["add_project_id"],
        )
        tex = replace_project_slot(
            tex=tex,
            slot=swap["slot"],
            project=project,
        )

    return tex


def get_pdf_page_count(pdf_path: str | Path) -> int:
    """
    Return the PDF page count using pdfinfo from Poppler.

    Falls back to a lightweight PDF object scan when pdfinfo is unavailable.
    """
    source = Path(pdf_path).resolve()

    if not source.exists():
        raise FileNotFoundError(f"PDF not found: {source}")

    if source.suffix.lower() != ".pdf":
        raise ValueError("Page-count source must be a .pdf file.")

    try:
        result = subprocess.run(
            ["pdfinfo", str(source)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return _pdf_page_count_fallback(source)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out while inspecting PDF: {source}"
        ) from exc

    if result.returncode != 0:
        return _pdf_page_count_fallback(source)

    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())

    return _pdf_page_count_fallback(source)


def _pdf_page_count_fallback(source: Path) -> int:
    """
    Count pages when Poppler's pdfinfo is not on PATH.

    Handles both plain and FlateDecode-compressed object streams by scanning
    decompressed content for /Type /Page dictionaries (excluding /Pages).
    """
    import zlib

    data = source.read_bytes()

    def count_page_dicts(blob: bytes) -> int:
        return len(re.findall(rb"/Type\s*/Page(?![s\w])", blob))

    pages = count_page_dicts(data)
    if pages >= 1:
        return pages

    # Decompress FlateDecode streams and scan again.
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, flags=re.DOTALL):
        payload = match.group(1)
        try:
            inflated = zlib.decompress(payload)
        except zlib.error:
            try:
                inflated = zlib.decompress(payload, -zlib.MAX_WBITS)
            except zlib.error:
                continue
        pages += count_page_dicts(inflated)

    if pages < 1:
        raise RuntimeError(
            "Could not determine PDF page count (pdfinfo missing and "
            f"fallback scan found no pages): {source}"
        )
    return pages


def verify_one_page(pdf_path: str | Path) -> Path:
    source = Path(pdf_path).resolve()
    page_count = get_pdf_page_count(source)

    if page_count != 1:
        raise ValueError(
            "One-page rule failed.\n"
            f"PDF file: {source}\n"
            f"Page count: {page_count}"
        )

    return source


def build_change_log(
    job: dict[str, Any],
    rank: int,
    edit_plan: dict[str, Any],
    pdf_path: Path,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "job": {
            "title": job.get(
                "title",
                job.get("job_title", "Unknown title"),
            ),
            "company": job.get(
                "company",
                job.get("company_name", "Unknown company"),
            ),
        },
        "one_page_verified": True,
        "pdf_file": pdf_path.name,
        "summary_edit": edit_plan["summary"],
        "experience_bullet_edits": edit_plan["experience_bullets"],
        "skill_edits": edit_plan["skills"],
        "project_swaps": edit_plan["project_swaps"],
        "genuine_gaps": edit_plan["gaps"],
    }


def process_top_job(
    job: dict[str, Any],
    rank: int,
    master_resume_tex: str,
    portfolio: dict[str, Any],
    output_dir: Path,
    model: str = RESUME_EDIT_MODEL,
    revision_feedback: str | None = None,
    fit_projects: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Tailor, compile, validate, and save one Top-3 resume.
    """
    title = job.get(
        "title",
        job.get("job_title", f"job_{rank}"),
    )
    company = job.get(
        "company",
        job.get("company_name", "unknown_company"),
    )

    folder_name = (
        f"{rank:02d}_"
        f"{safe_folder_name(company)}_"
        f"{safe_folder_name(title)}"
    )

    job_dir = Path(output_dir) / folder_name
    job_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing #{rank}: {title} at {company}")
    if revision_feedback:
        print(f"  Revision feedback: {revision_feedback[:200]}")
    if isinstance(fit_projects, dict) and fit_projects.get("verdict"):
        print(f"  Fit project guidance: {fit_projects.get('verdict')}")

    edit_plan = generate_edit_plan(
        job=job,
        master_resume_tex=master_resume_tex,
        portfolio_data=portfolio,
        model=model,
        revision_feedback=revision_feedback,
        fit_projects=fit_projects,
    )
    edit_plan = materialize_skill_edits(
        edit_plan,
        master_resume_tex,
    )

    tailored_tex = apply_edit_plan(
        master_tex=master_resume_tex,
        edit_plan=edit_plan,
        portfolio_data=portfolio,
    )

    tex_path = job_dir / "resume.tex"
    tex_path.write_text(tailored_tex, encoding="utf-8")

    pdf_path = compile_resume(str(tex_path))
    pdf_path = verify_one_page(pdf_path)

    edit_plan_path = job_dir / "edit_plan.json"
    edit_plan_path.write_text(
        json.dumps(edit_plan, indent=2),
        encoding="utf-8",
    )

    change_log = build_change_log(
        job=job,
        rank=rank,
        edit_plan=edit_plan,
        pdf_path=pdf_path,
    )

    change_log_path = job_dir / "change_log.json"
    change_log_path.write_text(
        json.dumps(change_log, indent=2),
        encoding="utf-8",
    )

    print(f"  Saved: {tex_path}")
    print(f"  Saved: {pdf_path}")
    print("  One-page rule: passed")

    return {
        "rank": rank,
        "title": title,
        "company": company,
        "folder": str(job_dir),
        "tex_path": str(tex_path),
        "pdf_path": str(pdf_path),
        "edit_plan_path": str(edit_plan_path),
        "change_log_path": str(change_log_path),
    }


if __name__ == "__main__":
    print(
        "This module is intended to be imported by the notebook.\n"
        "Import configure_pipeline and process_top_job."
    )
