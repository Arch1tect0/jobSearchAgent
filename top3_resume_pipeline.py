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


def sanitize_evidence_records(
    edit_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize supported evidence sources and remove unsupported ones.
    """

    def clean(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []

        cleaned: list[dict[str, Any]] = []

        aliases = {
            "master skills": "master_skills",
            "master-skills": "master_skills",
            "master_skill": "master_skills",
            "resume.tex": "resume",
            "portfolio.json": "portfolio",
            "candidate memory": "memory",
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

            cleaned.append(
                {
                    "source": source,
                    "reference": reference,
                }
            )

        return cleaned

    summary = edit_plan.get("summary")

    if isinstance(summary, dict):
        summary["evidence"] = clean(
            summary.get("evidence")
        )

    for bullet in edit_plan.get("experience_bullets", []):
        if isinstance(bullet, dict):
            bullet["evidence"] = clean(
                bullet.get("evidence")
            )

    skills = edit_plan.get("skills", {})

    if isinstance(skills, dict):
        for change in skills.get("changes", []):
            if isinstance(change, dict):
                change["evidence"] = clean(
                    change.get("evidence")
                )

    for swap in edit_plan.get("project_swaps", []):
        if isinstance(swap, dict):
            swap["evidence"] = clean(
                swap.get("evidence")
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

CURRENT LATEX:
{current_resume_tex}

Only these modifications are allowed:

1. Rewrite the Professional Summary.

2. Modify exactly two experience bullets:
   - experience-bullet-1
   - experience-bullet-2

3. Add, highlight, or surface-align skills only when supported by the
   resume, portfolio, master skills, or approved memory evidence.

4. Swap a project only when a portfolio project is genuinely more
   relevant than a current resume project.

Rules:

- Do not change names, contact information, employers, titles, dates,
  education, certifications, formatting, or LaTeX commands.
- Do not rewrite the entire resume.
- Do not invent skills, projects, employers, titles, dates, metrics,
  or results.
- Every edit must include evidence.
- Every added project must use an exact project_id from the portfolio.
- Unsupported requirements must be recorded under gaps.
- Empty skills.changes and project_swaps lists are valid.
- Return exactly two experience bullet edits.

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
    )

    raw_response = call_resume_edit_model(
        prompt,
        model=model,
    )

    edit_plan = parse_json_response(raw_response)

    edit_plan = sanitize_evidence_records(
        edit_plan
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

    updated, count = pattern.subn(
        render_project_latex(project, slot),
        tex,
        count=1,
    )

    if count != 1:
        raise ValueError(
            f"Could not locate project markers for {slot!r}."
        )

    return updated


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

    if edit_plan["skills"].get("changes"):
        tex = replace_skills_section(
            tex,
            edit_plan["skills"]["after"],
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
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The 'pdfinfo' command was not found. Install Poppler and "
            "ensure pdfinfo is available on PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out while inspecting PDF: {source}"
        ) from exc

    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            "PDF inspection failed.\n"
            f"PDF file: {source}\n\n"
            f"{details[-4000:]}"
        )

    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())

    raise RuntimeError(
        "pdfinfo completed, but no page count was found."
    )


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

    edit_plan = generate_edit_plan(
        job=job,
        master_resume_tex=master_resume_tex,
        portfolio_data=portfolio,
        model=model,
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
