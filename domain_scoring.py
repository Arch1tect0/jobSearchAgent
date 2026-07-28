"""
Deterministic industry/domain alignment for job scoring.

No LLM calls. Used by score_jobs to compute domain_alignment in [0, 1].

Why exact equality fails
------------------------
ai_ml_jobs.csv ``Industry/Domain`` values are long employer-vertical labels
(e.g. ``Healthcare / Hospital Systems (Enterprise Data & Analytics)``).
portfolio.json stores a separate ``industry`` (sector) and ``domain``
(ML technique), e.g. industry=``Healthcare``, domain=``Predictive Analytics``.
Case-insensitive equality against portfolio ``domain`` alone yields 0 overlaps
on the current 21-job board.

Matching rules (deterministic)
------------------------------
1. Normalize text: lowercase; replace ``/|-,()`` and similar with spaces;
   collapse whitespace.
2. Map text → concept tags via a fixed synonym group table (aliases matched
   as whole phrases, longest first). Job text is scored against each
   portfolio project's ``industry`` + ``domain`` (both fields).
3. Score for one job = max over portfolio projects of:
   - **1.0** if job and project share ≥1 synonym-group concept, OR if a
     normalized portfolio industry/domain string of length ≥5 is a substring
     of the job label (or vice versa).
   - **0.5** if non-stopword token Jaccard ≥ ``TOKEN_JACCARD_THRESHOLD``
     (default 0.25) and no stronger hit.
   - **0.0** otherwise.
4. Missing/blank job ``industry_domain`` → **0.5** (neutral), unchanged.

Synonym groups are intentionally conservative: they encode industry/technique
relatedness visible in this board (healthcare↔hospital/medical, energy↔oil &
gas/geothermal, HR↔recruiting, GenAI↔generative AI, etc.). Unrelated labels
(e.g. Legal with no legal portfolio project) stay at 0.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

TOKEN_JACCARD_THRESHOLD = 0.25
MIN_SUBSTRING_LEN = 5

# Each frozenset is one concept. Any alias hit tags that concept.
DOMAIN_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "healthcare",
            "health",
            "hospital",
            "medical",
            "medicine",
            "clinical",
            "surgery",
            "academic medicine",
            "academic medical",
        }
    ),
    frozenset(
        {
            "energy",
            "oil",
            "gas",
            "oilfield",
            "petroleum",
            "geothermal",
            "utilities",
            "clean energy",
            "energy and utilities",
        }
    ),
    frozenset(
        {
            "human resources",
            "hr",
            "hr technology",
            "recruiting",
            "talent acquisition",
            "human resources technology",
            "human resources solutions",
        }
    ),
    frozenset(
        {
            "legal",
            "law",
            "legal technology",
            "legal tech",
            "personal injury",
        }
    ),
    frozenset(
        {
            "education",
            "higher education",
            "education technology",
            "edtech",
            "community college",
            "tutoring",
            "instruction",
            "adaptive learning",
        }
    ),
    frozenset(
        {
            "cybersecurity",
            "cyber",
            "endpoint security",
            "ai security",
        }
    ),
    frozenset(
        {
            "saas",
            "software as a service",
            "daas",
            "enterprise software",
            "cloud platform",
            "customer engagement",
        }
    ),
    frozenset(
        {
            "consulting",
            "management consulting",
            "advisory",
            "public services",
            "government",
        }
    ),
    frozenset(
        {
            "generative ai",
            "genai",
            "gen ai",
            "generative",
        }
    ),
    frozenset(
        {
            "geospatial",
            "geoscience",
            "geographic",
        }
    ),
    frozenset(
        {
            "nlp",
            "natural language processing",
        }
    ),
    frozenset(
        {
            "computer vision",
            "vision",
        }
    ),
    frozenset(
        {
            "anomaly detection",
            "fraud detection",
            "anomaly",
        }
    ),
    frozenset(
        {
            "forecasting",
            "time-series",
            "time series",
            "demand forecasting",
        }
    ),
    frozenset(
        {
            "predictive analytics",
            "predictive",
        }
    ),
    frozenset(
        {
            "recommendation",
            "personalization",
            "recommendation systems",
        }
    ),
    frozenset(
        {
            "financial",
            "finance",
            "fintech",
            "financial services",
        }
    ),
    frozenset(
        {
            "retail",
            "e-commerce",
            "ecommerce",
        }
    ),
    frozenset(
        {
            "manufacturing",
        }
    ),
    frozenset(
        {
            "telecommunications",
            "telecom",
        }
    ),
    frozenset(
        {
            "transportation",
            "logistics",
            "delivery",
        }
    ),
    frozenset(
        {
            "knowledge management",
        }
    ),
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "of",
        "the",
        "to",
        "in",
        "on",
        "with",
        "or",
        "at",
        "by",
        "team",
        "center",
        "centre",
        "excellence",
        "enterprise",
        "applied",
        "data",
        "ai",
        "ml",
        "machine",
        "learning",
        "solutions",
        "platform",
        "software",
        "technology",
        "services",
        "systems",
        "system",
        "engineer",
        "engineering",
        "leadership",
        "impact",
        "governance",
        "research",
        "vertical",
        "co",
        "amp",
    }
)

_ALIAS_TO_GROUP: dict[str, str] = {}
for _idx, _group in enumerate(DOMAIN_SYNONYM_GROUPS):
    _gid = f"g{_idx}"
    for _alias in _group:
        _ALIAS_TO_GROUP[_alias] = _gid

# Longest aliases first so "human resources technology" wins over "human resources".
_ALIASES_LONGEST_FIRST: tuple[str, ...] = tuple(
    sorted(_ALIAS_TO_GROUP.keys(), key=len, reverse=True)
)


def normalize_domain_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[/|\\,;:()\[\]{}+\-–—]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _phrase_in_text(phrase: str, text: str) -> bool:
    """True if phrase appears as a whole-word/phrase span in normalized text."""
    if not phrase or not text:
        return False
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def extract_domain_concepts(text: str) -> set[str]:
    """Map free text to synonym-group ids (deterministic)."""
    norm = normalize_domain_text(text)
    if not norm:
        return set()
    concepts: set[str] = set()
    for alias in _ALIASES_LONGEST_FIRST:
        if _phrase_in_text(alias, norm):
            concepts.add(_ALIAS_TO_GROUP[alias])
    return concepts


def _content_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in normalize_domain_text(text).split()
        if len(tok) >= 4 and tok not in _STOPWORDS
    }


def token_jaccard(a: str, b: str) -> float:
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _substring_related(job_norm: str, phrase: str) -> bool:
    pn = normalize_domain_text(phrase)
    if len(pn) < MIN_SUBSTRING_LEN or len(job_norm) < MIN_SUBSTRING_LEN:
        return False
    return pn in job_norm or job_norm in pn


def score_project_domain_alignment(
    job_industry_domain: str,
    project: dict[str, Any],
    *,
    token_threshold: float = TOKEN_JACCARD_THRESHOLD,
) -> float:
    """Relatedness of one job label to one portfolio project in {0.0, 0.5, 1.0}."""
    job_norm = normalize_domain_text(job_industry_domain)
    if not job_norm:
        return 0.0

    industry = str(project.get("industry") or "")
    domain = str(project.get("domain") or "")
    project_blob = f"{industry} {domain}".strip()

    job_concepts = extract_domain_concepts(job_industry_domain)
    project_concepts = extract_domain_concepts(project_blob)
    if job_concepts & project_concepts:
        return 1.0

    if _substring_related(job_norm, industry) or _substring_related(job_norm, domain):
        return 1.0

    if token_jaccard(job_industry_domain, project_blob) >= token_threshold:
        return 0.5

    return 0.0


def score_domain_alignment(
    job: dict[str, Any],
    portfolio_list: Iterable[dict[str, Any]],
    *,
    token_threshold: float = TOKEN_JACCARD_THRESHOLD,
) -> float:
    """
    Domain alignment for scoring. Returns 0.0–1.0.

    Compares job ``industry_domain`` to each project's ``industry`` and ``domain``.
    """
    job_domain = job.get("industry_domain") or ""
    if not str(job_domain).strip():
        return 0.5

    best = 0.0
    for project in portfolio_list:
        if not isinstance(project, dict):
            continue
        best = max(
            best,
            score_project_domain_alignment(
                str(job_domain),
                project,
                token_threshold=token_threshold,
            ),
        )
        if best >= 1.0:
            return 1.0
    return best
