"""Repo-root path helpers for scripts living under src/."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

# Allow `python src/run_agent_e2e.py` without setting PYTHONPATH.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_DIR = REPO_ROOT / "data"
RESUME_DIR = REPO_ROOT / "resume"

JOBS_CSV = DATA_DIR / "ai_ml_jobs.csv"
CANDIDATE_PERSONA = DATA_DIR / "candidate_persona.json"
PORTFOLIO_JSON = DATA_DIR / "portfolio.json"
RESUME_TEX = RESUME_DIR / "resume.tex"
RESUME_PDF = RESUME_DIR / "resume.pdf"
MEMORY_JSON = REPO_ROOT / "memory.json"
ENV_FILE = REPO_ROOT / ".env"
NOTEBOOK_PATH = REPO_ROOT / "agent_notebook_v2.ipynb"
