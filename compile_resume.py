from pathlib import Path
import subprocess


def compile_resume(tex_path: str) -> Path:
    source = Path(tex_path).resolve()

    if not source.exists():
        raise FileNotFoundError(f"LaTeX source not found: {source}")

    if source.suffix.lower() != ".tex":
        raise ValueError("Resume source must be a .tex file.")

    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        source.name,
    ]

    result = subprocess.run(
        command,
        cwd=source.parent,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        log_path = source.with_suffix(".log")
        raise RuntimeError(
            "Resume compilation failed.\n"
            f"Log file: {log_path}\n\n"
            f"{result.stdout[-4000:]}"
        )

    pdf_path = source.with_suffix(".pdf")

    if not pdf_path.exists():
        raise RuntimeError("pdflatex completed but no PDF was produced.")

    return pdf_path


if __name__ == "__main__":
    compiled_pdf = compile_resume("resume.tex")
    print(f"Compiled resume: {compiled_pdf}")