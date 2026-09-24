"""
scripts/make_submission_zip.py

Build the journal upload archive from paper/latex/ and prove that the archive,
compiled on its own, gives the PDF we checked.

Why this exists. The journal compiles the reviewers' PDF from the uploaded
source archive, not from our PDF. A hand-assembled archive once shipped an
older bibliography and older figure files next to a current main.tex, and the
reviewers read eleven unresolved citations and four stale figures that our own
build did not have. A script that assembles the archive from what main.tex
actually includes, then compiles the unpacked copy in a clean directory and
compares it with paper/latex/main.pdf, closes that gap.

Checks, each fatal:
  - every file in the archive is byte-identical to its paper/latex/ source;
  - every \\includegraphics target in main.tex is present;
  - the clean compile of main.tex has no undefined citation or reference and
    prints no "[?]";
  - its extracted text equals the text of paper/latex/main.pdf.

Usage:
    python scripts/make_submission_zip.py --version 3.0
Writes paper/Dia-Med-VLM-v.<version>.zip.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATEX = ROOT / "paper" / "latex"
STATIC = ["main.tex", "supplementary.tex", "bibliography.bib", "sn-jnl.cls", "sn-mathphys-num.bst"]
MIKTEX = Path(r"C:\Program Files\MiKTeX\miktex\bin\x64")


def _tool(name: str) -> str:
    # Two pdftotext binaries exist on this machine; poppler's sits beside MiKTeX.
    cand = MIKTEX / f"{name}.exe"
    found = cand if cand.exists() else shutil.which(name)
    if not found:
        raise SystemExit(f"Required tool not found: {name}")
    return str(found)


def members() -> list[Path]:
    tex = (LATEX / "main.tex").read_text(encoding="utf-8")
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    files = [LATEX / f for f in STATIC] + [LATEX / f for f in figs]
    files += sorted((LATEX / "bst").glob("*.bst"))
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit("Missing files: " + ", ".join(missing))
    return files


def compile_tex(workdir: Path, stem: str) -> str:
    pdflatex, bibtex = _tool("pdflatex"), _tool("bibtex")
    def run(cmd: list[str]) -> None:
        # pdflatex exits non-zero on recoverable warnings; the log is what gets checked.
        subprocess.run(cmd, cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    run([pdflatex, "-interaction=nonstopmode", f"{stem}.tex"])
    if stem == "main":
        run([bibtex, stem])
    for _ in range(2):
        run([pdflatex, "-interaction=nonstopmode", f"{stem}.tex"])
    return (workdir / f"{stem}.log").read_text(encoding="utf-8", errors="replace")


def pdf_words(pdf: Path) -> list[str]:
    out = subprocess.run([_tool("pdftotext"), str(pdf), "-"], capture_output=True, check=True)
    return out.stdout.decode("utf-8", errors="replace").split()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", required=True, help="Archive version, e.g. 3.0")
    ap.add_argument("--seed", type=int, default=42, help="Unused; logged for consistency.")
    args = ap.parse_args()

    files = members()
    out = ROOT / "paper" / f"Dia-Med-VLM-v.{args.version}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(LATEX).as_posix())

    fails: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(out) as z:
            z.extractall(work)
        for f in files:
            if (work / f.relative_to(LATEX)).read_bytes() != f.read_bytes():
                fails.append(f"archive copy differs: {f.name}")
        for stem in ("main", "supplementary"):
            log = compile_tex(work, stem)
            if re.search(r"(Citation|Reference) `[^']+' .*undefined", log):
                fails.append(f"{stem}: undefined citation or reference in clean compile")
            words = pdf_words(work / f"{stem}.pdf")
            if any(w.startswith("[?") for w in words):
                fails.append(f"{stem}: '[?]' in clean compile")
            ref = LATEX / f"{stem}.pdf"
            if words != pdf_words(ref):
                fails.append(f"{stem}: clean-compile text differs from {ref.name}")

    print(f"{len(files)} files archived")
    if fails:
        print("FAILED:\n  " + "\n  ".join(fails))
        return 1
    print("clean compile matches paper/latex/main.pdf and supplementary.pdf")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
