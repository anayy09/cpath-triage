"""
scripts/make_deposit.py

Assemble the Zenodo deposit for the manuscript (revision items C-E2 and C-I4).
Writes a staging directory and a manifest; uploads nothing and mints nothing.

Why a script rather than a zip made by hand. The Editor's requirement is that the
archive contain what produced the numbers, and `results/` is gitignored, so the
public repository and the archive have different contents by construction. Making
the inclusion policy executable is the only way the two stay honest about which
files are in the archive and what they hash to.

What goes in, and why.

  Code                Everything a reader needs to rerun the analysis: src,
                      scripts, tests, requirements, lint config, .env.example.
  Prediction
  artifacts           Every .json, .parquet, .png, .md and .txt under results/.
                      About 13 MB. These are what make the deposit reproducible:
                      with them, all twelve tables and eight of the nine figures
                      regenerate with no API calls and no GPU. Without them a
                      reader would need roughly 34,000 billed calls and two GPU
                      trainings. Verified by running the analysis chain inside
                      the staging tree.
  CNN checkpoints     Excluded by default. The three best_model.pt files are
                      43 MB each and reproduce nothing that the saved predictions
                      do not already give. Pass --include-checkpoints to add them
                      if re-running eval_cnn without retraining matters more than
                      archive size.
  Manuscript source   Excluded by default, because paper/ is gitignored and the
                      decision to track it has not been made. Pass --include-paper
                      once it is.
  Data                Never. PathMNIST is redistributed by MedMNIST and MedMNIST+
                      under their own terms and is 13.7 GB.

Incomplete runs are skipped. A results/cnn/* directory holding a checkpoint but
no metrics.json is a training run still in flight, and shipping a partial run
would put an artifact in the archive that no table traces to.

Outputs:
    build/zenodo_deposit/            staging tree
    build/zenodo_deposit/MANIFEST.txt   sha256 and size for every file

Usage:
    python scripts/make_deposit.py
    python scripts/make_deposit.py --include-checkpoints --include-paper
    python scripts/make_deposit.py --out build/zenodo_deposit --zip
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CODE_DIRS = ("src", "scripts", "tests")
ROOT_FILES = (
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "requirements.txt",
    "ruff.toml",
    ".env.example",
)
RESULTS_SUFFIXES = {".json", ".parquet", ".png", ".md", ".txt"}
CHECKPOINT_SUFFIX = ".pt"
SKIP_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints"}

# The deposit's stated scope is the code and artifacts behind the manuscript's
# tables and figures, and the README maps each script to what it produces. These
# four produce nothing in the paper, so carrying them invites a reader to look
# for an output that is not there. Two build and publish the deposit itself, and
# two belong to the report-synthesis stage that Figure 1 greys out and the
# caption says is not evaluated. All four stay in the GitHub repository.
EXCLUDED_SCRIPTS = {
    "make_deposit.py",
    "zenodo_new_version.py",
    "generate_report.py",
    "error_analysis.py",
}

# Outputs of the excluded stage-7 scripts. The montage alone is 2.3 MB and no
# table, figure or claim in the manuscript refers to it.
EXCLUDED_RESULT_DIRS = {"stage7"}


def _iter_code_files() -> list[Path]:
    out: list[Path] = []
    for d in CODE_DIRS:
        root = PROJECT_ROOT / d
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if (
                p.is_file()
                and p.suffix not in (".pyc", ".pyo")
                and not any(part in SKIP_DIR_NAMES for part in p.parts)
                and not (d == "scripts" and p.name in EXCLUDED_SCRIPTS)
            ):
                out.append(p)
    for name in ROOT_FILES:
        p = PROJECT_ROOT / name
        if p.exists():
            out.append(p)
    return out


def _incomplete_cnn_runs() -> set[Path]:
    """Training directories with a checkpoint but no metrics.json are still running."""
    cnn = PROJECT_ROOT / "results" / "cnn"
    if not cnn.exists():
        return set()
    return {
        d for d in cnn.iterdir()
        if d.is_dir() and not (d / "metrics.json").exists()
    }


def _iter_result_files(include_checkpoints: bool) -> tuple[list[Path], list[Path]]:
    root = PROJECT_ROOT / "results"
    if not root.exists():
        return [], []
    skipped_dirs = _incomplete_cnn_runs()
    keep: list[Path] = []
    skipped: list[Path] = []
    wanted = set(RESULTS_SUFFIXES)
    if include_checkpoints:
        wanted.add(CHECKPOINT_SUFFIX)
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.relative_to(root).parts[0] in EXCLUDED_RESULT_DIRS:
            skipped.append(p)
            continue
        if any(parent in skipped_dirs for parent in p.parents):
            skipped.append(p)
            continue
        (keep if p.suffix in wanted else skipped).append(p)
    return keep, skipped


def _iter_paper_files() -> list[Path]:
    root = PROJECT_ROOT / "paper" / "latex"
    if not root.exists():
        return []
    return [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".tex", ".bib", ".cls", ".bst", ".png"}
    ]


def _empty_directory(path: Path) -> None:
    """
    Clear a staging directory without requiring that it be removable.

    A sync client can hold an open handle on the directory itself, which makes
    rmtree fail on the final rmdir even after every child is gone. Emptying the
    contents is what the rebuild actually needs.
    """
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="build/zenodo_deposit")
    parser.add_argument("--include-checkpoints", action="store_true",
                        help="Add the ResNet-18 best_model.pt files (43 MB each).")
    parser.add_argument("--include-paper", action="store_true",
                        help="Add paper/latex sources. Only once paper/ is tracked.")
    parser.add_argument("--zip", action="store_true", help="Also write a .zip beside the tree.")
    args = parser.parse_args()

    out_dir = (PROJECT_ROOT / args.out).resolve()
    _empty_directory(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_code_files()
    results, skipped = _iter_result_files(args.include_checkpoints)
    files += results
    if args.include_paper:
        files += _iter_paper_files()

    manifest: list[str] = []
    total = 0
    unreadable: list[Path] = []
    for src in files:
        rel = src.relative_to(PROJECT_ROOT)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            digest = _sha256(src)
        except OSError:
            # This repository lives inside OneDrive, where a file can be a
            # cloud placeholder that fails to hydrate on read. Collect them all
            # and report once, rather than failing on the first.
            unreadable.append(rel)
            continue
        size = src.stat().st_size
        total += size
        manifest.append(f"{digest}  {size:>12}  {rel.as_posix()}")

    if unreadable:
        print("Could not read the following files. If this repository is inside "
              "OneDrive, set the folder to 'Always keep on this device' and retry:",
              file=sys.stderr)
        for rel in unreadable:
            print(f"  {rel.as_posix()}", file=sys.stderr)
        print("The deposit is incomplete; not writing a manifest.", file=sys.stderr)
        return 1

    manifest_path = out_dir / "MANIFEST.txt"
    header = [
        "Zenodo deposit manifest for cpath-triage.",
        f"{len(files)} files, {total / 1048576:.1f} MB.",
        f"CNN checkpoints: {'included' if args.include_checkpoints else 'excluded'}.",
        f"Manuscript source: {'included' if args.include_paper else 'excluded'}.",
        "",
        "sha256                                                            bytes  path",
    ]
    manifest_path.write_text("\n".join(header + manifest) + "\n", encoding="utf-8")

    if args.zip:
        zip_path = out_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out_dir.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(out_dir).as_posix())
        print(f"Wrote: {zip_path}")

    incomplete = _incomplete_cnn_runs()
    if incomplete:
        print("Skipped as still in flight (checkpoint present, metrics.json absent):")
        for d in sorted(incomplete):
            print(f"  {d.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Excluded by policy: {len(skipped)} file(s) under results/.")
    print(f"Staged {len(files)} files, {total / 1048576:.1f} MB")
    print(f"Wrote: {out_dir}")
    print(f"Wrote: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
