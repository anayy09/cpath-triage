"""
scripts/diff_inject_old_labels.py

Give labels that exist only in the old source their old numbers inside a
latexdiff output, so struck-out text that cites a removed figure, table or
section prints the number it had rather than ??.

Each label is defined in the preamble (\\newlabel is preamble-only) with the
number and page it had in the old build's .aux, pointing at the document start.

Usage:
    python scripts/diff_inject_old_labels.py diff.tex old/main.aux fig:a tab:b ...
"""
import re
import sys
from pathlib import Path


def main() -> int:
    diff, old_aux = Path(sys.argv[1]), Path(sys.argv[2])
    keys = sys.argv[3:]
    aux = old_aux.read_text(encoding="utf-8", errors="ignore")
    lines = []
    for k in keys:
        m = re.search(r"\\newlabel\{" + re.escape(k) + r"\}\{\{([^}]*)\}\{([^}]*)\}", aux)
        if not m:
            print(f"{k} not in {old_aux}", file=sys.stderr)
            return 1
        lines.append(rf"\newlabel{{{k}}}{{{{{m.group(1)}}}{{{m.group(2)}}}{{}}{{Doc-Start}}{{}}}}")
    with open(diff, encoding="utf-8", newline="") as f:
        s = f.read()
    eol = "\r\n" if "\r\n" in s else "\n"
    anchor = r"\begin{document}"
    if s.count(anchor) != 1:
        print(f"expected one {anchor} in {diff}", file=sys.stderr)
        return 1
    block = eol.join([r"\makeatletter", *lines, r"\makeatother", anchor])
    with open(diff, "w", encoding="utf-8", newline="") as f:
        f.write(s.replace(anchor, block, 1))
    print("injected:", ", ".join(keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
