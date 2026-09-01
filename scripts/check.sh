#!/usr/bin/env bash
# scripts/check.sh
# Lint + integrity scan. Run before every commit. Exits non-zero on any failure.
set -uo pipefail

fail=0

# Prefer the project interpreter so the check does not silently skip when ruff is
# installed in the venv but not on PATH, which is how it used to pass by default.
# The interpreter can live outside the repo (see .venv/VENV.txt), so allow an
# explicit override before falling back to the usual locations.
PY_BIN=""
VENV_HINT=""
if [ -f .venv/VENV.txt ]; then
  VENV_HINT="$(tr -d '
"' < .venv/VENV.txt | sed 's/^Use *//')"
fi
for cand in "${PYTHON:-}"             "${VIRTUAL_ENV:-}/Scripts/python.exe" "${VIRTUAL_ENV:-}/bin/python"             "${VENV_HINT:+$VENV_HINT/Scripts/python.exe}"             "${VENV_HINT:+$VENV_HINT/bin/python}"             ".venv/Scripts/python.exe" ".venv/bin/python"             "$(command -v python3 || true)" "$(command -v python || true)"; do
  if [ -n "$cand" ] && [ -x "$cand" ]; then PY_BIN="$cand"; break; fi
done
[ -n "$PY_BIN" ] && echo "python: $PY_BIN"

echo "== ruff lint =="
if [ -n "$PY_BIN" ] && "$PY_BIN" -m ruff --version >/dev/null 2>&1; then
  "$PY_BIN" -m ruff check src scripts tests || fail=1
elif command -v ruff >/dev/null 2>&1; then
  ruff check src scripts tests || fail=1
else
  echo "ruff not found. Install it: pip install -r requirements.txt"
  fail=1
fi

echo
echo "== router tie-handling regression tests =="
if [ -n "$PY_BIN" ] && "$PY_BIN" -m pytest --version >/dev/null 2>&1; then
  "$PY_BIN" -m pytest tests -q || fail=1
else
  echo "pytest not found. Install it: pip install -r requirements.txt"
  fail=1
fi

echo
echo "== forbidden AI phrases in docs and paper prose =="
# These phrases are banned by the project writing rules. Scan markdown and tex.
PHRASES='it is worth noting|it is important to note|plays a (vital|key|crucial|important) role|delve|delves into|sheds light on|underscores|leverages\b|seamless|groundbreaking|state-of-the-art'
if grep -rEin --include='*.md' --include='*.tex' "$PHRASES" docs paper README.md 2>/dev/null; then
  echo "Forbidden phrase(s) found above. Rewrite per skills/research-paper-writing."
  fail=1
else
  echo "clean"
fi

echo
echo "== em dash scan =="
if grep -rn --include='*.md' --include='*.tex' --include='*.py' $'\xe2\x80\x94' docs paper src scripts README.md 2>/dev/null; then
  echo "Em dash(es) found above. Replace with comma, period, or parentheses."
  fail=1
else
  echo "clean"
fi

echo
echo "== fabricated-number guard =="
# Any results table cell that should hold a metric must either be a real number
# or an explicit TODO. Flag suspicious 'XX', '0.00 (placeholder)', 'TBD-as-number'.
if grep -rEinI 'placeholder|XX\.X|\bTBD\b' paper docs/results* results 2>/dev/null; then
  echo "Possible placeholder metric above. Leave blank or mark TODO, never a fake number."
  fail=1
else
  echo "clean"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "ALL CHECKS PASSED"
else
  echo "CHECKS FAILED"
fi
exit $fail
