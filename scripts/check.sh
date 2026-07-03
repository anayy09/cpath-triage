#!/usr/bin/env bash
# scripts/check.sh
# Lint + integrity scan. Run before every commit. Exits non-zero on any failure.
set -uo pipefail

fail=0

echo "== ruff lint =="
if command -v ruff >/dev/null 2>&1; then
  ruff check src scripts || fail=1
else
  echo "ruff not found; skipping lint (install: pip install ruff)"
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
if grep -rEin 'placeholder|XX\.X|\bTBD\b' paper docs/results* results 2>/dev/null; then
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
