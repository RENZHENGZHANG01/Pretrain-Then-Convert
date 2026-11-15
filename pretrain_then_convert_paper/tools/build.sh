#!/bin/bash
# Build the paper (pdflatex + bibtex on PATH) in a scratch directory and check the 8-page limit.
#   tools/build.sh [build dir]        (default: build/)  ->  <build dir>/main.pdf
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=${1:-$ROOT/build}
mkdir -p "$B"
cp -r "$ROOT"/paper/. "$B"/
cd "$B"
pdflatex -interaction=nonstopmode main.tex > build.log 2>&1 || true
bibtex main > bib.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1 || true
if grep -E "^!" build.log; then exit 1; fi
grep -E "(Reference|Citation).*undefined" main.log && exit 1
echo "built $B/main.pdf"
python3 "$ROOT/tools/check_pages.py" "$B"
