#!/usr/bin/env bash
# PAI fork: rebase pai-main onto upstream/main and re-run tests before pushing.
#
# Run this manually before every planned release. We don't auto-rebase.
set -euo pipefail

git fetch upstream
git checkout pai-main
git rebase upstream/main

echo
echo "Rebase clean. Running pytest..."
python -m pytest tests/ -q

echo
echo "All green. Review then push with:"
echo "  git push --force-with-lease origin pai-main"
echo
echo "Note: --force-with-lease (not --force) prevents stomping any commits that"
echo "landed on origin/pai-main since your last fetch."
