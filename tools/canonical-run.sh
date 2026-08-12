#!/bin/bash
# Run any command with the release's pinned data-science dependencies.
#
# Usage: tools/canonical-run.sh <command...>
#   tools/canonical-run.sh python script.py
#   tools/canonical-run.sh python task/validity_harness.py

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WITH_ARGS=()
while IFS= read -r pin; do
  [[ -z "$pin" || "$pin" == \#* ]] && continue
  WITH_ARGS+=(--with "$pin")
done < "$ROOT/canonical-env.txt"

exec uv run --quiet --no-project "${WITH_ARGS[@]}" "$@"
