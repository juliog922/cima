#!/usr/bin/env bash
# Reject AI/conversational and session-narrative comment phrasing. Comments
# should state contracts and invariants; incident history belongs in commit
# messages. Runs in CI (Lane A).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# Phrases that betray conversational authorship or change-diary comments.
patterns=(
  "as you can see"
  "we can see"
  "let's "
  "let us "
  "I've "
  "I have "
  "field-observed"
  "earlier revision"
  "previously read"
  "in an earlier"
  "used to be"
  "TODO"
  "FIXME"
  "HACK"
)
fail=0
for p in "${patterns[@]}"; do
  # Search only comment lines in Rust/CUDA sources.
  hits=$(grep -rniE "(//|/\*|\*).*${p}" src --include='*.rs' --include='*.cu' || true)
  if [[ -n "$hits" ]]; then
    echo "banned comment phrase: '${p}'"
    echo "$hits"
    fail=1
  fi
done
[[ $fail == 0 ]] && echo "comment lint: clean"
exit $fail