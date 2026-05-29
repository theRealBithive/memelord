#!/usr/bin/env bash
# gather-diff.sh — collect the change set the bug-hunt sub-agents will review.
#
# Why this exists: the orchestrator and all N sub-agents must look at the *same*
# diff, and the diff has to survive across separate agent processes. So we
# materialise it once to files under a temp dir (default /tmp/bug-hunt) rather
# than passing a giant patch inline in every agent prompt — cold-start agents
# read the file and open the real source files for context. We write to /tmp,
# not the repo, so a review never dirties the working tree it is reviewing.
#
# Three modes (deterministic, no LLM):
#   (none)        working-tree changes  — staged + unstaged tracked edits
#                 (git diff HEAD) plus any untracked files, appended as
#                 add-diffs via `git diff --no-index`. This is the default
#                 because the common case is "review what I'm about to commit".
#   --ref BASE    branch review         — git diff BASE...HEAD (merge-base
#                 three-dot, i.e. "what this branch added since BASE").
#   --range R     arbitrary range       — git diff R (e.g. HEAD~3..HEAD).
#   --pr N        GitHub PR             — gh pr diff N. Requires the gh CLI to
#                 be installed and authenticated; the other modes do not.
#
# Outputs (under $BUG_HUNT_DIR, default /tmp/bug-hunt):
#   diff.patch         the unified diff every agent reads
#   changed-files.txt  one path per changed file
#   summary.txt        git diffstat + mode/target header
#
# Exit codes: 0 = diff written; 2 = no changes to review (orchestrator must
# check this and NOT dispatch agents on an empty diff); 1 = usage/tool error.

set -euo pipefail

OUT_DIR="${BUG_HUNT_DIR:-/tmp/bug-hunt}"
mode="worktree"
target=""

while [ $# -gt 0 ]; do
  case "$1" in
    --ref)   mode="ref";   target="${2:-}"; shift 2 ;;
    --range) mode="range"; target="${2:-}"; shift 2 ;;
    --pr)    mode="pr";    target="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,33p' "$0"; exit 0 ;;
    *)
      echo "gather-diff.sh: unknown argument '$1'" >&2
      echo "usage: gather-diff.sh [--ref BASE | --range A..B | --pr N]" >&2
      exit 1 ;;
  esac
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "gather-diff.sh: not inside a git work tree" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
patch="$OUT_DIR/diff.patch"
: > "$patch"

case "$mode" in
  worktree)
    label="working-tree changes (staged + unstaged + untracked)"
    git diff HEAD >> "$patch"
    # Untracked files are invisible to `git diff HEAD`; append each as an
    # add-diff. --no-index exits 1 when it finds differences (always, here),
    # so guard with `|| true` to keep `set -e` from aborting the loop.
    while IFS= read -r f; do
      [ -n "$f" ] || continue
      git diff --no-index -- /dev/null "$f" >> "$patch" 2>/dev/null || true
    done < <(git ls-files --others --exclude-standard)
    ;;
  ref)
    [ -n "$target" ] || { echo "gather-diff.sh: --ref needs a BASE" >&2; exit 1; }
    label="branch diff: ${target}...HEAD"
    git diff "${target}...HEAD" >> "$patch"
    ;;
  range)
    [ -n "$target" ] || { echo "gather-diff.sh: --range needs A..B" >&2; exit 1; }
    label="range diff: ${target}"
    git diff "$target" >> "$patch"
    ;;
  pr)
    [ -n "$target" ] || { echo "gather-diff.sh: --pr needs a number" >&2; exit 1; }
    command -v gh >/dev/null 2>&1 || {
      echo "gather-diff.sh: --pr mode needs the gh CLI (not installed)" >&2
      exit 1
    }
    label="GitHub PR #${target}"
    gh pr diff "$target" >> "$patch"
    ;;
esac

if [ ! -s "$patch" ]; then
  echo "gather-diff.sh: no changes to review ($label)" >&2
  exit 2
fi

# Derive the file list and a human summary from the patch itself so all three
# modes share one code path (diffstat from a saved patch needs --stat on apply).
grep -E '^\+\+\+ b/|^diff --git' "$patch" \
  | sed -E 's#^diff --git a/.* b/##; s#^\+\+\+ b/##' \
  | grep -v '^/dev/null$' | sort -u > "$OUT_DIR/changed-files.txt"

added=$(grep -cE '^\+[^+]' "$patch" || true)
removed=$(grep -cE '^-[^-]' "$patch" || true)
nfiles=$(wc -l < "$OUT_DIR/changed-files.txt" | tr -d ' ')

{
  echo "bug-hunt target: $label"
  echo "files changed: $nfiles   (+$added / -$removed lines)"
  echo "diff:  $patch"
  echo "files: $OUT_DIR/changed-files.txt"
  echo
  echo "changed files:"
  sed 's/^/  /' "$OUT_DIR/changed-files.txt"
} > "$OUT_DIR/summary.txt"

cat "$OUT_DIR/summary.txt"
