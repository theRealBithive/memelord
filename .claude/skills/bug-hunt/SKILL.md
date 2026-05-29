---
name: bug-hunt
description: >-
  Find bugs in a diff using several parallel Claude sub-agents, then merge,
  verify, and report. This skill should be used when the user wants to
  "find bugs", "hunt for bugs", "review the diff / my changes / this PR for
  bugs", "do a parallel review", or "multi-agent code review". By default it
  reviews working-tree changes; accepts --ref BASE, --range A..B, or --pr N.
  Spawns N cold-start sub-agents (default 4) — this is deliberately expensive;
  do not run it on a trivial or empty diff.
---

# Bug Hunt — parallel multi-agent bug review

Dispatch several sub-agents at a diff, each hunting a different class of bug and
ranking findings critical/high/medium/low. When they finish, you (the
orchestrator) merge the findings, **verify each against the real code to drop
false positives**, and write a final report. That verification pass is the whole
point — it is what separates this from a single-pass review that floods the user
with plausible-but-wrong findings.

**Cost:** each sub-agent is a cold start that re-reads the diff and source. The
default is 4. Don't run on an empty/trivial diff (Step 1 guards this). Paths
below are relative to the repo root.

## Step 1 — Gather the diff (deterministic, no LLM)

```bash
bash .claude/skills/bug-hunt/scripts/gather-diff.sh
```

Modes (pass through from the user's invocation):

| Invocation | Command |
|---|---|
| working-tree changes *(default)* | `gather-diff.sh` |
| branch vs base | `gather-diff.sh --ref main` |
| arbitrary range | `gather-diff.sh --range HEAD~3..HEAD` |
| GitHub PR (needs `gh`) | `gather-diff.sh --pr 42` |

It writes `/tmp/bug-hunt/diff.patch`, `changed-files.txt`, and `summary.txt`,
and prints the summary. **Check the exit code: `2` means no changes — stop and
tell the user, do not dispatch agents.** Read `summary.txt`; if the diff is huge
(say >3000 changed lines) tell the user it may exceed a sub-agent's useful
context and offer to scope down with `--range`.

## Step 2 — Dispatch the sub-agents (default 4, parallel)

Use the **Agent** tool with `subagent_type: "general-purpose"`. Dispatch all
agents **in a single assistant turn** (independent tool calls run in parallel).
If the diff is large or agents are slow, instead pass `run_in_background: true`
on each and collect results as they complete.

Default split — one agent per focus area:

1. **Correctness & logic** — off-by-one, None/null handling, inverted/ wrong
   conditionals, control-flow gaps, unhandled edge cases, bad error handling,
   state/ordering mistakes, wrong data transforms.
2. **Security & data safety** — injection (SQL/command/template), missing
   authz/authn or permission checks, leaked secrets, unsafe deserialization,
   path traversal, SSRF, CSRF, missing input validation.
3. **Concurrency, resources & performance** — race conditions, deadlocks,
   unclosed files/connections, leaks, N+1 queries, unbounded growth/loops,
   blocking calls on hot/async paths, transaction misuse.
4. **API contracts, types & integration** — signature/return-type mismatches,
   breaking API or schema/migration changes, dependency/version issues, and
   missing test coverage for the changed code.

If the user asked for a different count (e.g. "run 6"), keep these categories
and split the broad ones (split #1 into logic vs error-handling; #3 into
concurrency vs performance) or merge down for fewer. Never fewer than 2.

Give **every** agent this prompt (fill in `<FOCUS>` and its bullet list). The
template is fenced with four backticks so the inner ```` ```yaml ```` block
survives copying:

````
You are one of several parallel bug-review agents. Review ONLY the change set,
focusing on: <FOCUS>.

The diff is at /tmp/bug-hunt/diff.patch and the changed files are listed in
/tmp/bug-hunt/changed-files.txt. Read the diff, then open the real source files
with Read/Grep/Glob for surrounding context. Read CLAUDE.md if present to learn
project conventions before judging something a bug.

Rules:
- Do NOT spawn sub-agents. Use Read/Grep/Glob directly.
- Do NOT edit, fix, or write any files. Report only.
- Report only genuine bugs in your focus area — not style, naming, or
  preferences. Prefer precision over volume; a wrong finding costs the
  reviewer more than a missed one.
- Anchor every finding to a real file:line that appears in the diff or its
  immediate context.

Output: a single fenced ```yaml block — a list of findings in EXACTLY this
shape (and nothing else after it):

```yaml
- severity: high          # critical | high | medium | low
  title: "one-line summary"
  file: "path/from/repo/root.py"
  line: "142"             # single line or "142-150"
  category: "<FOCUS slug>"
  evidence: |
    the exact offending line(s), copied verbatim
  impact: "what breaks, and the conditions that trigger it"
  fix: "concrete suggested change"
  confidence: medium       # high | medium | low
```

If you find nothing, output an empty YAML list — a single line containing `[]`
inside a ```yaml fence.
````

**Severity rubric** (tell agents to apply it):
- **critical** — data loss/corruption, security breach, or a crash/incorrect
  result on a common path.
- **high** — wrong behaviour on an important path, a security weakness needing
  specific conditions, or a resource leak that accrues in normal use.
- **medium** — edge-case bug, error path that's wrong but rarely hit.
- **low** — minor incorrectness, missing defensive guard, small inefficiency.

## Step 3 — Merge and verify (false-positive ruling)

Collect all findings. Deduplicate by `file`+`line`+root cause (different agents
often flag the same line). For **each** surviving finding, open the cited code
yourself (Read the file at that line; check project conventions in CLAUDE.md)
and classify:

- **Confirmed** — the code does what the finding claims; the bug is real.
- **Needs human verification** — plausible but you cannot prove it from the code
  alone (depends on runtime data, external service, caller you can't see). **Keep
  it**, tagged as such. Do not silently drop it.
- **False positive** — *only* when the cited code provably contradicts the claim
  (e.g. agent says "missing None check" but the line is `if x is None: return`,
  or the "unvalidated input" is validated two lines up). Record it with the
  one-line reason it was dismissed.

The bar to dismiss is high: provable contradiction, not "I couldn't reproduce."
When in doubt, downgrade confidence and keep it.

## Step 4 — Write the report

Write `bug-hunt-report.md` at the repo root (durable artifact), then print a
short summary to the user (counts per severity + the report path). This repo's
`.gitignore` already excludes `bug-hunt-report.md`; in a repo that doesn't,
mention it so the report isn't accidentally committed.

```markdown
# Bug Hunt Report

**Target:** <from summary.txt>  ·  **Agents:** <N>  ·  **Date:** <date>

## Summary
| Severity | Confirmed | Needs verification |
|---|---|---|
| Critical | n | n |
| High | n | n |
| Medium | n | n |
| Low | n | n |

## Findings
<ordered critical → low. For each:>
### [SEVERITY] Title  ·  `file:line`  ·  _confirmed / needs verification_
- **Impact:** …
- **Evidence:** the cited code
- **Fix:** …

## Dismissed (false positives)
<one line each: finding + why the code contradicts it>
```

## Gotchas

- **Empty diff = exit 2 from Step 1.** Always check it before spawning anything.
  The most common waste is dispatching 4 agents on a clean tree.
- **`--pr` needs `gh`** installed and authenticated; the other three modes need
  only git. If `gh` is absent the script exits 1 with that message.
- **Untracked files** are included in working-tree mode (appended as add-diffs);
  brand-new files do get reviewed.
- **Don't trust counts over correctness.** An agent that returns 20 findings is
  usually noisier, not better. The verify pass in Step 3 is mandatory, not
  optional polish.
- **Sub-agents must not fan out or edit.** The prompt forbids both; if you
  rewrite the prompt, keep those two rules or you risk runaway spawns / a
  review that mutates the code it's reviewing.
