---
name: implement
description: "Implement a piece of work based on a spec or set of tickets."
disable-model-invocation: true
---

Implement the work described by the user in the spec or tickets.

Use /tdd where possible, at pre-agreed seams.

## Quality gates

Run the gates for the language you touched after every change, review fixes included: `make python`, `make ts`, or `make check` (see AGENTS.md, Quality gates). Fix a failing gate before moving on.

**Coverage rule:** If new code cannot be covered by a test, **remove it** — do not leave dead code. Prefer deleting or simplifying untestable branches over coverage exemptions.

Once done, use /code-review to review the work, then re-run gates after applying review fixes.

Commit your work to the current branch.
