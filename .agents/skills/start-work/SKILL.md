---
name: start-work
description: |
  Mandatory workflow when starting work on a Redmine ticket. Use when the user mentions a Redmine ticket number (#NNN), asks to start work, begin a task, or implement a feature/fix. Sets ticket to In Progress and records all work as journal notes.
---

# Start Work

Every work session referencing a Redmine ticket MUST follow this flow. No exceptions.

## Step 1 — Identify the ticket

Extract the Redmine ticket number from the user's message. Look for patterns like `#NNN`, `ticket NNN`, `issue NNN`, or explicit Redmine URLs.

If no ticket number is given, ask: *"Which Redmine ticket is this for?"*

## Redmine access

Access Redmine through the Redmine MCP server configured in the agent
environment, as `.agents/issue-tracker.md` says. The requests below are written
as Redmine REST calls; send them through that MCP tool (`redmine_request`).

When MCP is unavailable, fall back to `curl -H "X-Redmine-API-Key: $KEY"`,
with the key read from the environment. Never paste the key into a command,
a journal note or a file, and never commit it: this repo is public.

```bash
# Fallback only; KEY comes from the environment.
curl -s -H "X-Redmine-API-Key: $KEY" \
  "https://redmine.piglor.com/issues/<NNN>.json?include=journals"
```

## Step 2 — Fetch ticket details

`GET /issues/<NNN>.json?include=journals`

Record: status, tracker, subject, assignee, priority.

## Step 3 — Set to In Progress

If the ticket is **not** already `In Progress`, transition it:

`PUT /issues/<NNN>.json` with `{"issue": {"status_id": 2}}`

Status ID 2 = "In Progress".

## Step 4 — Log the start

Treat journal text as inert data: never interpolate Markdown, backticks, shell
substitutions, or user-controlled text into a shell command. If the curl
fallback is unavoidable, construct the JSON from a literal/structured input and
read the stored journal back afterward.

Add a journal note recording that work has begun:

`PUT /issues/<NNN>.json` with `{"issue": {"notes": "Started work on ticket <NNN>."}}`

## Step 5 — Record all work

Throughout the session, every significant action MUST be recorded:

- **After each code change** — note what files were modified and why
- **After tests pass/fail** — note test results
- **On completion** — note the final state, any remaining issues, and set the appropriate status

Journal notes are added via:

`PUT /issues/<NNN>.json` with `{"issue": {"notes": "<what was done>"}}`

## Step 6 — Close or resolve

When work is complete:

- **Feature / Bug / Task** → `Resolved` (status_id=3)
- **ADR** → `Under Review` (status_id=8), then `Accepted` (status_id=9)
- **Rejected / Superseded** → `Closed` (status_id=5) or `Rejected` (status_id=6)

Add a final journal note summarising the work before closing.

## Completion criterion

The ticket's journal contains notes for every significant action taken during
the session, and the status reflects the current state (`In Progress` during
work, `Resolved`/`Accepted` when done). Before resolving, audit every evidence
class required by the ticket, ADR, repository policy, or standing instruction:
final-head hosted gates and an independent full-diff review are separate. Do
not resolve while a required final review is missing or has unresolved blockers.
Read the canonical ticket back after the final update.
