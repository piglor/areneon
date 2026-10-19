# Issue tracker: Redmine (Internal Roadmap)

The internal roadmap, features, and tickets for this repo live in **Redmine** at `redmine.piglor.com/projects/areneon`.
Access is through the Redmine MCP server configured in the agent environment. Never commit Redmine API keys to this public repo.
Use `curl -H "X-Redmine-API-Key: $KEY"` for direct API access when MCP is unavailable.

> **Note on GitHub Issues**: Redmine is for internal team contributions. GitHub Issues (`piglor/areneon`) is for open source community contributions — the repo is public. Do not use GitHub Issues for internal work.

## Conventions

- **Create an issue**: `POST /issues.json` with `{ "issue": { "project_id": "areneon", "subject": "...", "description": "..." } }`
- **Read an issue**: `GET /issues/<number>.json?include=journals,attachments`
- **List issues**: `GET /projects/areneon/issues.json?status_id=open&limit=25`
- **Comment on an issue**: `PUT /issues/<number>.json` with `{ "issue": { "notes": "..." } }`
- **Apply a label/status**: `PUT /issues/<number>.json` with `{ "issue": { "status_id": <id> } }`
- **Close**: `PUT /issues/<number>.json` with `{ "issue": { "status_id": 5 } }` (status 5 = Closed in default Redmine)

## When a skill says "publish to the issue tracker"

Create a Redmine issue via `POST /issues.json`.

## When a skill says "fetch the relevant ticket"

Run `GET /issues/<number>.json?include=journals` via the Redmine MCP.

## ADRs

The ADR text is a wiki page on Redmine, parented under `ADR`. Redmine wiki pages accept **Markdown** (not Textile). Create them at:
`PUT /projects/areneon/wiki/ADR-XXX_<Title>.json` with `{ "wiki_page": { "text": "<markdown content>", "parent_title": "ADR", "comments": "..." } }`

Each ADR has a standalone issue; once Accepted, its implementation Epic goes on the roadmap:

| Record | Tracker | Parent | Fixed version |
|---|---|---|---|
| `[Epic][ADR-NNN] <title>` (created after Accepted) | Epic (9) | — | Wave roadmap version |
| `[ADR-NNN] <title>` | ADR (5), Proposed (7) → Under Review (8) → Accepted (9) / Rejected (6) / Superseded (10) | — (standalone; related to its Epic) | none |
| Implementation tickets | Feature (2) / Task (4) / Bug (1) | the Epic | none (inherits) |

Reviews are journal notes on the ADR issue. Only ADR pages go under the `ADR` wiki parent.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a Redmine issue with child issues as tickets.

- **Map**: a Redmine issue with subject `[Wayfinder] <effort>` holding Notes / Decisions-so-far / Fog in the description.
- **Child ticket**: a Redmine issue with `parent_issue_id` set to the map issue number.
- **Blocking**: set relation via `POST /issues/<n>/relations.json` with `{ "relation": { "issue_to_id": <depends-on>, "relation_type": "blocks" } }`. Example: `#116 blocks #117` → `POST /issues/116/relations.json` with `"issue_to_id": 117, "relation_type": "blocks"`.
- **Claim**: `PUT /issues/<n>.json` with `{ "issue": { "assigned_to_id": "me" } }`
- **Resolve**: add a journal note with the answer, then close the issue.
