# State Schema

This document is the canonical field reference for `orchestrator/state.json`.
Other control-plane documents may point here, but should not duplicate the full
schema table.

Use a small machine-oriented state file that supports safe serial defaults plus
explicit parallel execution.

`roadmap_id` naming rule:

- `YYYY-MM-DD` is the roadmap family's creation date in the repo-local controller context.
- `NN` is the zero-based roadmap-family ordinal for that day: `00`, `01`, `02`, and so on.
- `<slug>` is the stable descriptive control-plane family slug.
- Runtime should preserve this value verbatim once scaffolded; do not recompute it from the roadmap title later.

## Top-Level Fields

| Key | Type | Description |
|-----|------|-------------|
| `contract_version` | string | Control-plane contract version. New scaffolds use `orchestrator-v2` |
| `roadmap_style` | string | Roadmap shape for the active family: `strategy-backlog` |
| `base_branch` | string | Branch that accepted rounds merge into |
| `roadmap_id` | string | Stable family identifier for the active roadmap, normally `YYYY-MM-DD-NN-<slug>` |
| `roadmap_revision` | string | Active roadmap revision (e.g. `rev-001`) |
| `roadmap_dir` | string | Path to the active roadmap bundle directory |
| `controller_stage` | string | Controller stage: `dispatch-rounds`, `update-roadmap`, `done`, or `blocked`; new scaffolds with unfinished roadmap work start at `dispatch-rounds` |
| `max_parallel_rounds` | number | Hard cap on concurrent live rounds (default `1` for serial) |
| `active_rounds` | array | Canonical live round records (see below) |
| `roadmap_update` | object or null | Active `update-roadmap` branch/worktree/artifact state, or `null` when no roadmap update is in progress |
| `last_completed_round` | string or null | Most recently merged round id |
| `resume_errors` | object | Structured controller recoverable errors, including `controller` for controller-level blockage |
| `retry` | object or null | Controller-owned retry subloop state; `null` when no retry is active |

## Round Record Fields (each entry in `active_rounds`)

| Key | Type | Description |
|-----|------|-------------|
| `round_id` | string | Stable round identifier |
| `milestone_id` | string | Selected milestone that supplied the extracted work |
| `direction_id` | string | Candidate direction chosen by the guider |
| `extracted_item_id` | string | Stable id for the concrete round item derived from the selected direction |
| `selected_roadmap_revision` | string | Revision the round was selected from |
| `selected_roadmap_dir` | string | Bundle path the round was selected from |
| `current_task` | string | Human-readable summary of the extracted round scope |
| `stage` | string | One of: `select-task`, `plan`, `implement`, `review`, `pending-merge`, `merge`, `done`, `blocked` |
| `branch` | string | Canonical round branch |
| `worktree_path` | string | Canonical round worktree |
| `active_round_dir` | string | Repo-relative canonical round artifact directory, resolved against `worktree_path` while the round is live and against the parent checkout only after merge |
| `round_artifacts` | object | Canonical round artifact path map; repo-relative paths are resolved against `worktree_path` while live |
| `depends_on_round_ids` | array | Round ids that must merge before this round may merge |
| `merge_after_item_ids` | array | Extracted item ids that must merge before this round may merge |
| `parallel_group` | string or null | Co-scheduling group for this round |
| `worker_mode` | string | `none`, `fanout`, or `integrate` |
| `worker_records` | object | Worker state keyed by worker id, following `orchestrator/worker-plan-schema.md` |
| `merge_ready` | boolean | Whether this round may legally enter `merge` now |
| `resume_error` | string or null | Per-round recoverable error |

## Roadmap Update Record Fields (`roadmap_update`)

| Key | Type | Description |
|-----|------|-------------|
| `source_round_id` | string | Merged round that triggered this roadmap update |
| `branch` | string | Canonical roadmap-update branch |
| `worktree_path` | string | Canonical roadmap-update worktree |
| `update_artifact` | string | Repo-relative `roadmap-update.md` path |
| `review_artifact` | string | Repo-relative `roadmap-update-review.md` path |
| `prior_roadmap_revision` | string | Active revision before the update |
| `proposed_roadmap_revision` | string | Proposed active revision after the update |
| `status` | string | One of: `authoring`, `review`, `approved`, `blocked` |
| `resume_error` | string or null | Recoverable roadmap-update error |

## Invariants

When no roadmap update is in progress, `roadmap_update` should be `null`.
Pending-merge queues are derived from live round records whose `stage` is
`pending-merge`.

Round lineage comes from `milestone_id`, `direction_id`, and
`extracted_item_id`.
