# Retry Subloop Policy

Active bundle: `orchestrator/roadmaps/2026-05-16-00-sgnav-isaac-benchmark-completion/rev-001/`.

This file records only repo- and roadmap-specific retry policy. Keep shared
runtime mechanics in the runtime skill references and controller state schema.

## Same-Round Retry

- Default: disabled.
- Enabled extracted items:
  - none
- Enabled review outcomes:
  - none
- Worker-slice retry:
  - disabled unless `worker-plan.json` explicitly enables it for the current
    round.
- SG-Nav benchmark safety:
  - a retry may fix an implementation or test failure, but it must not make a
    failing metric run pass by switching YOLO-World/SAM2 to dry-run, seeded GT
    memory, distance-only navigation, static-map shortcuts, or missing-model
    stubs unless the extracted item explicitly targets a non-metric/debug path
    and the output is labeled as such.
- Missing external assets:
  - Isaac, InteriorAgent, YOLO-World, SAM2, or LLM service absence should become
    a clear skip or clear setup error; retry is not authorized to hide the
    absence with a metric-counted fallback.

## Pending-Merge Policy

- A reviewed round may pause in `pending-merge` only for declared dependency
  ordering, extracted-item ordering, or base-branch freshness.
- If a `pending-merge` round needs substantive code refresh, return it to the
  owner named by the current round state: whole-round implementer when
  `worker_mode` is `none`, integration implementer when worker fan-out is
  active.

## Roadmap Revision Rule

- If `update-roadmap` changes future coordination, sequencing, boundaries, or
  retry meaning, publish a new roadmap revision directory instead of rewriting a
  used revision.

## Roadmap Overrides

- Keep controller execution serial by default (`max_parallel_rounds: 1`) until a
  later reviewed roadmap update proves non-overlapping ownership for concurrent
  rounds.
