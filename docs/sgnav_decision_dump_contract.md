# SG-Nav Decision Dump Contract

`python -m isaac_bench.scripts.dump_sgnav_step ...` writes a JSON artifact that
captures one SG-Nav decision surface. It can write an explicit schema-only
smoke artifact, or convert a runtime `graph_step_*.json` plus an optional
episode result row into the same contract shape.

## Required JSON Keys

Every dump artifact must include:

- `objects`
- `groups`
- `rooms`
- `room_context`
- `room_segmentation`
- `room_semantics`
- `edges`
- `subgraphs`
- `subgraph_texts_or_payloads`
- `llm_scores`
- `subgraph_probabilities`
- `frontier_scores`
- `selected_frontier`
- `candidate_goals`
- `reperception_state`
- `stop_state`

## Semantics

- `objects`, `groups`, `rooms`, and `edges` describe the online scene graph
  state used by the decision.
- `room_context` records whether room context was invoked for frontier scoring,
  whether segmentation or VLM labeling ran, cache hit state, label request/cache
  counts, and the testable call order
  `frontier_extraction -> room_context_for_frontier_scoring -> scenegraph_update
  -> hcot_subgraph_scoring -> frontier_interpolation -> frontier_selection`.
- `room_segmentation` records online geometry room masks, doorway metadata,
  partial state, and mask confidence.
- `room_semantics` records VLM/LLM room labels, allowed categories, `unknown`
  reasons, `vlm_self_confidence`, evidence-derived `label_reliability`,
  reliability factors, and backend state.
- `subgraphs` and `subgraph_texts_or_payloads` capture the SG-Nav-compatible
  reasoning input.
- `llm_scores` and `subgraph_probabilities` capture scorer output before
  frontier interpolation.
- `frontier_scores` and `selected_frontier` capture exploration target choice.
- `candidate_goals`, `reperception_state`, and `stop_state` capture candidate
  confirmation state, accumulated credibility, last `s_k`, supporting subgraphs,
  STOP eligibility, and STOP blocking reasons.

Empty lists or objects are acceptable only for smoke/schema dumps that are
explicitly not metric benchmark decisions.

## Runtime Conversion

Use `--graph-debug-dump debug/graphs/graph_step_000123.json` to populate the
scene graph, subgraph, frontier, and decision fields from a real run. Use
`--result-row data/isaac_bench_runs/.../results.jsonl` to enrich candidate,
re-perception, and STOP state from the episode row. The dump remains a debug
artifact; it does not make a run metric-valid.
