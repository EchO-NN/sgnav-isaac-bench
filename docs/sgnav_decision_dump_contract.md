# SG-Nav Decision Dump Contract

`python -m isaac_bench.scripts.dump_sgnav_step ...` writes a JSON artifact that
captures one SG-Nav decision surface. The command is a schema contract and
debugging entrypoint; it must not be used to hide missing runtime assets.

## Required JSON Keys

Every dump artifact must include:

- `objects`
- `groups`
- `rooms`
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
- `subgraphs` and `subgraph_texts_or_payloads` capture the SG-Nav-compatible
  reasoning input.
- `llm_scores` and `subgraph_probabilities` capture scorer output before
  frontier interpolation.
- `frontier_scores` and `selected_frontier` capture exploration target choice.
- `candidate_goals`, `reperception_state`, and `stop_state` capture candidate
  confirmation state and STOP eligibility.

Empty lists or objects are acceptable only for smoke/schema dumps that are
explicitly not metric benchmark decisions.
