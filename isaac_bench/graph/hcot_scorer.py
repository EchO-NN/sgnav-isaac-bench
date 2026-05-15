from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from isaac_bench.graph.edge_builder import LLMClient
from isaac_bench.graph.subgraph_builder import Subgraph


@dataclass
class SubgraphScore:
    subgraph_id: str
    central_object_id: str
    goal_category: str
    estimated_distance_m: float
    p_sub: float
    summary_reason: str
    raw_llm_response: dict
    central_world: np.ndarray


class HCoTSubgraphScorer:
    def __init__(self, llm_client: Optional[LLMClient] = None, min_distance_m: float = 0.25, max_retries: int = 2):
        self.llm_client = llm_client
        self.min_distance_m = max(1e-3, float(min_distance_m))
        self.max_retries = max(0, int(max_retries))
        self._cache: Dict[str, SubgraphScore] = {}

    def score(self, subgraphs: List[Subgraph], goal_category: str, graph_version: int = 0) -> List[SubgraphScore]:
        scores = [self.score_subgraph(subgraph, goal_category, graph_version=graph_version) for subgraph in subgraphs]
        if not scores:
            return []
        raw = np.asarray([score.p_sub for score in scores], dtype=np.float32)
        hi = float(np.max(raw))
        if hi > 0.0:
            for score in scores:
                score.p_sub = float(score.p_sub / hi)
        return scores

    def score_subgraph(self, subgraph: Subgraph, goal_category: str, graph_version: int = 0) -> SubgraphScore:
        key = "%s|%s|%s" % (graph_version, goal_category, subgraph.id)
        if key in self._cache:
            return self._cache[key]
        if self.llm_client is None:
            result = self._fallback_score(subgraph, goal_category)
        else:
            result = score_subgraph_with_hcot(subgraph, goal_category, self.llm_client, self.min_distance_m, self.max_retries)
        self._cache[key] = result
        return result

    def _fallback_score(self, subgraph: Subgraph, goal_category: str) -> SubgraphScore:
        same_category = subgraph.central_object_category == goal_category
        related = any(goal_category in str(node.get("category", "")) for node in subgraph.nodes)
        distance = self.min_distance_m if same_category else (1.0 if related else 4.0)
        return SubgraphScore(
            subgraph_id=subgraph.id,
            central_object_id=subgraph.central_object_id,
            goal_category=goal_category,
            estimated_distance_m=float(distance),
            p_sub=float(1.0 / max(distance, self.min_distance_m)),
            summary_reason="deterministic fallback subgraph distance estimate",
            raw_llm_response={"fallback": True},
            central_world=np.asarray(subgraph.central_world, dtype=np.float32).copy(),
        )


def score_subgraph_with_hcot(
    subgraph: Subgraph,
    goal_category: str,
    llm_client: LLMClient,
    min_distance_m: float = 0.25,
    max_retries: int = 2,
) -> SubgraphScore:
    prior = _call_json(
        llm_client,
        (
            "Goal object: %s\nCentral object: %s\n"
            "Predict the most likely distance between the central object and the goal object in an indoor environment.\n"
            "Return strict JSON: {\"prior_distance_m\": float, \"reason\": \"short explanation\"}."
        )
        % (goal_category, subgraph.central_object_category),
        max_retries,
    )
    questions = _call_json(
        llm_client,
        (
            "Goal object: %s\nCentral object: %s\n"
            "Ask useful questions about the central object and the goal object for predicting their distance.\n"
            "Return strict JSON: {\"questions\": [\"...\", \"...\", \"...\"]}."
        )
        % (goal_category, subgraph.central_object_category),
        max_retries,
    )
    answers = _call_json(
        llm_client,
        (
            "Goal object: %s\nCentral object: %s\nSubgraph nodes:\n%s\nSubgraph edges:\n%s\nQuestions:\n%s\n"
            "Answer the questions using only the subgraph. Return strict JSON: {\"answers\": [{\"question\": \"...\", \"answer\": \"...\"}]}."
        )
        % (
            goal_category,
            subgraph.central_object_category,
            json.dumps(subgraph.nodes, ensure_ascii=False),
            json.dumps(subgraph.edges, ensure_ascii=False),
            json.dumps(questions.get("questions", []), ensure_ascii=False),
        ),
        max_retries,
    )
    final = _call_json(
        llm_client,
        (
            "Goal object: %s\nCentral object: %s\nPrior distance:\n%s\nQuestion-answer evidence:\n%s\n"
            "Determine the most likely distance between this subgraph and the goal object.\n"
            "Return strict JSON: {\"estimated_distance_m\": float, \"confidence\": 0.0-1.0, \"summary_reason\": \"short explanation\"}."
        )
        % (goal_category, subgraph.central_object_category, json.dumps(prior, ensure_ascii=False), json.dumps(answers, ensure_ascii=False)),
        max_retries,
    )
    estimated_distance = float(final.get("estimated_distance_m", final.get("distance", prior.get("prior_distance_m", 4.0))))
    p_sub = 1.0 / max(float(estimated_distance), float(min_distance_m))
    return SubgraphScore(
        subgraph_id=subgraph.id,
        central_object_id=subgraph.central_object_id,
        goal_category=goal_category,
        estimated_distance_m=float(estimated_distance),
        p_sub=float(p_sub),
        summary_reason=str(final.get("summary_reason", final.get("reason", ""))),
        raw_llm_response={"prior": prior, "questions": questions, "answers": answers, "final": final},
        central_world=np.asarray(subgraph.central_world, dtype=np.float32).copy(),
    )


def _call_json(llm_client: LLMClient, prompt: str, max_retries: int) -> dict:
    last_error: Optional[Exception] = None
    for _ in range(max(1, int(max_retries) + 1)):
        try:
            return _parse_json_object(llm_client.complete_json(prompt))
        except Exception as exc:
            last_error = exc
            prompt = "Repair the previous answer and return strict JSON only.\n" + prompt
    raise ValueError("failed to parse strict JSON response") from last_error


def _parse_json_object(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match is None:
                raise
            parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("response must be a JSON object")
