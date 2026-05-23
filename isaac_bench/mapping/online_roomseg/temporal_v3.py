from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class TemporalSeparatorState:
    key: tuple[int, int, int]
    score_ema: float
    seen_count: int
    last_step: int
    metadata: dict = field(default_factory=dict)


class TemporalRoomSegTrackerV3:
    def __init__(self, config: Mapping[str, object] | None = None):
        self.config = dict(config or {})
        self.separators: dict[tuple[int, int, int], TemporalSeparatorState] = {}

    def update_separator_scores(self, candidates, *, step: int) -> dict:
        if not bool(self.config.get("enabled", True)):
            return {"enabled": False, "tracked_separator_count": int(len(self.separators))}
        alpha = float(self.config.get("separator_score_ema_alpha", 0.40))
        hard_seen = int(self.config.get("separator_seen_count_min", 2))
        hard_score = float(self.config.get("separator_hard_accept_score_min", 0.60))
        accepted_by_temporal = 0
        for candidate in candidates:
            key = _candidate_key(candidate)
            score_new = float(getattr(candidate, "final_score", 0.0) or getattr(candidate, "confidence", 0.0))
            prev = self.separators.get(key)
            if prev is None:
                state = TemporalSeparatorState(key=key, score_ema=score_new, seen_count=1, last_step=int(step))
            else:
                state = TemporalSeparatorState(
                    key=key,
                    score_ema=float(alpha * score_new + (1.0 - alpha) * float(prev.score_ema)),
                    seen_count=int(prev.seen_count) + 1,
                    last_step=int(step),
                    metadata=dict(prev.metadata),
                )
            self.separators[key] = state
            candidate.temporal_score = float(np.clip(state.score_ema, 0.0, 1.0))
            candidate.debug["temporal_score_ema"] = float(state.score_ema)
            candidate.debug["temporal_seen_count"] = int(state.seen_count)
            if state.score_ema >= hard_score and state.seen_count >= hard_seen:
                candidate.debug["temporal_hard_accept"] = True
                accepted_by_temporal += 1
        return {
            "enabled": True,
            "tracked_separator_count": int(len(self.separators)),
            "temporal_hard_accept_count": int(accepted_by_temporal),
        }


def _candidate_key(candidate) -> tuple[int, int, int]:
    p0 = np.rint(np.asarray(getattr(candidate, "p0_rc", [0, 0]), dtype=np.float32)).astype(np.int32)
    p1 = np.rint(np.asarray(getattr(candidate, "p1_rc", [0, 0]), dtype=np.float32)).astype(np.int32)
    mid = 0.5 * (p0 + p1)
    theta_bin = int(round(float(getattr(candidate, "theta", 0.0)) / max(np.deg2rad(20.0), 1e-6)))
    return int(round(float(mid[0]))), int(round(float(mid[1]))), int(theta_bin)
