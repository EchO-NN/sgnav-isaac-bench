from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RobotState:
    x: float
    y: float
    z: float
    yaw: float
    collided: bool = False

