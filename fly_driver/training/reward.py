"""Per-term reward components. Progress is the only large positive term."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class RewardTerms:
    progress: float = 0.0
    on_track: float = 0.0
    alive: float = 0.0
    collision: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {f"reward/{k}": float(v) for k, v in asdict(self).items()}


def shaped_reward(terms: RewardTerms) -> float:
    """Sum of logged terms. Alive bonus is intentionally zero so standing still cannot farm."""
    return float(terms.progress + terms.on_track + terms.alive + terms.collision)


def farms_without_progress(terms: RewardTerms, *, max_idle: float = 0.05) -> bool:
    """True if the agent could harvest reward while making no track progress."""
    if terms.progress > 0.0:
        return False
    idle = terms.on_track + terms.alive + max(terms.collision, 0.0)
    return idle > max_idle
