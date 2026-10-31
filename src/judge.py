"""Aggregation of a mirrored duel, with the thresholds of stage S1 of the official judge (docs 06 §6.3)."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

DRAW_THRESHOLD = 0.7
BOTH_BAD_AVERAGE = 8.0
BOTH_BAD_MAX_DIFF = 1.5


class Score(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    penalty_1: int
    penalty_2: int
    issues: str


@dataclass(frozen=True)
class Verdict:
    winner: int | None  # 0 for the first model, 1 for the second, None for a draw
    penalties: tuple[float, float]
    reason: str


def score(reply: str | None) -> Score | None:
    if reply is None:
        return None
    try:
        parsed = Score.model_validate_json(reply)
    except ValidationError:
        return None
    return Score(
        penalty_1=max(0, min(10, parsed.penalty_1)),
        penalty_2=max(0, min(10, parsed.penalty_2)),
        issues=parsed.issues,
    )


def verdict(forward: Score | None, backward: Score | None) -> Verdict:
    """`forward` shows (a, b), `backward` shows (b, a); a missing or contradictory pair is a draw."""
    if forward is None or backward is None:
        return Verdict(None, (5.0, 5.0), "unreadable judgment")
    a = (forward.penalty_1 + backward.penalty_2) / 2
    b = (forward.penalty_2 + backward.penalty_1) / 2
    first, second = forward.penalty_1 - forward.penalty_2, backward.penalty_2 - backward.penalty_1
    if first * second < 0:
        return Verdict(None, (a, b), "the two orders disagree")
    if (a + b) / 2 >= BOTH_BAD_AVERAGE and abs(a - b) <= BOTH_BAD_MAX_DIFF:
        return Verdict(None, (a, b), "both are far from the reference")
    if abs(a - b) <= DRAW_THRESHOLD:
        return Verdict(None, (a, b), "too close to call")
    return Verdict(0 if a < b else 1, (a, b), forward.issues)
