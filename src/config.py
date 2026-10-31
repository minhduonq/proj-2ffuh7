"""Every tunable knob of the pipeline, in one place.

Change the defaults here (or build another `Config` and pass it to `pipeline.run`).
Everything that affects requests must stay fixed between generation and replay, so
keep it as committed source, never environment variables or wall-clock values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from renderer_client import RenderOptions

ASTRA = "gpt-6-astra"
SOL = "gpt-6.1-sol"
LUNA = "gpt-6-luna"

Effort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
Detail = Literal["low", "high", "auto"]


@dataclass(frozen=True)
class Role:
    """One kind of model call: which model, how hard it thinks and how much it may write."""

    model: str
    effort: Effort = "medium"
    max_output_tokens: int = 32768
    detail: Detail = "high"
    # Construction strategy appended to a draft's request; empty for refinements and repairs.
    strategy: str = ""


STRATEGIES = (
    "",
    "Strategy: model each part directly at its final position and size.",
    "Strategy: first block out the whole silhouette with coarse shapes and check the proportions against the "
    "target, then refine each block into its detailed shape without changing the proportions.",
    "Strategy: build sub-assemblies as named Groups positioned relative to their parent part; write small helper "
    "functions for repeated or mirrored parts and reuse geometries and materials.",
)


@dataclass(frozen=True)
class Config:
    # Stage 0: one independent chain per entry (at most 8), each with its own construction strategy.
    drafts: tuple[Role, ...] = tuple(Role(SOL, "high", strategy=strategy) for strategy in STRATEGIES)
    # Refinement rounds; in each round every surviving chain refines its own current module once.
    refine_rounds: tuple[Role, ...] = (Role(SOL, "high"), Role(SOL, "high"))
    # Called with the source and checker errors when a candidate is rejected, and for the
    # compact retry after a truncated reply.
    repairer: Role = Role(SOL)
    repair_attempts: int = 1
    # Retry once with a "be compact" hint when a reply hits max_output_tokens.
    retry_truncated: bool = True
    # Stage-1 penalties for duels, both seat orders: the refinement gates and the final bracket.
    judge: Role = Role(SOL, "low", max_output_tokens=1024)
    # A refinement replaces its parent unless the parent wins the duel outright; a draw keeps the refinement.
    keep_refinement_on_draw: bool = True
    # Tilt objects lying flat 15 degrees towards the front cameras in the returned module only.
    tilt_flat: bool = True
    # The four near-front views the official judge scores in Stage 1 (judge-service S1_ANGLES), on white.
    # This render validates every candidate, is what the judge here compares, and is shown to refinements.
    front_preview: RenderOptions = field(
        default_factory=lambda: RenderOptions(
            thetas=(30.0, 330.0, 0.0, 0.0), phis=(0.0, 0.0, 15.0, -30.0), bg_color="ffffff", lighting="neutral"
        )
    )
    # +X, back, -X and top on gray, shown to refinements so they also see what the side guard checks.
    side_preview: RenderOptions = field(
        default_factory=lambda: RenderOptions(
            thetas=(90.0, 180.0, 270.0, 0.0), phis=(0.0, 0.0, 0.0, -90.0), bg_color="808080", lighting="neutral"
        )
    )
    # Self-imposed limits, kept under the official 5,000,000 credits / 60 calls per prompt. Reservations hold the
    # worst case of every call before it starts, so spending can reach these caps but never pass them.
    credit_cap: int = 4_950_000
    call_cap: int = 58
    # Conservative token estimate per input image, used to reserve worst-case credits.
    image_tokens: int = 2000
    # Return a simple coloured box instead of None when no candidate survives.
    placeholder: bool = True

    def __post_init__(self) -> None:
        if not 0 < len(self.drafts) <= 8:
            raise ValueError("Between 1 and 8 drafts")
        if not 0 <= self.repair_attempts <= 8:
            raise ValueError("repair_attempts must be 0 to 8")
        if len(self.refine_rounds) > 8:
            raise ValueError("At most 8 refinement rounds")


DEFAULT = Config()
