"""Generate Three.js assets from images using a witness server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src import pipeline

__all__ = ["generate"]

if TYPE_CHECKING:
    from witness_client import AsyncWitnessClient


async def generate(image: bytes, seed: int, witness: AsyncWitnessClient) -> str | None:
    # The runtime awaits async entry points with exactly (image, seed, witness): model calls go
    # through witness.responses.create(...) and previews through witness.render(...).
    return await pipeline.run(image, seed, witness)
