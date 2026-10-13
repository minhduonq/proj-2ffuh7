"""Prompt-scoped random streams (refine critics)."""

from __future__ import annotations

import hashlib


def prompt_seed(round_seed: int, label: str) -> int:
    """Reproducible 31-bit seed from the revealed round seed and a label that names the prompt and the use.

    Independent of request ordering and batch boundaries, so an audit regeneration samples the same stream.
    """
    payload = f"{round_seed}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF
