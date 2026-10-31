"""Chains of draft → refinements, each refinement gated by a mirrored Stage-1 duel, then a bracket of the chains.

From the round-0 leader (reports/round0-competitor-analysis.md): every chain refines its own module on renders of the
judge's views, the generator asks for a detailed module, the asset is fitted at 0.99 and a flat object is tilted
towards the front cameras in the returned module. From 3d-gen-api: four chains with different construction
strategies, a mirrored duel before a refinement replaces its parent, and a bracket of duels to pick the chain.

Replay determinism rules followed here:
- request IDs come from (stage, slot, attempt), never from completion order;
- budget is reserved per slot at stage boundaries, so admission never depends on timing;
- shuffles are seeded from the round seed, the image hash and the stage.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from renderer_client import RenderRejected

from . import judge, prompts
from .budget import Ledger, OverBudget, Reservation
from .config import DEFAULT, Config, Role
from .js import extract, normalize, placeholder, precheck
from .session import CallFailed, Session, text

if TYPE_CHECKING:
    from witness_client import AsyncWitnessClient

logger = logging.getLogger(__name__)
JUDGE_FAILURES = (CallFailed, OverBudget, ValueError, KeyError, TypeError)
GATE_STAGE = 50  # the gate duels of refinement round r use stage GATE_STAGE + r
BRACKET_STAGE = 90  # bracket round k uses stage BRACKET_STAGE + k
MAX_ERROR_CHARS = 4000


@dataclass(frozen=True)
class Candidate:
    raw: str  # what the model wrote; shown back to the model when refining
    source: str  # normalized module that was rendered
    front: bytes  # Config.front_preview: the judge's near-front views on white
    sides: bytes  # Config.side_preview: +X, back, -X and top on gray


def request_id(stage: int, slot: int, attempt: int) -> int:
    return 1 + stage * 100 + slot * 10 + attempt


async def run(
    image: bytes,
    seed: int,
    witness: AsyncWitnessClient,
    config: Config = DEFAULT,
) -> str | None:
    digest = hashlib.sha256(image).hexdigest()
    session = Session(witness, config, Ledger(config.credit_cap, config.call_cap), digest[:12])
    pipeline = Pipeline(session, image, seed, digest)

    duels = len(config.drafts) - 1
    bracket = pipeline.ledger.reserve(pipeline.duel_worst() * duels, 2 * duels)
    chains = await pipeline.stage(0, [(slot, role, None) for slot, role in enumerate(config.drafts)])
    for number, role in enumerate(config.refine_rounds, start=1):
        chains = await pipeline.refine(number, role, chains)
    if bracket is not None:
        pipeline.ledger.release(bracket)

    finalists = [chain for chain in chains if chain is not None]
    if not finalists:
        logger.warning("%s | no valid draft", session.tag)
        return fallback(image) if config.placeholder else None
    winner = await pipeline.bracket(finalists)
    logger.info("%s | done: %s", session.tag, session.ledger)
    return normalize(winner.raw, tilt=config.tilt_flat)


class Pipeline:
    def __init__(self, session: Session, image: bytes, seed: int, digest: str) -> None:
        self.session = session
        self.config = session.config
        self.ledger = session.ledger
        self.image = image
        self.seed = seed
        self.digest = digest

    # Budget planning -------------------------------------------------------------------

    def slot_calls(self) -> int:
        return 1 + self.config.repair_attempts + int(self.config.retry_truncated)

    def slot_worst(self, role: Role, parent: Candidate | None) -> Fraction:
        """Worst case of the first call plus one repair. Further follow-ups only run on what the
        slot's own earlier calls left unspent, so admission never depends on other slots' timing."""
        chars = len(prompts.GENERATOR) + len(prompts.DRAFT) + len(role.strategy) + len(prompts.COMPACT)
        images = 1
        if parent is not None:
            chars += len(prompts.REFINE) + len(prompts.FRONT_SHEET) + len(prompts.SIDE_SHEET) + len(parent.raw)
            images += 2
        first = self.session.worst_for(role, text_chars=chars, images=images)
        if self.slot_calls() == 1:
            return first
        repair_chars = len(prompts.GENERATOR) + len(prompts.DRAFT) + len(prompts.REPAIR) + MAX_ERROR_CHARS
        repair_chars += role.max_output_tokens * 4 + 8192
        return first + self.session.worst_for(self.config.repairer, text_chars=repair_chars, images=1)

    def duel_content(self, first: bytes, second: bytes) -> list[dict[str, str]]:
        session, role = self.session, self.config.judge
        return [
            text("Reference image (target object):"),
            session.image(self.image, role),
            text("3D model 1:"),
            session.image(first, role),
            text("3D model 2:"),
            session.image(second, role),
            text(prompts.JUDGE_USER),
        ]

    def duel_worst(self) -> Fraction:
        """Both seat orders of one duel."""
        one = self.session.worst_case(self.config.judge, self.duel_content(b"", b""), prompts.JUDGE_SYSTEM)
        return 2 * one

    # Stages ----------------------------------------------------------------------------

    async def stage(self, number: int, jobs: list[tuple[int, Role, Candidate | None]]) -> list[Candidate | None]:
        """Run one call per (slot, role, parent) concurrently; the result lines up with `jobs`, None where a slot
        was not admitted by the budget or produced no valid module."""
        reservations: list[Reservation] = []
        admitted: list[bool] = []
        calls = []
        for slot, role, parent in jobs:
            reservation = self.ledger.reserve(self.slot_worst(role, parent), self.slot_calls())
            admitted.append(reservation is not None)
            if reservation is None:
                logger.info("%s | stage %d: budget leaves no call for slot %d", self.session.tag, number, slot)
                continue
            reservations.append(reservation)
            calls.append(self.candidate(number, slot, role, reservation, parent))
        results = iter(await asyncio.gather(*calls))
        self.ledger.release(*reservations)
        return [next(results) if ok else None for ok in admitted]

    async def refine(self, number: int, role: Role, chains: list[Candidate | None]) -> list[Candidate | None]:
        """Every live chain refines its module once; a refinement replaces its parent unless the parent wins."""
        live = [slot for slot, chain in enumerate(chains) if chain is not None]
        if not live:
            return chains
        gate = self.ledger.reserve(self.duel_worst() * len(live), 2 * len(live))
        if gate is None:
            logger.info("%s | round %d: budget leaves no gate", self.session.tag, number)
            return chains
        parents = {slot: chain for slot, chain in enumerate(chains) if chain is not None}
        children = await self.stage(number, [(slot, role, parents[slot]) for slot in live])
        contested = [(slot, child) for slot, child in zip(live, children, strict=True) if child is not None]
        verdicts = await self.duels(GATE_STAGE + number, [(parents[slot], child) for slot, child in contested], gate)
        self.ledger.release(gate)
        result = list(chains)
        for (slot, child), verdict in zip(contested, verdicts, strict=True):
            keep = verdict.winner == 1 or (verdict.winner is None and self.config.keep_refinement_on_draw)
            logger.info(
                "%s | round %d chain %d: %s %s (%s)",
                self.session.tag,
                number,
                slot,
                "refinement kept" if keep else "parent kept",
                verdict.penalties,
                verdict.reason[:160],
            )
            if keep:
                result[slot] = child
        return result

    async def candidate(
        self,
        stage: int,
        slot: int,
        role: Role,
        reservation: Reservation,
        parent: Candidate | None,
    ) -> Candidate | None:
        session, config = self.session, self.config
        if parent is None:
            request = prompts.DRAFT + ("\n" + role.strategy if role.strategy else "")
            content = [text(request), session.image(self.image, role)]
        else:
            content = [
                text("TARGET:"),
                session.image(self.image, role),
                text(prompts.REFINE.format(source=parent.raw)),
                text(prompts.FRONT_SHEET),
                session.image(parent.front, role),
                text(prompts.SIDE_SHEET),
                session.image(parent.sides, role),
            ]
        current, attempt, repairs, compacted = role, 0, 0, False
        while True:
            rid = request_id(stage, slot, attempt)
            try:
                reply = await session.call(current, rid, prompts.GENERATOR, content, reservation)
            except CallFailed as error:
                if error.truncated and config.retry_truncated and not compacted:
                    compacted, attempt, current = True, attempt + 1, config.repairer
                    content = [*content, text(prompts.COMPACT)]
                    continue
                logger.warning("%s | %s", session.tag, error)
                return None
            except OverBudget:
                logger.info("%s | call %d skipped: over its reservation", session.tag, rid)
                return None
            raw = extract(reply)
            source = normalize(raw)
            errors = precheck(source)
            if not errors:
                try:
                    front = await session.render(source, config.front_preview)
                    sides = await session.render(source, config.side_preview)
                    return Candidate(raw, source, front, sides)
                except RenderRejected as rejection:
                    errors = [rejection.detail_json]
            logger.info("%s | call %d rejected: %s", session.tag, rid, "; ".join(errors)[:300])
            if repairs >= config.repair_attempts:
                return None
            repairs, attempt, current = repairs + 1, attempt + 1, config.repairer
            content = [
                text(prompts.DRAFT),
                session.image(self.image, current),
                text(prompts.REPAIR.format(errors="\n".join(errors)[:MAX_ERROR_CHARS], source=raw)),
            ]

    # Judging ---------------------------------------------------------------------------

    async def duels(
        self, stage: int, pairs: list[tuple[Candidate, Candidate]], reservation: Reservation
    ) -> list[judge.Verdict]:
        """Mirrored Stage-1 duels on the front sheets, all sent together; a failed call makes its duel a draw."""
        calls = []
        for index, (a, b) in enumerate(pairs):
            for order, (first, second) in enumerate(((a, b), (b, a))):
                calls.append(
                    self.session.call(
                        self.config.judge,
                        request_id(stage, index, order),
                        prompts.JUDGE_SYSTEM,
                        self.duel_content(first.front, second.front),
                        reservation,
                        schema=prompts.JUDGE_SCHEMA,
                    )
                )
        scores: list[judge.Score | None] = []
        for reply in await asyncio.gather(*calls, return_exceptions=True):
            if isinstance(reply, JUDGE_FAILURES):
                logger.warning("%s | judge failed: %s", self.session.tag, reply)
                scores.append(None)
            elif isinstance(reply, BaseException):
                raise reply
            else:
                scores.append(judge.score(reply))
        return [judge.verdict(scores[2 * i], scores[2 * i + 1]) for i in range(len(pairs))]

    async def bracket(self, finalists: list[Candidate]) -> Candidate:
        """Single elimination in a seeded order; a draw goes to the lower mean penalty, then to the earlier entry."""
        alive = list(finalists)
        random.Random(f"{self.seed}:{self.digest}:bracket").shuffle(alive)  # nosec B311
        number = 0
        while len(alive) > 1:
            pairs = [(alive[i], alive[i + 1]) for i in range(0, len(alive) - 1, 2)]
            reservation = self.ledger.reserve(self.duel_worst() * len(pairs), 2 * len(pairs))
            if reservation is None:
                logger.info("%s | bracket: budget leaves no duel", self.session.tag)
                return alive[0]
            verdicts = await self.duels(BRACKET_STAGE + number, pairs, reservation)
            self.ledger.release(reservation)
            winners = []
            for (a, b), verdict in zip(pairs, verdicts, strict=True):
                pick = verdict.winner
                if pick is None:
                    pick = 1 if verdict.penalties[1] < verdict.penalties[0] else 0
                logger.info("%s | bracket %d: %s %s", self.session.tag, number, pick, verdict.penalties)
                winners.append(b if pick else a)
            alive = winners + alive[2 * len(pairs) :]
            number += 1
        return alive[0]


def fallback(image: bytes) -> str:
    """A valid placeholder coloured like the target's foreground, or neutral grey without Pillow."""
    try:
        from PIL import Image

        pixels = list(Image.open(io.BytesIO(image)).convert("RGB").resize((32, 32), Image.Resampling.NEAREST).getdata())
    except Exception:  # noqa: BLE001 - any decode problem just means a grey box
        return placeholder("b0b0b0")
    corner = pixels[0]
    foreground = [p for p in pixels if sum(abs(a - b) for a, b in zip(p, corner, strict=True)) > 48] or pixels
    channels = [sorted(p[i] for p in foreground)[len(foreground) // 2] for i in range(3)]
    return placeholder("".join(f"{min(c, 0xE6):02x}" for c in channels))
