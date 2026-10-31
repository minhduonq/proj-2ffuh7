"""The only door to the outside world: model calls and preview renders for one prompt.

Error policy, matching what replay can reproduce:
- infrastructure failures (`APIConnectionError`, `APITimeoutError`, `RenderUnavailable`)
  propagate, so the runner leaves the prompt pending and retries it later;
- provider rejections (`APIStatusError`, non-completed responses) are recorded outcomes
  and become `CallFailed`, which only discards the candidate that made the call.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, cast

from openai import APIConnectionError, APIStatusError
from openai.types.responses import Response, ResponseInputParam
from renderer_client import RenderOptions

from .budget import Ledger, Reservation, credits
from .config import Config, Role

if TYPE_CHECKING:
    from witness_client import AsyncWitnessClient

logger = logging.getLogger(__name__)


class CallFailed(Exception):
    """A model call returned no usable result; the caller drops that candidate."""

    def __init__(self, message: str, *, truncated: bool = False) -> None:
        super().__init__(message)
        self.truncated = truncated


def png_data_url(image: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


def text(value: str) -> dict[str, str]:
    return {"type": "input_text", "text": value}


@dataclass
class Session:
    witness: AsyncWitnessClient
    config: Config
    ledger: Ledger
    tag: str = ""

    def image(self, png: bytes, role: Role) -> dict[str, str]:
        return {"type": "input_image", "image_url": png_data_url(png), "detail": role.detail}

    def worst_case(self, role: Role, content: list[dict[str, str]], instructions: str) -> Fraction:
        """Upper bound on a call's credits: generous input estimate plus the full output allowance."""
        chars = len(instructions) + sum(len(item.get("text", "")) for item in content)
        images = sum(item["type"] == "input_image" for item in content)
        return credits(role.model, chars // 3 + 64 + images * self.config.image_tokens, role.max_output_tokens)

    def worst_for(self, role: Role, *, text_chars: int, images: int) -> Fraction:
        """Worst case for a call that has not been built yet, used when planning a stage."""
        return credits(role.model, text_chars // 3 + 64 + images * self.config.image_tokens, role.max_output_tokens)

    async def call(
        self,
        role: Role,
        request_id: int,
        instructions: str,
        content: list[dict[str, str]],
        reservation: Reservation,
        *,
        schema: dict[str, object] | None = None,
    ) -> str:
        worst = self.worst_case(role, content, instructions)
        reservation.take(worst)
        params: dict[str, Any] = {
            "model": role.model,
            "store": False,
            "service_tier": "default",
            "reasoning": {"effort": role.effort},
            "max_output_tokens": role.max_output_tokens,
            "instructions": instructions,
            "input": cast(ResponseInputParam, [{"role": "user", "content": content}]),
        }
        if schema is not None:
            params["text"] = {"format": {"type": "json_schema", "name": "reply", "strict": True, "schema": schema}}
        try:
            result: Response = await self.witness.responses.create(request_id=request_id, **params)
        except APIConnectionError:
            raise
        except APIStatusError as error:
            self._settle(reservation, worst, worst)
            raise CallFailed(f"call {request_id}: provider status {error.status_code}") from error
        usage = result.usage
        actual = worst if usage is None else credits(role.model, usage.input_tokens, usage.output_tokens)
        self._settle(reservation, worst, actual)
        logger.info(
            "%s call %d %s | %s in / %s out | %s",
            self.tag,
            request_id,
            role.model,
            usage.input_tokens if usage else "?",
            usage.output_tokens if usage else "?",
            self.ledger,
        )
        if result.status != "completed":
            reason = getattr(result.incomplete_details, "reason", None)
            raise CallFailed(f"call {request_id}: status {result.status} ({reason})", truncated=reason == "max_output_tokens")
        return result.output_text

    def _settle(self, reservation: Reservation, worst: Fraction, actual: Fraction) -> None:
        reservation.settle(worst, actual)
        self.ledger.charge(actual)

    async def render(self, source: str, options: RenderOptions) -> bytes:
        """Preview through the witness; replay reproduces and verifies every render."""
        return await self.witness.render(source, options)
