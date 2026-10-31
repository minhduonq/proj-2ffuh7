"""Per-prompt credit and call accounting with deterministic worst-case reservations.

Credits follow the frozen rate file: input tokens once at the full input rate, output
tokens at the output rate, and the whole request at long-context rates above the
threshold. A stage reserves the worst case of every call it may make before any of
them starts, so whether a call is allowed never depends on completion order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

PRICING = json.loads(Path(__file__).with_name("pricing.json").read_text(encoding="utf-8"))
LONG_CONTEXT = int(PRICING["long_context_above_input_tokens"])


class OverBudget(RuntimeError):
    """A call would not fit its reservation."""


def credits(model: str, input_tokens: int, output_tokens: int) -> Fraction:
    rates = PRICING["models"][model]
    prefix = "long_" if input_tokens > LONG_CONTEXT else ""
    return input_tokens * Fraction(str(rates[prefix + "input_credits_per_token"])) + output_tokens * Fraction(
        str(rates[prefix + "output_credits_per_token"])
    )


@dataclass
class Reservation:
    credits: Fraction
    calls: int
    used_credits: Fraction = Fraction(0)
    used_calls: int = 0

    def take(self, worst: Fraction) -> None:
        if self.used_calls + 1 > self.calls or self.used_credits + worst > self.credits:
            raise OverBudget("Call does not fit the stage reservation")
        self.used_calls += 1
        self.used_credits += worst

    def settle(self, worst: Fraction, actual: Fraction) -> None:
        self.used_credits += actual - worst


class Ledger:
    def __init__(self, credit_cap: int, call_cap: int) -> None:
        self.credit_cap = Fraction(credit_cap)
        self.call_cap = call_cap
        self.spent = Fraction(0)
        self.calls = 0
        self.open: list[Reservation] = []

    def reserve(self, credits_needed: Fraction, calls_needed: int) -> Reservation | None:
        """Reserve a slot's worst case. Call only between stages, never while calls run."""
        held = sum((r.credits - r.used_credits for r in self.open), Fraction(0))
        held_calls = sum(r.calls - r.used_calls for r in self.open)
        if self.spent + held + credits_needed > self.credit_cap or self.calls + held_calls + calls_needed > self.call_cap:
            return None
        reservation = Reservation(credits_needed, calls_needed)
        self.open.append(reservation)
        return reservation

    def release(self, *reservations: Reservation) -> None:
        """Return the unused part of these reservations once their calls have finished."""
        self.open = [r for r in self.open if all(r is not done for done in reservations)]

    def charge(self, actual: Fraction) -> None:
        self.spent += actual
        self.calls += 1

    def __str__(self) -> str:
        return f"{self.calls} calls, {float(self.spent):,.0f} credits"
