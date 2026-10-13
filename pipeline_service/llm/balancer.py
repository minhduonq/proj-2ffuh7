from __future__ import annotations

from openai import AsyncOpenAI


import contextvars

# Set by callers (JudgeAgent.compare) to pin every request of one task to one replica, so the task's shared
# prefix (system prompt + reference image) stays hot in a single engine's prefix cache. None = round-robin.
affinity_key: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_affinity_key", default=None)


class LoadBalancedClient:
    """Weighted round-robin over AsyncOpenAI clients serving the same model,
    with optional per-task affinity (see `affinity_key`).
    """

    def __init__(self, clients: list[AsyncOpenAI], weights: list[int]) -> None:
        if len(clients) != len(weights):
            raise ValueError("clients and weights must have the same length")
        schedule: list[AsyncOpenAI] = []
        # Interleave by weight (e.g. weights [2,1] -> [a, b, a]) so bursts
        # don't land on a single replica.
        counters = [0] * len(clients)
        total = sum(max(1, w) for w in weights)
        for step in range(total):
            best = max(
                range(len(clients)),
                key=lambda i: max(1, weights[i]) / (counters[i] + 1),
            )
            counters[best] += 1
            schedule.append(clients[best])
        self._schedule = schedule
        self._i = 0
        self._clients = list(clients)
        self._weights = [max(1, w) for w in weights]
        self._assign: dict[str, int] = {}
        self._keys = [0] * len(clients)

    def pick(self):
        key = affinity_key.get()
        if key is not None:
            # First-seen keys go to the replica with the fewest keys per unit weight, so a 32-stem batch splits
            # 16/16 (a hash split gave 22/10 on a real batch). A key keeps its replica for the process lifetime.
            idx = self._assign.get(key)
            if idx is None:
                idx = min(range(len(self._clients)), key=lambda i: (self._keys[i] / max(1, self._weights[i]), i))
                self._assign[key] = idx
                self._keys[idx] += 1
            return self._clients[idx]
        client = self._schedule[self._i % len(self._schedule)]
        self._i += 1
        return client

    @property
    def chat(self):
        return self.pick().chat
