# NOTE
- The `local.py` file contains the `httpx` library and is prohibited for submission. This file is intended solely for development purposes and must not be included when submitting your worker.
- Files not required when submitting a worker: `local.py`, `prompts.txt`, `README.md`, `TESTING.md`. `instructions.md` **is** required: `src/prompts.py` reads it at import.

# 3d-gen-api

Turns one PNG image into a deterministic Three.js module using OpenAI models through
a witness server. The repository root `main.py` exports the entry point the runtime calls:

```python
async def generate(image: bytes, seed: int, witness) -> str | None
```

Model calls go through `witness.responses.create(...)` and previews through
`witness.render(...)`. Return the final JavaScript, or `None` when no candidate ever
renders. Submitting the result is the runner's job, not this repository's.

## Layout

| File | What to change there |
| --- | --- |
| `src/config.py` | All knobs: models, effort, token limits, coders, repair rounds, image detail |
| `src/prompts.py` | Every text and JSON schema sent to the models |
| `instructions.md` | The Three.js output specification appended to the coder prompt |
| `src/pipeline.py` | Stage logic: spec → candidates → repairs → self-view → gate → bracket, with the request id of each call site |
| `src/session.py` | Model calls, previews, error policy, waves that settle the budget in a fixed order |
| `src/budget.py`, `src/pricing.json` | Credits per the frozen rate file, worst-case holds |
| `src/js.py`, `src/normalize.js` | Reply extraction and the wrapper that turns and fits the module |
| `src/views.py` | Camera views, the reference-camera estimate and the orientation yaw |
| `src/judge.py` | Aggregation of a mirrored S1 duel |
| `local.py` | Local run over a `prompts.txt` with your API key and a local renderer; never submits. See [TESTING.md](TESTING.md) |

## Default pipeline

1. **Spec** (`gpt-6-astra`): a JSON build spec with parts, proportions, colors,
   the reference camera and a checklist.
2. **Candidates**: four modules in parallel, three `gpt-6.1-sol` prompt variants
   (direct, blockout, hierarchy) and one `gpt-6-astra`.
3. **Repairs**: up to two rounds by Sol for candidates the renderer rejected or
   that returned no module.
4. **Self-view**: each valid candidate sees its own renders (front, sides, the
   estimated reference view) and writes a revision. A revision that the renderer
   rejects gets one repair.
5. **Gate**: a mirrored S1-style duel decides whether the revision replaces its
   parent.
6. **Bracket**: single elimination over the survivors with the same duel.

A typical run uses about 26 model calls and three previews per version.

**Orientation.** The judge scores near-front views. When the spec says the photo
was taken from the side or the back, the wrapper turns the submitted model by a
multiple of 90 degrees so that the photographed side faces +Z.

**Failures.** Failed candidates are discarded. A draw in the bracket goes to the
lower mean penalty. If no candidate ever renders, `generate` returns `None`.

## Rules the code keeps (do not break them when extending)

- **Replay determinism.** The same code, image and seed must produce the same requests, in the same identity, and the same output.
  - Every call site has a fixed `request_id` (constants at the top of `src/pipeline.py`).
  - Calls run in waves: a wave starts only after the previous one finished, and decisions read only this prompt's completed calls in a fixed order.
  - The bracket order is shuffled with `random.Random(f"{seed}:{sha256(image)}")`.
- **Budget.** 5,000,000 credits and 60 calls per prompt.
  - The spec, the four candidates and a reserve for the bracket fit even if every call uses its full output limit.
  - A repair or self-view runs only if the credits actually spent, plus its own worst case, still fit.
- **Errors.**
  - `APIConnectionError` and any `WitnessError` whose source is not the provider propagate, so the runner retries the prompt later.
  - Provider errors only discard the candidate that made the call.
- **Requests.** `service_tier="default"` (Flex is forbidden) and `store=False`.
- **Dependencies.** Only the standard library, `openai`, `pydantic`, `numpy`, `Pillow` and the supplied clients are available.
  - Nothing may be installed at runtime.
  - Commit every file that affects generation.
