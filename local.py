"""Run generate() locally over a prompts.txt, with your OpenAI API key (or ChatGPT
subscription through a Codex login) and a local renderer.

Nothing is submitted. Every model call is cached under the output directory, keyed by its
request body and request_id, so re-running the same prompts with unchanged code costs nothing.
Not imported by main.py; the runtime never sees this file.

    python local.py --prompts prompts.txt --seed 1313431840 --renderer-id sha256:...
    python local.py --prompts prompts.txt --seed 1313431840 --renderer-id sha256:... --codex
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import selectors
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from openai.types.responses import Response
from renderer_client import AsyncRendererClient, RenderOptions

import main
from src.budget import credits

logger = logging.getLogger("local")
CODEX_URL = "https://chatgpt.com/backend-api/codex"
FINAL_EVENTS = ("response.completed", "response.failed", "response.incomplete")


class ApiProvider:
    """Pay-per-token OpenAI API, authenticated by OPENAI_API_KEY."""

    name = "api"

    def __init__(self) -> None:
        self.client = AsyncOpenAI(timeout=1800)

    async def create(self, params: dict[str, Any]) -> Response:
        return await self.client.responses.create(**params)


class CodexProvider:
    """ChatGPT subscription through the Codex backend, as the witness does in its default mode.

    Uses the file login written by `codex -c 'cli_auth_credentials_store="file"' login`. The
    backend only streams, never stores, and needs `instructions`; the final event carries the
    complete response including usage.
    """

    name = "codex"

    def __init__(self, home: Path, executable: str = "codex") -> None:
        self.home = home
        self.executable = executable
        self.lock = asyncio.Lock()
        self.token: str | None = None
        self.client: AsyncOpenAI | None = None

    def read_login(self) -> tuple[str, str, float]:
        data = json.loads((self.home / "auth.json").read_text())
        if data.get("auth_mode") != "chatgpt":
            raise RuntimeError(f"{self.home}/auth.json is not a ChatGPT login; run codex login first")
        token = data["tokens"]["access_token"]
        payload = token.split(".")[1]
        expiry = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["exp"]
        return token, data["tokens"]["account_id"], float(expiry)

    async def session(self) -> AsyncOpenAI:
        async with self.lock:
            token, account, expiry = self.read_login()
            if expiry < time.time() + 300:
                logger.info("Refreshing the Codex login")
                await asyncio.to_thread(refresh_codex, self.home, self.executable)
                token, account, expiry = self.read_login()
                if expiry < time.time() + 300:
                    raise RuntimeError("Codex login expires too soon; sign in again with codex login")
            if token != self.token or self.client is None:
                self.client = AsyncOpenAI(
                    base_url=CODEX_URL,
                    api_key=token,
                    default_headers={"ChatGPT-Account-Id": account, "Accept": "text/event-stream"},
                    timeout=1800,
                )
                self.token = token
            return self.client

    async def create(self, params: dict[str, Any]) -> Response:
        client = await self.session()
        body = {"tools": [], **params, "store": False}
        body.setdefault("instructions", "")
        final: Response | None = None
        stream = await client.responses.create(stream=True, **body)
        async for event in stream:
            if event.type in FINAL_EVENTS:
                final = event.response
        if final is None:
            raise RuntimeError("Codex stream ended without a final response")
        return final


def refresh_codex(home: Path, executable: str) -> None:
    """Ask the Codex CLI to refresh its file login (Codex owns the refresh token)."""
    process = subprocess.Popen(  # noqa: S603
        [executable, "app-server", "--stdio", "-c", 'cli_auth_credentials_store="file"'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "CODEX_HOME": str(home)},
    )
    assert process.stdin is not None and process.stdout is not None

    def send(value: dict[str, Any]) -> None:
        process.stdin.write(json.dumps(value).encode() + b"\n")
        process.stdin.flush()

    pending, deadline = b"", time.monotonic() + 30
    try:
        send({"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "local_runner", "version": "0.1.0"}}})
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while time.monotonic() < deadline and selector.select(max(0, deadline - time.monotonic())):
                data = os.read(process.stdout.fileno(), 65536)
                if not data:
                    break
                pending += data
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    message = json.loads(line)
                    if message.get("error"):
                        raise RuntimeError("Codex login refresh failed")
                    if message.get("id") == 1:
                        send({"method": "initialized"})
                        send({"id": 2, "method": "account/read", "params": {"refreshToken": True}})
                    elif message.get("id") == 2:
                        if ((message.get("result") or {}).get("account") or {}).get("type") != "chatgpt":
                            raise RuntimeError("A ChatGPT subscription login is required")
                        return
        raise RuntimeError("Codex login refresh did not finish")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


@dataclass
class LocalWitness:
    """Stands in for the witness: model calls go to OpenAI, previews to the local renderer."""

    provider: ApiProvider | CodexProvider
    renderer: AsyncRendererClient
    cache: Path
    folder: Path
    calls: int = 0
    cached: int = 0
    spent: Fraction = Fraction(0)
    renders: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def responses(self) -> LocalWitness:
        return self

    async def create(self, *, request_id: int = 0, **params: Any) -> Response:
        body = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
        key = hashlib.sha256(f"{self.provider.name}\n{request_id}\n{body}".encode()).hexdigest()
        path = self.cache / f"{key}.json"
        hit = path.exists()
        if hit:
            try:
                result = Response.model_validate_json(path.read_text())
                self.cached += 1
            except ValueError:
                # Written by an older version of this script (field names, not API names): buy it again.
                logger.warning("Discarding unreadable cache entry %s", path.name)
                hit = False
        if not hit:
            result = await self.provider.create(params)
            # API field names (e.g. text.format.schema), so the cached response loads back unchanged.
            path.write_text(result.model_dump_json(by_alias=True, exclude_unset=True))
        self.calls += 1
        usage = result.usage
        if usage is not None:
            self.spent += credits(params["model"], usage.input_tokens, usage.output_tokens)
        self.log.append(
            {
                "request_id": request_id,
                "model": params["model"],
                "status": result.status,
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
                "cached": hit,
                "output_text": result.output_text,
            }
        )
        return result

    async def render(self, source: str, options: RenderOptions) -> bytes:
        self.renders += 1
        name = f"{self.renders:02d}"
        (self.folder / "renders" / f"{name}.js").write_text(source)
        png = await self.renderer.render(source, options)
        (self.folder / "renders" / f"{name}.png").write_bytes(png)
        return png


async def load_image(source: str, http: httpx.AsyncClient, target: Path) -> bytes:
    if target.exists():
        return target.read_bytes()
    if source.startswith(("http://", "https://")):
        response = await http.get(source)
        response.raise_for_status()
        data = response.content
    else:
        data = Path(source).expanduser().read_bytes()
    target.write_bytes(data)
    return data


def check(checker: str | None, source: Path) -> str:
    if checker is None:
        return "-"
    node = shutil.which("node")
    if node is None:
        return "no node"
    result = subprocess.run([node, checker, str(source)], capture_output=True, text=True)  # noqa: S603
    (source.parent / "checker.txt").write_text(result.stdout + result.stderr)
    return "PASSED" if result.returncode == 0 else "FAILED"


async def run_prompt(
    prompt_id: str, source: str, args: argparse.Namespace, provider: ApiProvider | CodexProvider,
    renderer: AsyncRendererClient, http: httpx.AsyncClient, cache: Path,
) -> dict[str, Any]:
    folder = args.out / prompt_id
    (folder / "renders").mkdir(parents=True, exist_ok=True)
    witness = LocalWitness(provider, renderer, cache, folder)
    started = time.monotonic()
    row: dict[str, Any] = {"prompt": prompt_id}
    try:
        image = await load_image(source, http, folder / "image.png")
        output = await main.generate(image, args.seed, witness)
        if output is None:
            row["status"] = "None"
        else:
            (folder / "final.js").write_text(output)
            (folder / "final.png").write_bytes(await renderer.render(output, RenderOptions(lighting="neutral")))
            row["status"] = "ok"
            row["checker"] = check(args.checker, folder / "final.js")
    except Exception as error:  # noqa: BLE001 - one broken prompt must not stop the batch
        logger.exception("%s failed", prompt_id)
        row["status"] = f"error: {type(error).__name__}"
    row.update(
        calls=witness.calls,
        cached=witness.cached,
        credits=float(witness.spent),
        renders=witness.renders,
        seconds=round(time.monotonic() - started, 1),
    )
    (folder / "calls.json").write_text(json.dumps(witness.log, indent=2, ensure_ascii=False))
    (folder / "result.json").write_text(json.dumps(row, indent=2))
    return row


async def amain() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True, help="one image URL or file path per line")
    parser.add_argument("--seed", type=int, required=True, help="the round seed to test with")
    parser.add_argument("--limit", type=int, help="only the first N prompts")
    parser.add_argument("--only", action="append", help="prompt ID to run (repeatable)")
    parser.add_argument("--parallel", type=int, default=4, help="prompts in flight (default 4)")
    parser.add_argument("--out", type=Path, default=Path("outputs/local"))
    parser.add_argument("--renderer-url", default=os.environ.get("RENDERER_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--renderer-id", default=os.environ.get("RENDERER_IMAGE_ID"))
    parser.add_argument("--checker", help="path to the Three.js checker's validate.js")
    parser.add_argument("--codex", action="store_true", help="use the ChatGPT subscription via a Codex login instead of OPENAI_API_KEY")
    parser.add_argument("--codex-home", type=Path, default=Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser())
    args = parser.parse_args()
    if not args.renderer_id:
        parser.error("--renderer-id or RENDERER_IMAGE_ID is required (the renderer image's sha256 ID)")

    if args.codex:
        provider: ApiProvider | CodexProvider = CodexProvider(args.codex_home)
        try:
            provider.read_login()
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            parser.error(f"no usable Codex login in {args.codex_home}: {error}")
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            parser.error("set OPENAI_API_KEY, or pass --codex to use a ChatGPT subscription")
        provider = ApiProvider()
    logger.info("Model calls via %s", "ChatGPT subscription (Codex)" if args.codex else "OpenAI API key")

    lines = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    prompts = {Path(line.split("?")[0]).stem: line for line in lines}
    if args.only:
        prompts = {k: v for k, v in prompts.items() if k in set(args.only)}
    if args.limit:
        prompts = dict(list(prompts.items())[: args.limit])
    cache = args.out / ".cache"
    cache.mkdir(parents=True, exist_ok=True)

    gate = asyncio.Semaphore(args.parallel)
    async with (
        AsyncRendererClient(args.renderer_url, renderer_id=args.renderer_id, timeout=240) as renderer,
        httpx.AsyncClient(timeout=60, follow_redirects=True) as http,
    ):

        async def guarded(prompt_id: str, source: str) -> dict[str, Any]:
            async with gate:
                return await run_prompt(prompt_id, source, args, provider, renderer, http, cache)

        rows = await asyncio.gather(*(guarded(k, v) for k, v in prompts.items()))

    print(f"\n{'prompt':<34} {'status':<22} {'calls':>5} {'cached':>6} {'credits':>11} {'renders':>7} {'sec':>7}  checker")
    for row in rows:
        print(
            f"{row['prompt'][:34]:<34} {row['status'][:22]:<22} {row['calls']:>5} {row['cached']:>6} "
            f"{row['credits']:>11,.0f} {row['renders']:>7} {row['seconds']:>7}  {row.get('checker', '')}"
        )
    over = [r["prompt"] for r in rows if r["credits"] > 5_000_000 or r["calls"] > 60]
    total = sum(r["credits"] for r in rows)
    print(f"\n{len(rows)} prompts, {total:,.0f} credits (≈ ${total / 1e6:.2f} at Standard prices)")
    if over:
        print("Over the per-prompt limit:", ", ".join(over))
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))
    return 0 if all(r["status"] == "ok" for r in rows) and not over else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sys.exit(asyncio.run(amain()))
