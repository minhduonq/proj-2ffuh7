from __future__ import annotations

import asyncio
import gc
import time
import os
from pathlib import Path
from typing import NamedTuple

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from config.settings import LLMClientConfig, SettingsConf
from logger_config import logger
from modules.js_checker.module import JSCheckerModule
from modules.renderer.module import RendererModule
from pipeline.candidate_export import write_dump
from pipeline.factory import build_pipeline
from pipeline.orchestrator import Pipeline
from pipeline.state import MinerState
from pipeline.task import PipelineTask

_PLACEHOLDER_KEYS = {"", "placeholder", "your-key-here", "sk-or-...", "changeme"}

# Per batch: the orchestrator's /status poll, the /results download and the next POST /generate.
_RESULTS_RESERVE_S = 120.0
# Tasks ship their best candidate this long before the batch's hard stop.
_SHIP_RESERVE_S = 30.0
_MIN_BATCH_BUDGET_S = 60.0
# Failure stages that come from running out of time: a retry would fail the same way.
_TIME_FAILURES = {"deadline", "batch_deadline"}


class BatchSlot(NamedTuple):
    batch_no: int        # 1-based position of the batch within its round
    round_left_s: float  # time left in the round's window when the batch started
    share_s: float       # round_left_s split evenly over this and the round's later batches


class RoundClock:
    """Splits a round's generation window across its batches.

    The orchestrator's clock runs from the first POST /generate of a round to the last /results
    download, so time a batch leaves unused carries over to the later ones. A new round starts on the
    first batch, after `batches` batches, once the window has passed, or when the seed changes (each
    round publishes one seed). A replacement pod cannot know how much of the window its predecessor
    used and starts a fresh one.
    """

    def __init__(self, window_s: float, batches: int) -> None:
        self.window_s = window_s
        self.batches = max(1, batches)
        self.started_at: float | None = None
        self.seed: int | None = None
        self.batch_no = 0

    def start_batch(self, seed: int, now: float) -> BatchSlot:
        if (
            self.started_at is None
            or seed != self.seed
            or self.batch_no >= self.batches
            or now - self.started_at >= self.window_s
        ):
            self.started_at, self.seed, self.batch_no = now, seed, 0
        self.batch_no += 1
        left = self.window_s - (now - self.started_at)
        return BatchSlot(self.batch_no, left, left / (self.batches - self.batch_no + 1))


class JudgeBatchStats(BaseModel):
    """Judge summary over one batch, from the per-duel records in task.meta["judge_duels"]."""
    duels: int = 0
    s1_decided: int = 0
    s2_decided: int = 0
    s3_decided: int = 0
    s4_stepdown: int = 0
    draws: int = 0
    s1_early_stopped: int = 0
    s1_requests_saved: int = 0
    compare_mean_s: float = 0.0
    compare_p90_s: float = 0.0
    queue_mean_s: float = 0.0

    def log_line(self) -> str:
        n = max(self.duels, 1)
        return (
            f"[Batch judge] duels={self.duels} s1={100 * self.s1_decided / n:.0f}% s2={100 * self.s2_decided / n:.0f}% "
            f"s3={100 * self.s3_decided / n:.0f}% s4_stepdown={self.s4_stepdown} draws={self.draws} "
            f"s1_early={self.s1_early_stopped} (saved {self.s1_requests_saved} S1 requests) "
            f"compare_mean={self.compare_mean_s:.1f}s p90={self.compare_p90_s:.1f}s queue_mean={self.queue_mean_s:.1f}s"
        )


def _judge_batch_stats(tasks) -> JudgeBatchStats:
    duels = [d for t in tasks for d in (t.meta or {}).get("judge_duels", [])]
    stats = JudgeBatchStats(duels=len(duels))
    for d in duels:
        by = str(d.get("decided_by", ""))
        s1_slim = (d.get("detail") or {}).get("s1_slim") or {}
        if s1_slim.get("early_stopped"):
            stats.s1_early_stopped += 1
            stats.s1_requests_saved += 2 * (4 - int(s1_slim.get("n_total", 4)))
        if "tie-break" in by or "draw ->" in by:
            stats.draws += 1
        elif by.startswith("S4"):
            stats.s4_stepdown += 1
        elif by.startswith("S3"):
            stats.s3_decided += 1
        elif by.startswith("S2"):
            stats.s2_decided += 1
        elif by.startswith("S1"):
            stats.s1_decided += 1
    if duels:
        compare = sorted(float(d.get("compare_s", 0.0)) for d in duels)
        stats.compare_mean_s = sum(compare) / len(compare)
        stats.compare_p90_s = compare[min(len(compare) - 1, int(0.9 * len(compare)))]
        stats.queue_mean_s = sum(float(d.get("queue_s", 0.0)) for d in duels) / len(duels)
    return stats
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")


def _is_local_endpoint(base_url: str) -> bool:
    return any(h in base_url for h in _LOCAL_HOSTS)


def _resolve_api_key(name: str, cfg: LLMClientConfig) -> str | None:
    """Resolve api key for one client"""
    raw = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else ""
    api_key = raw.strip().strip('"').strip("'")
    if not api_key:
        api_key = cfg.api_key.strip()

    if _is_local_endpoint(cfg.base_url):
        return api_key or "local"

    if api_key.lower() in _PLACEHOLDER_KEYS:
        env_hint = f" (env var {cfg.api_key_env})" if cfg.api_key_env else ""
        logger.warning(
            f"Skipping LLM client {name!r}{env_hint}: placeholder/empty API key "
        )
        return None
    return api_key


class GenerationPipeline:
    """Top-level pipeline driver. Constructed once per app lifecycle."""

    def __init__(self, settings: SettingsConf, state: MinerState) -> None:
        self.settings = settings
        self.state = state

        # DET modules
        self.js_checker = JSCheckerModule(settings.js_checker)
        self.renderer = RendererModule(settings.renderer)

        self._clients: dict[str, AsyncOpenAI] = {}
        self._http_client: httpx.AsyncClient | None = None
        self._pipeline: Pipeline | None = None
        self._round = RoundClock(settings.pipeline.round_time_budget, settings.pipeline.round_batches)

    # Lifecycle

    async def startup(self) -> None:
        logger.info("GenerationPipeline starting")

        for name, cfg in self.settings.llm_clients.items():
            if not cfg.enabled:
                continue
            api_key = _resolve_api_key(name, cfg)
            if api_key is None:
                continue
            self._clients[name] = AsyncOpenAI(
                base_url=cfg.base_url,
                api_key=api_key,
                timeout=httpx.Timeout(3600.0, connect=120.0),
                max_retries=0,
            )
            logger.info(
                f"llm client ready | name={name} base_url={cfg.base_url} "
                f"local={_is_local_endpoint(cfg.base_url)}"
            )
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
        self._http_client = httpx.AsyncClient(timeout=30.0, limits=limits)

        await self.js_checker.startup()
        await self.renderer.startup()
        self._pipeline = build_pipeline(
            settings=self.settings,
            clients=self._clients,
            js_checker=self.js_checker,
            renderer=self.renderer,
            http_client=self._http_client,
        )
        await self._pipeline.start()
        actors = self.settings.actors
        eb = self.settings.event_bus
        use_planner = self.settings.pipeline.use_planner
        logger.info(f"[Pipeline initialized — waiting for models]")
        if use_planner:
            logger.info(f"Planner: {actors.planner.client} | Model: {actors.planner.model}")
        else:
            logger.info("Planner: DISABLED (pipeline.use_planner=false — pure-image coder)")
        logger.info(
            f"Coder: {actors.coder.client} | Model: {actors.coder.model} | "
            f"multimodal={actors.coder.multimodal} | ensemble_size={actors.coder.ensemble_size} | "
            f"ensemble_temperature={actors.coder.ensemble_temperature}"
        )
        logger.info(f"Critic: {actors.critic.client} | Model: {actors.critic.model}")
        if actors.coder.ensemble_size > 1:
            logger.info(
                f"Judge: {actors.judge.client} | Model: {actors.judge.model} | "
                f"max_stage={actors.judge.max_stage.value} | s1_concurrency={actors.judge.s1_concurrency} | "
                f"bracket={self.settings.pipeline.bracket.model_dump()}"
            )
        else:
            logger.info("Judge: DISABLED (ensemble_size=1)")
        logger.info(f"Iter cap: {eb.max_iter} | Deadline: {eb.task_deadline_s:.0f}s | Threshold: {eb.score_threshold:.2f}")

    async def shutdown(self) -> None:
        logger.info("[Pipeline shutting down]")
        if self._pipeline is not None:
            await self._pipeline.stop()
        await self.renderer.shutdown()
        await self.js_checker.shutdown()
        if self._http_client is not None:
            await self._http_client.aclose()
        for name, client in list(self._clients.items()):
            try:
                await client.close()
            except Exception as exc:
                logger.warning(f"[Pipeline shutdown] Client {name} close failed: {exc}")

    def _cleanup_batch_memory(self, context: str) -> None:
        if self._pipeline is None:
            return
        sessions = self._pipeline.session_store.clear()
        statuses = len(self._pipeline.task_status)
        self._pipeline.task_status.clear()
        collected = gc.collect()
        logger.info(
            f"[{context} cleanup] sessions={sessions} "
            f"task_status={statuses} gc_collected={collected}"
        )

    async def run_batch(self, tasks: list[PipelineTask]) -> None:
        if self._pipeline is None:
            logger.error("[Batch failed] Run called before startup")
            return
        if tasks:
            batch_seed = tasks[0].seed
            if self._pipeline.planner is not None:
                self._pipeline.planner.seed = batch_seed
            self._pipeline.coder.seed = batch_seed
            self._pipeline.critic.seed = batch_seed
            if self._pipeline.judge is not None:
                self._pipeline.judge.seed = batch_seed
            planner_seed = self._pipeline.planner.seed if self._pipeline.planner is not None else "n/a"
            judge_seed = self._pipeline.judge.seed if self._pipeline.judge is not None else "n/a"
            logger.info(
                f"[Batch seed] Agents updated | seed={batch_seed} "
                f"(planner={planner_seed}, coder={self._pipeline.coder.seed}, "
                f"critic={self._pipeline.critic.seed}, judge={judge_seed})"
            )
            
        cfg = self.settings.pipeline
        dump_root = cfg.candidate_dump_dir
        loop = asyncio.get_running_loop()
        started = self.state.batch_started_at or time.time()
        slot = self._round.start_batch(tasks[0].seed if tasks else self.state.seed, started)
        budget = max(min(cfg.batch_time_budget, slot.share_s) - _RESULTS_RESERVE_S, _MIN_BATCH_BUDGET_S)
        # Deadlines on the loop clock, counted from the POST /generate that opened the batch.
        batch_end = loop.time() - (time.time() - started) + budget
        task_deadline = batch_end - _SHIP_RESERVE_S

        async def run_one(task: PipelineTask) -> None:
            while True:
                result = await (await self._pipeline.submit(task, deadline=task_deadline))
                retry = self._should_retry(result, batch_end - loop.time())
                if not retry:
                    # Record before dumping, so a slow dump near the hard stop cannot lose the result.
                    self.state.record_task(result)
                if dump_root and result.candidates:
                    await self._dump_candidates(result, Path(dump_root))
                if not retry:
                    return
                task = PipelineTask(
                    stem=task.stem, image_url=task.image_url, seed=task.seed,
                )
                task.attempt = result.attempt + 1
                logger.info(f"[Batch retry] {task.stem} | attempt={task.attempt}")

        try:
            logger.info(
                f"[Batch starting] {len(tasks)} tasks | round batch {slot.batch_no}/{cfg.round_batches} | "
                f"round left={slot.round_left_s:.0f}s | budget={budget:.0f}s | "
                f"tasks ship by {task_deadline - loop.time():.0f}s"
            )
            await asyncio.wait_for(
                asyncio.gather(*(run_one(t) for t in tasks)),
                timeout=max(0.0, batch_end - loop.time()),
            )

        except asyncio.TimeoutError:
            logger.warning(f"[Batch deadline] hit {budget:.0f}s")
            for task in tasks:
                recorded = self.state.tasks.get(task.stem)
                if recorded is None or recorded.failed or not recorded.js_code:
                    task.failed = True
                    task.failure_reason = f"batch budget exceeded ({budget:.0f}s)"
                    task.failure_stage = "batch_deadline"
                    self.state.record_task(task)
            
        except asyncio.CancelledError:
            raise
        finally:
            self._cleanup_batch_memory("Batch")
            logger.info(
                f"[Batch done] {len(self.state.results)} ok, "
                f"{len(self.state.failed)} failed"
            )
            logger.info(_judge_batch_stats(self.state.tasks.values()).log_line())

    def _should_retry(self, task: PipelineTask, remaining_s: float) -> bool:
        """Retry a failed task with a fresh one, unless retries are used up, it failed by running out
        of time (a retry would fail the same way and take coder slots from the other tasks), or the
        batch is about to end."""
        if not task.failed:
            return False
        cfg = self.settings.pipeline
        if task.attempt >= cfg.task_retries:
            reason = f"{cfg.task_retries} retr{'y' if cfg.task_retries == 1 else 'ies'} used"
        elif task.failure_stage in _TIME_FAILURES:
            reason = f"failed on time ({task.failure_stage})"
        elif remaining_s < cfg.retry_min_remaining_s:
            reason = f"only {remaining_s:.0f}s left in the batch"
        else:
            return True
        logger.info(f"[Batch retry] {task.stem} | not retried: {reason} | stage={task.failure_stage}")
        return False

    async def _dump_candidates(self, task: PipelineTask, root: Path) -> None:
        """Persist every candidate of one task attempt (code, renders, judge inputs) for offline experiments."""
        try:
            out, count = await asyncio.to_thread(write_dump, task, root, self.state.batch_index)
            logger.info(
                f"[Candidate dump] {task.stem} attempt={task.attempt} -> {out} "
                f"({len(task.candidates)} candidates, {count} files)"
            )
        except Exception as exc:
            logger.warning(f"[Candidate dump] {task.stem} failed: {exc!r}")

    async def run_coder_probe(self) -> float | None:
        """Measure the coder endpoint's aggregate output tok/s with `concurrency` text-only code requests.

        Returns tok/s (None when disabled / no client / every request failed). Records
        state.diag['probe_tps'] and friends. The batch wall-clock is coder tokens / tok/s, so this
        number predicts whether the batch fits its time budget; the caller decides REPLACE vs continue.
        """
        cfg = self.settings.pipeline.coder_probe
        if not cfg.enabled:
            return None
        coder = self.settings.actors.coder
        client = self._clients.get(coder.client or "")
        if client is None or not coder.model:
            logger.warning("[Coder probe] no coder client — skipped")
            return None
        prompt = (
            "Write a complete Three.js ES module `export default function generate(THREE)` that builds a detailed "
            "wooden park bench with slats, armrests and cast-iron legs, centered and fitted into a unit cube. "
            "Use only geometry and materials, no textures. Output only the code."
        )
        think = coder.enable_thinking if coder.enable_thinking is not None else bool(coder.reasoning_effort)

        async def one(i: int) -> int:
            resp = await client.chat.completions.create(
                model=coder.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=cfg.max_tokens,
                temperature=coder.ensemble_temperature,
                seed=10_000 + i,
                extra_body={"chat_template_kwargs": {"enable_thinking": think}},
            )
            usage = getattr(resp, "usage", None)
            return int(getattr(usage, "completion_tokens", 0) or 0)

        t_w = time.monotonic()
        try:
            warm = await asyncio.wait_for(
                asyncio.gather(*(one(i) for i in range(cfg.concurrency)), return_exceptions=True),
                timeout=cfg.timeout_s,
            )
            w_toks = sum(r for r in warm if isinstance(r, int)); w_wall = max(time.monotonic() - t_w, 1e-3)
            logger.info(f"[Coder probe] warm-up burst: {w_toks} tokens in {w_wall:.1f}s = {w_toks / w_wall:.0f} tok/s (not scored)")
            self.state.diag["probe_tps_cold"] = round(w_toks / w_wall)
        except Exception as exc:
            logger.warning(f"[Coder probe] warm-up burst failed: {exc!r} — timing the probe anyway")
        t0 = time.monotonic()
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(one(i) for i in range(cfg.concurrency)), return_exceptions=True),
                timeout=cfg.timeout_s,
            )
        except asyncio.TimeoutError:
            logger.error(f"[Coder probe] timed out after {cfg.timeout_s:.0f}s")
            self.state.diag["probe_tps"] = 0.0
            return 0.0
        wall = max(time.monotonic() - t0, 1e-3)
        toks = sum(r for r in results if isinstance(r, int))
        errs = [r for r in results if not isinstance(r, int)]
        tps = toks / wall
        self.state.diag.update({
            "probe_tps": round(tps), "probe_tokens": toks, "probe_wall_s": round(wall, 1),
            "probe_conc": cfg.concurrency, "probe_errors": len(errs),
        })
        logger.info(
            f"[Coder probe] {cfg.concurrency} x {cfg.max_tokens} tok: {toks} output tokens in {wall:.1f}s "
            f"= {tps:.0f} tok/s | errors={len(errs)} | min_tps={cfg.min_tps:.0f}"
            + (f" | first error: {errs[0]!r}" if errs else "")
        )
        if errs and not toks:
            return None
        return tps

    async def run_warmup(self, task: PipelineTask) -> None:
        if self._pipeline is None:
            logger.error("[Warmup failed] Run called before startup")
            return
        try:
            logger.info(f"[Warmup task starting]")
            future = await self._pipeline.submit(task)
            result = await future
            if getattr(result, "failed", False):
                raise RuntimeError(
                    f"warmup produced no module: {getattr(result, 'failure_reason', 'unknown')}"
                )
            logger.info(f"[Warmup task completed]")
        except asyncio.CancelledError:
            logger.info("[Warmup task cancelled]")
            raise
        except Exception as exc:
            logger.exception(f"[Warmup failed] {exc}")
            raise
        finally:
            self._cleanup_batch_memory("Warmup")
