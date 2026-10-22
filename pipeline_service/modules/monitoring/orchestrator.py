import asyncio
import multiprocessing
import queue
import time

from fastapi import FastAPI
from logger_config import logger
from pipeline.state import MinerState, MinerStatus
from pipeline.task import PipelineTask
from pipeline.generation_pipeline import GenerationPipeline

from pipeline.state import ServiceState

from .module import _checker_worker
from .settings import ProbeResult

import base64
from pathlib import Path

class OrchestratorChecker:
    def __init__(self, settings) -> None:
        self.settings = settings
        self._queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=4)
        self._last: ProbeResult = ProbeResult.starting(
            getattr(settings, "vllm_model_name", "?"), "not yet probed"
        )
        self._process: multiprocessing.Process | None = None

    @property
    def result(self) -> ProbeResult:
        try:
            while True:
                self._last = self._queue.get_nowait()
        except queue.Empty:
            pass
        return self._last

    def start(self) -> None:
        self._process = multiprocessing.Process(
            target=_checker_worker,
            args=(self.settings, self._queue),
            daemon=True,
        )
        self._process.start()

    def stop(self) -> None:
        if self._process.is_alive():
            self._process.terminate() 
            self._process.join(timeout=5)
        if self._process.is_alive():
            logger.warning(f"checker {self._process.pid} didn't die on SIGTERM, killing")
            self._process.kill()  # SIGKILL
            self._process.join(timeout=2)


async def _abort_generation_and_complete(state: MinerState, generation_task: asyncio.Task | None) -> None:
    if generation_task and not generation_task.done():
        generation_task.cancel()
        try:
            await asyncio.wait_for(generation_task, timeout=10)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception:
            logger.exception("generation task raised during abort")
    state.mark_complete()
    

async def _watch_models(state: MinerState, app: FastAPI, pipeline: GenerationPipeline) -> None:
    """Watch the status of the models and update the state accordingly.

    While generating, a client counts as down only after failing every probe for
    `monitoring.down_grace_s`: replacing the pod or aborting drops the whole batch, and a single
    missed probe is routine when the vLLM API server is busy.
    """
    grace = pipeline.settings.monitoring.down_grace_s
    down_since: dict[str, float] = {}  # client -> monotonic time of its first failed probe in a row
    while True:
        try:
            checkers: dict[str, OrchestratorChecker] = getattr(app.state, "checkers", {}) or {}
            now = time.monotonic()
            for name, checker in checkers.items():
                probe_state = checker.result.state
                state.set_llm_status(name, probe_state)
                if probe_state == ServiceState.RUNNING:
                    down_since.pop(name, None)
                else:
                    down_since.setdefault(name, now)

            if state.status == MinerStatus.WARMING_UP and state.models_ready():
                probe_cfg = pipeline.settings.pipeline.coder_probe
                if probe_cfg.enabled and "probe_tps" not in state.diag:
                    tps = await pipeline.run_coder_probe()
                    if tps is not None and probe_cfg.min_tps > 0 and tps < probe_cfg.min_tps:
                        if state.replacements_remaining > 0:
                            logger.warning(
                                f"[Coder probe] {tps:.0f} tok/s < min_tps {probe_cfg.min_tps:.0f} "
                                f"(replacements_remaining={state.replacements_remaining}) — requesting REPLACE"
                            )
                            state.status = MinerStatus.REPLACE
                            await asyncio.sleep(10)
                            continue
                        logger.warning(
                            f"[Coder probe] {tps:.0f} tok/s < min_tps {probe_cfg.min_tps:.0f} but no replacements "
                            f"remain — continuing on this pod"
                        )
                logger.info("[Warmup] running warmup generation")
                warmup_image_path = Path(__file__).parent.parent.parent / "warmup.png"
                warmup_data_url = f"data:image/png;base64,{base64.b64encode(warmup_image_path.read_bytes()).decode()}"  
                warmup_task = PipelineTask(
                    stem="warmup_task",
                    image_url=warmup_data_url,
                    seed=42
                )
                if pipeline.settings.warmup:
                    await pipeline.run_warmup(warmup_task)
                    
                logger.success("[Warmup] completed successfully")
                state.status = MinerStatus.READY
                logger.info(f"models up — ready ({len(checkers)} probed)")

            elif state.status == MinerStatus.GENERATING and not state.models_ready():
                failing = {n: now - t for n, t in down_since.items()}
                down = [n for n, dt in failing.items() if dt >= grace]
                if not down:
                    logger.warning(
                        f"model(s) failing probes: "
                        f"{', '.join(f'{n} for {dt:.0f}s' for n, dt in failing.items())} — "
                        f"acting only after {grace:.0f}s"
                    )
                elif state.replacements_remaining > 1:
                    logger.warning(
                        f"model(s) down ({down}) | replacements_remaining="
                        f"{state.replacements_remaining} — requesting replace"
                    )
                    state.status = MinerStatus.REPLACE
                else:
                    logger.warning(
                        f"model(s) down ({down}) | replacements_remaining="
                        f"{state.replacements_remaining} (<=1) — keeping pod, "
                        f"completing partial results"
                    )
                    await _abort_generation_and_complete(state, app.state.generation_task)

        except Exception:
            logger.exception("_watch_models iteration failed")

        await asyncio.sleep(10)