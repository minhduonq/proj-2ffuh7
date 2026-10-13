"""Refine: CRITIC -> FRESH REGENERATION (the coder never sees the old code) of the bracket winner.

Our own addition (lab branch C1, carried by workers v2/v3). Evidence:
  - independent check, 40 prompts, second opponent, both seat orders: C1 -0.11 S1 penalty [-0.24; +0.00];
  - full 128-prompt run of round 49: margin vs leader -0.047 -> -0.008 (16 prompts changed: 9 better, 3 worse).

Per prompt (N = refine.candidates, default 4):
  1. N visual critiques (critic GLM, one seed each) + N BUILD-TECHNIQUE critiques (GLM).
  2. N programs written from scratch: reference + render grid of the winner + the critiques; NO old code.
  3. Each takes exactly the bracket-candidate path: checker (+ rule fixes) -> orientation -> render + judge views
     + DINO. A checker/render failure gets one coder fix with the real error.
  4. Each live regeneration duels the winner (full judge, all 4 S1 angles). It is accepted only if it WINS and
     beats the winner by >= accept_margin S1 penalty points; with several, the largest margin. None -> winner kept.
Every error inside this stage is swallowed; the bracket winner already sits on the task and is only replaced in
the final lines.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from config.settings import OrientationConfig
from logger_config import logger
from pipeline.seeding import prompt_seed
from pipeline.stages import _CandidateFactory, _code_digest, _is_live
from pipeline.task import Candidate, PipelineTask

CRITIC_TEMPERATURE = 0.6
REGEN_SEED_OFFSET = 500_000      # keeps regeneration seeds apart from the K bracket candidates' stream
REGEN_K_BASE = 1000              # regeneration candidate numbers (k = 1000 + ci) never collide with bracket ks


def s1_margin(verdict: Any) -> float | None:
    """Mean S1 penalty (over the judged angles, each already averaged over both seat orders) of B minus A.

    A = regeneration, B = bracket winner. > 0 means the regeneration is BETTER (lower penalty).
    None when the judge returned no S1 detail.
    """
    detail = getattr(verdict, "detail", None) or {}
    angles = (detail.get("s1_slim") or {}).get("angles") or []
    if not angles:
        return None
    pen_a = sum(float(a["pen_a"]) for a in angles) / len(angles)
    pen_b = sum(float(a["pen_b"]) for a in angles) / len(angles)
    return pen_b - pen_a


def build_feedback(report: Any, struct: dict | None) -> str | None:
    """Critique text handed to the coder (lab format). None when both critiques failed."""
    if report is None and struct is None:
        return None
    parts: list[str] = []
    if report is not None:
        issues = [i.model_dump(mode="json") if hasattr(i, "model_dump") else i for i in (report.issues or [])]
        parts.append(
            "ISSUES from the visual critic (score %.2f):\n%s\n\nKEEP (already matches): %s"
            % (report.overall_score, json.dumps(issues, indent=2, ensure_ascii=False), list(report.matching_aspects or []))
        )
    struct_issues = (struct or {}).get("structural_issues") or []
    parts.append(
        "BUILD-TECHNIQUE errors (a construction reviewer):\n"
        + (json.dumps(struct_issues, indent=2, ensure_ascii=False) if struct_issues else "(none found)")
    )
    return "\n\n".join(parts)


def _promote(task: PipelineTask, cand: Candidate) -> None:
    task.winner_k = cand.k
    task.js_code = cand.js_code
    task.js_valid = cand.js_valid
    task.js_errors = list(cand.js_errors)
    task.scene_json = cand.scene_json
    task.rendered_png = cand.rendered_png


async def refine_regen_stage(
    task: PipelineTask,
    *,
    coder,
    critic,
    judge,
    embedder,
    js_checker,
    renderer,
    sem_coder: asyncio.Semaphore,
    sem_checker: asyncio.Semaphore,
    sem_renderer: asyncio.Semaphore,
    sem_judge: asyncio.Semaphore,
    sem_critic: asyncio.Semaphore,
    status: dict[str, str],
    n_cand: int = 4,
    accept_margin: float = 0.25,
    thinking: bool = True,
    max_tokens: int = 20000,
    seed_offset: int = 0,
    render_from_object: bool = False,
    orientation: OrientationConfig | None = None,
    code_fix: bool = False,
) -> dict:
    """Run refine for one prompt. Returns (and stores in task.meta['refine']) a summary."""
    t0 = time.time()
    info: dict[str, Any] = {"mode": "regen", "n_cand": n_cand, "accepted": None, "cands": []}
    task.meta["refine"] = info

    winner = next((c for c in (task.candidates or []) if c.k == task.winner_k), None)
    if (
        judge is None or critic is None or winner is None or not _is_live(winner)
        or not task.image_bytes or not winner.judge_white_views
    ):
        info["skipped"] = "no judge/critic or no live bracket winner with judge views"
        return info

    # 1) critiques: N visual + N build-technique, all in parallel (GLM)
    async def visual(ci: int):
        async with sem_critic:
            try:
                return await critic.critique(
                    task_id=task.stem,
                    image_bytes=task.image_bytes,
                    image_mime=task.image_mime,
                    render_png=winner.rendered_png,
                    artifact_context={"kind": "coder_v1", "js_code": winner.js_code, "osd": None},
                    seed_override=prompt_seed(task.seed, f"{task.stem}:refine:visual:{ci}"),
                    temperature_override=CRITIC_TEMPERATURE,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[REFINE] {task.stem} critic c{ci} failed: {type(exc).__name__}: {exc}")
                return None

    async def structural(ci: int):
        async with sem_critic:
            try:
                return await critic.struct_critique(
                    task_id=task.stem,
                    image_bytes=task.image_bytes,
                    image_mime=task.image_mime,
                    render_png=winner.rendered_png,
                    js_code=winner.js_code or "",
                    seed=prompt_seed(task.seed, f"{task.stem}:refine:structural:{ci}"),
                    temperature=0.3 if ci == 0 else 0.7,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[REFINE] {task.stem} struct critic c{ci} failed: {type(exc).__name__}: {exc}")
                return None

    status[task.stem] = "refine:critic"
    reports, structs = await asyncio.gather(
        asyncio.gather(*[visual(ci) for ci in range(n_cand)]),
        asyncio.gather(*[structural(ci) for ci in range(n_cand)]),
    )

    # 2-3) regenerate + check + orient + render (+ one fix on failure): the bracket-candidate path
    factory = _CandidateFactory(
        task,
        coder=coder,
        judge=judge,
        embedder=embedder,
        js_checker=js_checker,
        renderer=renderer,
        sem_coder=sem_coder,
        sem_checker=sem_checker,
        sem_renderer=sem_renderer,
        status=status,
        osd=None,
        send_image=True,
        checker_mode="with_object" if render_from_object else "sanity",
        ensemble_temperature=0.0,
        seed_offset=seed_offset,
        orientation=orientation,
        code_fix=code_fix,
    )
    await factory.embed_reference()
    old_digest = _code_digest(winner.js_code)
    feedbacks: dict[int, str] = {}

    async def make(ci: int) -> Candidate | None:
        feedback = build_feedback(reports[ci], structs[ci])
        if feedback is None:
            return None
        feedbacks[ci] = feedback
        cand = Candidate(k=REGEN_K_BASE + ci, seed=task.seed + seed_offset + REGEN_SEED_OFFSET + 100 * ci)
        t1 = time.time()
        try:
            async with sem_coder:
                cand.js_code = await coder.code_regen(
                    task.stem,
                    image_bytes=task.image_bytes,
                    image_mime=task.image_mime,
                    render_png=winner.rendered_png,
                    feedback=feedback,
                    seed=cand.seed,
                    temperature=0.7 if ci == 0 else 0.9,
                    enable_thinking=thinking,
                    max_tokens=max_tokens,
                    actor=f"coder#regen{ci}",
                )
        except Exception as exc:  # noqa: BLE001
            cand.drop_reason = f"coder:{type(exc).__name__}"
            cand.elapsed_s = time.time() - t1
            return cand
        if _code_digest(cand.js_code) == old_digest:
            cand.drop_reason = "identical_to_winner"
            return cand
        raw = cand.js_code                       # _validate may inject a rotation -> the fix must see the RAW code
        await factory._validate(cand)
        if cand.drop_reason is not None and cand.drop_reason.split(":")[0] in ("checker", "renderer"):
            error = "; ".join((cand.js_errors or []) + (cand.render_errors or [])) or cand.drop_reason
            try:
                async with sem_coder:
                    fixed = await coder.code_regen_fix(
                        task.stem, js_code=raw, error=error,
                        seed=cand.seed + 7, max_tokens=max_tokens, actor=f"coder#regenfix{ci}",
                    )
                cand.js_code, cand.drop_reason = fixed, None
                cand.js_valid, cand.js_errors, cand.render_errors = None, [], []
                cand.rendered_png, cand.scene_json = None, None
                cand.judge_white_views, cand.judge_gray_views, cand.judge_embeddings = {}, {}, None
                await factory._validate(cand)
            except Exception as exc:  # noqa: BLE001
                cand.drop_reason = f"fix:{type(exc).__name__}"
        cand.elapsed_s = time.time() - t1
        return cand

    status[task.stem] = "refine:coder"
    cands = await asyncio.gather(*[make(ci) for ci in range(n_cand)])
    made = [c for c in cands if c is not None]
    live = [c for c in made if _is_live(c) and c.judge_white_views]

    # 4) duel the bracket winner: regeneration = A (left), winner = B (right); full S1 (no early stop)
    async def duel(cand: Candidate):
        ci = cand.k - REGEN_K_BASE
        try:
            async with sem_judge:
                verdict = await judge.compare(
                    task_id=task.stem,
                    match_label=f"REFINE c{ci}-vs-k{winner.k}",
                    reference_bytes=task.image_bytes,
                    reference_mime=task.image_mime,
                    render_a=cand.rendered_png,
                    render_b=winner.rendered_png,
                    white_views_a=cand.judge_white_views,
                    white_views_b=winner.judge_white_views,
                    gray_views_a=cand.judge_gray_views,
                    gray_views_b=winner.judge_gray_views,
                    embeddings_a=cand.judge_embeddings,
                    embeddings_b=winner.judge_embeddings,
                    early_stop=False,
                )
            return cand, verdict, s1_margin(verdict)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[REFINE] {task.stem} duel c{ci} failed: {type(exc).__name__}: {exc}")
            return cand, None, None

    results: list = []
    if live:
        status[task.stem] = "refine:judge"
        results = await asyncio.gather(*[duel(c) for c in live])

    passing = []
    for cand, verdict, margin in results:
        ok = verdict is not None and verdict.winner == "A" and margin is not None and margin >= accept_margin
        info["cands"].append({
            "c": cand.k - REGEN_K_BASE, "winner": getattr(verdict, "winner", None),
            "decided_by": getattr(verdict, "decided_by", None),
            "s1_margin": None if margin is None else round(margin, 3), "passed": ok,
            "elapsed_s": round(cand.elapsed_s, 1), "feedback": feedbacks.get(cand.k - REGEN_K_BASE),
        })
        if ok:
            passing.append((margin, cand))
    for c in made:
        if not (_is_live(c) and c.judge_white_views):
            info["cands"].append({"c": c.k - REGEN_K_BASE, "dropped": c.drop_reason or "no judge views"})

    task.candidates = list(task.candidates) + made
    old_k = winner.k
    if passing:
        margin, best = max(passing, key=lambda x: x[0])
        _promote(task, best)
        info["accepted"] = best.k - REGEN_K_BASE
        info["accepted_margin"] = round(margin, 3)
    info["elapsed_s"] = round(time.time() - t0, 1)
    logger.info(
        f"[REFINE] {task.stem} | winner=k{old_k} | live={len(live)}/{n_cand} | passed={len(passing)} | "
        f"accepted={info['accepted']} | margins={[c.get('s1_margin') for c in info['cands'] if 's1_margin' in c]} | "
        f"elapsed={info['elapsed_s']}s"
    )
    return info
