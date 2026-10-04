from __future__ import annotations

import base64
import json
import time
from typing import Any

from llm.session import SessionAgent
from llm.session_store import SessionStore
from logger_config import logger
from modules.base_agent import BaseAgent
from modules.critic.schema import Issue
from modules.scene_coder.prompts import (
    CODER_SYSTEM_PROMPT,
    CODER_USER_TEMPLATE_CHECKER_REPAIR,
    CODER_USER_TEMPLATE_CHECKER_REPAIR_IMAGE,
    CODER_USER_TEMPLATE_CRITIC_REPAIR,
    CODER_USER_TEMPLATE_CRITIC_REPAIR_IMAGE,
    CODER_USER_TEMPLATE_IMAGE_ONLY,
    CODER_USER_TEMPLATE_OSD,
    CODER_USER_TEMPLATE_REGEN_FIX,
    CODER_USER_TEMPLATE_REGEN_NOCODE,
)
from modules.scene_planner.schema import OSD
from config.settings import ActorConfig


class SceneCoderAgent(BaseAgent):
    """Per-pipeline JS code generator."""
    actor = "coder"

    def __init__(
        self,
        client: AsyncOpenAI,
        settings: ActorConfig,
        *,
        session_store: SessionStore,
        max_tool_iters: int = 4,
        max_output_retries: int = 2,
    ) -> None:
        super().__init__(client, settings)
        self.session_store = session_store
        self.max_tool_iters = max_tool_iters
        self.max_output_retries = max_output_retries

    def _build_session(
        self,
        task_id: str,
        actor: str,
        *,
        seed_override: int | None = None,
        temperature_override: float | None = None,
    ) -> SessionAgent:
        return SessionAgent(
            task_id=task_id,
            actor=actor,
            system_prompt=CODER_SYSTEM_PROMPT,
            model=self.model,
            tools=None,
            response_model=None,
            client=self.client,
            temperature=self.temperature if temperature_override is None else temperature_override,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed if seed_override is None else seed_override,
            max_tokens=self.max_tokens,
            max_tool_iters=self.max_tool_iters,
            enable_thinking=self.enable_thinking,
            backend=self.backend,
            providers=self.providers,
        )

    async def code(
        self,
        task_id: str,
        osd: OSD | None,
        image_bytes: bytes | None = None,
        image_mime: str = "image/jpeg",
        *,
        actor_override: str | None = None,
        seed_override: int | None = None,
        temperature_override: float | None = None,
    ) -> str:
        actor = actor_override or self.actor
        session = self.session_store.get_or_create(
            task_id, actor,
            lambda tid, act: self._build_session(
                tid, act,
                seed_override=seed_override,
                temperature_override=temperature_override,
            ),
        )
        if osd is not None:
            text = CODER_USER_TEMPLATE_OSD.format(osd_json=osd.model_dump_json(indent=2))
        else:
            if not image_bytes:
                raise RuntimeError(
                    f"Coder.code called for Task ID: {task_id} without OSD and without image — "
                    "use_planner=false requires multimodal coder input."
                )
            text = CODER_USER_TEMPLATE_IMAGE_ONLY
        if image_bytes:
            ref_b64 = base64.b64encode(image_bytes).decode()
            user_msg: str | list[dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{ref_b64}"}},
                {"type": "text", "text": text},
            ]
        else:
            user_msg = text
        logger.info(
            f"[2/7 Coder] Started Task {task_id} | Actor: {actor} | Model: {self.model} | "
            f"OSD Parts: {len(osd.parts) if osd else 'none'} | Multimodal: {bool(image_bytes)}"
        )
        t0 = time.time()
        js_code = await self._run_until_valid_js(session, user_msg)
        dt = time.time() - t0
        logger.info(
            f"[2/7 Coder] Finished Task {task_id} | Actor: {actor} | Elapsed: {dt:.1f}s | "
            f"Bytes: {len(js_code.encode('utf-8'))} | Lines: {len(js_code.splitlines())}"
        )
        return js_code

    async def code_repair(
        self,
        task_id: str,
        *,
        osd: OSD | None,
        js_errors: list[str],
    ) -> str:
        session = self.session_store.get(task_id, self.actor)
        if session is None:
            raise RuntimeError(
                f"Code repair called for Task ID: {task_id} but no Coder session exists; "
                "code() must run first."
            )
        errors_block = "\n".join(f"- {e}" for e in js_errors) or "- (no specific errors returned)"
        if osd is not None:
            user_msg = CODER_USER_TEMPLATE_CHECKER_REPAIR.format(
                osd_json=osd.model_dump_json(indent=2),
                errors_block=errors_block,
            )
        else:
            user_msg = CODER_USER_TEMPLATE_CHECKER_REPAIR_IMAGE.format(
                errors_block=errors_block,
            )
        logger.info(
            f"[Coder Repair] Started Task {task_id} | Repair: checker | Errors: {len(js_errors)}"
        )
        t0 = time.time()
        js_code = await self._run_until_valid_js(session, user_msg)
        dt = time.time() - t0
        logger.info(
            f"[Coder Repair] Finished Task {task_id} | Elapsed: {dt:.1f}s | Bytes: {len(js_code.encode('utf-8'))}"
        )
        return js_code

    async def code_critic_repair(
        self,
        task_id: str,
        *,
        osd: OSD | None,
        issues: list[Issue] | list[dict[str, Any]],
        overall_score: float,
        matching_aspects: list[str] | None = None,
        image_bytes: bytes | None = None,
        image_mime: str = "image/png",
        render_png: bytes | None = None,
    ) -> str:
        """Run the Coder in repair mode."""
        session = self.session_store.get(task_id, self.actor)
        if session is None:
            raise RuntimeError(
                f"code_critic_repair called for Task ID: {task_id} but no Coder session exists; "
                "code() must run first."
            )
        normalized_issues = []
        for issue in issues:
            if hasattr(issue, "model_dump"):
                normalized_issues.append(issue.model_dump(mode="json"))
            else:
                normalized_issues.append(issue)
        matching_block = (
            "\n".join(f"- {m}" for m in matching_aspects)
            if matching_aspects else "- (none flagged by critic — proceed carefully)"
        )
        if osd is not None:
            user_text = CODER_USER_TEMPLATE_CRITIC_REPAIR.format(
                osd_json=osd.model_dump_json(indent=2),
                overall_score=f"{overall_score:.2f}",
                issues_json=json.dumps(normalized_issues, indent=2, ensure_ascii=False),
                matching_block=matching_block,
            )
        else:
            user_text = CODER_USER_TEMPLATE_CRITIC_REPAIR_IMAGE.format(
                overall_score=f"{overall_score:.2f}",
                issues_json=json.dumps(normalized_issues, indent=2, ensure_ascii=False),
                matching_block=matching_block,
            )
        multimodal = image_bytes is not None and render_png is not None
        user_content: str | list[dict[str, Any]]
        if multimodal:
            ref_b64 = base64.b64encode(image_bytes).decode()
            render_b64 = base64.b64encode(render_png).decode()
            user_content = [
                {"type": "text",
                 "text": "Below: (1) the REFERENCE image we're trying to match, "
                         "and (2) the RENDER of your previous JS module. "
                         "Compare them yourself and decide what to fix. "
                         "Critic feedback follows but you should rely on the "
                         "visual comparison primarily.\n"},
                {"type": "text", "text": user_text},
                {"type": "image_url",
                 "image_url": {"url": f"data:{image_mime};base64,{ref_b64}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{render_b64}"}},
            ]
        else:
            user_content = user_text

        logger.info(
            f"[6/7 Patcher] Started Task {task_id} | Repair: critic | Issues: {len(normalized_issues)} | Score: {overall_score:.2f} | Multimodal: {multimodal}"
        )
        t0 = time.time()
        js_code = await self._run_until_valid_js(session, user_content)
        dt = time.time() - t0
        logger.info(
            f"[6/7 Patcher] Finished Task {task_id} | Elapsed: {dt:.1f}s | Bytes: {len(js_code.encode('utf-8'))}"
        )
        return js_code

    def _build_regen_session(
        self, task_id: str, actor: str, *, seed: int, temperature: float,
        enable_thinking: bool, max_tokens: int,
    ) -> SessionAgent:
        """Phien MOI, khong lich su: ban ve lai khong duoc thay ma cu (khac code_critic_repair)."""
        return SessionAgent(
            task_id=task_id,
            actor=actor,
            system_prompt=CODER_SYSTEM_PROMPT,
            model=self.model,
            tools=None,
            response_model=None,
            client=self.client,
            temperature=temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            seed=seed,
            max_tokens=max_tokens,
            max_tool_iters=self.max_tool_iters,
            enable_thinking=enable_thinking,
            backend=self.backend,
            providers=self.providers,
        )

    async def code_regen(
        self,
        task_id: str,
        *,
        image_bytes: bytes,
        image_mime: str,
        render_png: bytes,
        feedback: str,
        seed: int,
        temperature: float,
        enable_thinking: bool = True,
        max_tokens: int = 20000,
        actor: str = "coder#regen",
    ) -> str:
        """Refine: VE LAI tu dau, khong dua ma cu. Dau vao: anh de + luoi render ban cu + nhan xet.

        Giong nhanh C1 cua lab: system prompt goc, user = [anh de, luoi render, REGEN_NOCODE(feedback)],
        bat thinking. Session KHONG luu vao session_store (moi ban ve lai doc lap).
        """
        session = self._build_regen_session(
            task_id, actor, seed=seed, temperature=temperature,
            enable_thinking=enable_thinking, max_tokens=max_tokens,
        )
        ref_b64 = base64.b64encode(image_bytes).decode()
        render_b64 = base64.b64encode(render_png).decode()
        user_msg: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{ref_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{render_b64}"}},
            {"type": "text", "text": CODER_USER_TEMPLATE_REGEN_NOCODE.format(feedback=feedback)},
        ]
        logger.info(
            f"[Refine Coder] Started Task {task_id} | Actor: {actor} | Seed: {seed} | "
            f"Temp: {temperature} | Thinking: {enable_thinking}"
        )
        t0 = time.time()
        js_code = await self._run_until_valid_js(session, user_msg, normalize=self._normalize_regen_output)
        logger.info(
            f"[Refine Coder] Finished Task {task_id} | Actor: {actor} | Elapsed: {time.time() - t0:.1f}s | "
            f"Bytes: {len(js_code.encode('utf-8'))}"
        )
        return js_code

    async def code_regen_fix(
        self,
        task_id: str,
        *,
        js_code: str,
        error: str,
        seed: int,
        max_tokens: int = 20000,
        actor: str = "coder#regen-fix",
    ) -> str:
        """Refine: ban ve lai khong chay duoc -> dua loi that + ma hong, xin ban sua (thinking TAT, nhu lab)."""
        session = self._build_regen_session(
            task_id, actor, seed=seed, temperature=0.3, enable_thinking=False, max_tokens=max_tokens,
        )
        text = CODER_USER_TEMPLATE_REGEN_FIX.format(
            err=error[:400], fmt="Return ONLY the full corrected JavaScript module, no fences.",
        ) + "\n\nFAILED MODULE:\n" + js_code
        return await self._run_until_valid_js(session, text, normalize=self._normalize_regen_output)

    async def _run_until_valid_js(
        self,
        session: SessionAgent,
        user_msg: str | list[dict[str, Any]],
        normalize=None,
    ) -> str:
        normalize = normalize or self._normalize_js_output
        raw = await session.run(user_msg)
        js = normalize(str(raw))
        for attempt in range(self.max_output_retries + 1):
            if self._looks_like_js_module(js):
                return js
            if attempt >= self.max_output_retries:
                break
            raw = await session.run(
                "Your previous response was not a valid raw JavaScript module. "
                "Return ONLY the full JS source with the exact signature "
                "`export default function generate(THREE)` and no markdown fences."
            )
            js = normalize(str(raw))
        raise ValueError(
            "Coder did not return a valid JS module with "
            "`export default function generate(THREE)`."
        )

    @staticmethod
    def _normalize_js_output(raw: str) -> str:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    @classmethod
    def _normalize_regen_output(cls, raw: str) -> str:
        """Ban ve lai (bat thinking): bo khoi <think>...</think> truoc ma, roi cat loi dan truoc chu ky module."""
        text = raw.strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        text = cls._normalize_js_output(text)
        i = text.find("export default function generate")
        if i > 0:
            text = text[i:]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3].rstrip()
        return text

    @staticmethod
    def _looks_like_js_module(text: str) -> bool:
        return (
            "export default function generate(THREE)" in text
            and "return" in text
        )
