"""Kiem thu loc truoc DINOv3 + refine (critic -> ve lai) + chot thoi gian, bang LLM/renderer gia lap.

Chay duoc khong can pytest:
    trong image:  docker run --rm -v "$PWD/tests:/tests:ro" -w /workspace <image> python -m unittest discover -s /tests -p 'test_v3_*.py'
    tai repo:     python -m unittest discover -s tests -p 'test_v3_*.py'   (can cac goi trong docker/requirements.lock.txt)
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SVC = os.path.join(_REPO, "pipeline_service")
if not os.path.isdir(_SVC):
    _SVC = "/workspace"                                   # tests mount vao image
_CFG = os.path.join(_REPO, "configuration.yaml")
if not os.path.isfile(_CFG):
    _CFG = "/workspace/configuration.yaml"
os.environ["CONFIG_FILE"] = _CFG
sys.path.insert(0, _SVC)
os.chdir(tempfile.mkdtemp(prefix="w3test_"))              # logger_config ghi logs/pipeline.log theo CWD

from config.settings import BracketConfig, OrientationConfig, RefineConfig, SettingsConf  # noqa: E402
from modules.judge.schema import JudgeVerdict  # noqa: E402
from modules.scene_coder.agent import SceneCoderAgent  # noqa: E402
from pipeline import orchestrator as orch_mod  # noqa: E402
from pipeline.orchestrator import Pipeline  # noqa: E402
from pipeline.refine import REGEN_K_BASE, refine_regen_stage, s1_margin  # noqa: E402
from pipeline.stages import _completed, _prefilter_bracket  # noqa: E402
from pipeline.task import Candidate, PipelineTask  # noqa: E402

FRONT = ("front_left", "front_right", "front_below", "front_above")
ALL_VIEWS = FRONT + ("right", "back", "left", "top_down")


class _Views:
    def __init__(self, tag: str) -> None:
        self.tag = tag


class FakeJudge:
    """Quyet theo bang chat luong an (cao = tot); diem phat S1 = 9 - chat luong o ca 4 goc."""

    def __init__(self, quality: dict[str, float]) -> None:
        self.quality = quality
        self.calls: list[tuple[str, str]] = []

    async def encode_reference(self, image_bytes, mime):
        return "data:ref"

    async def encode_views(self, *, grid, white, gray, embeddings):
        return _Views(grid.decode())

    def early_stop_for_round(self, round_no):
        return False

    async def similarity(self, views):
        return None

    async def compare(self, *, task_id, match_label, prompt_url, left_views, right_views,
                      s1_temperature=0.0, seed=None, max_stage=None, early_stop=None):
        qa, qb = self.quality[left_views.tag], self.quality[right_views.tag]
        self.calls.append((left_views.tag, right_views.tag))
        angles = [{"label": n, "pen_a": 9 - qa, "pen_b": 9 - qb, "ab": {}, "ba": {}, "contradictory": False}
                  for n in FRONT]
        return JudgeVerdict(winner="A" if qa >= qb else "B", decided_by="S1",
                            detail={"s1_slim": {"angles": angles}})


class FakeEmbedder:
    def __init__(self, scores: dict[int, float]) -> None:
        self.scores = scores
        self.asked: list[int] = []

    async def prefilter_scores(self, reference, views_by_k, top=2):
        self.asked = sorted(views_by_k)
        return {k: v for k, v in self.scores.items() if k in views_by_k}


def _cand(k: int, live: bool = True) -> Candidate:
    c = Candidate(k=k, seed=k)
    c.js_code = f"export default function generate(THREE) {{ return TAG:k{k} }}"
    if live:
        c.rendered_png = f"k{k}".encode()
        c.judge_white_views = {n: b"w" for n in ALL_VIEWS}
    return c


def _task() -> PipelineTask:
    t = PipelineTask(stem="stem0", image_url="https://example.invalid/ref.png")
    t.image_bytes = b"ref"
    return t


class PrefilterTests(unittest.IsolatedAsyncioTestCase):
    async def test_keeps_top_by_dino_and_judges_only_them(self):
        quality = {f"k{k}": q for k, q in enumerate([1, 9, 3, 4, 5, 6, 7, 8])}   # k1 tot nhat nhung DINO thap
        scores = {0: 0.9, 1: 0.1, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5, 6: 0.4, 7: 0.3}
        cands = [_cand(k) for k in range(8)]
        task, judge = _task(), FakeJudge(quality)
        winner = await _prefilter_bracket(
            task=task, leaf_tasks={c.k: _completed(c) for c in cands}, embedder=FakeEmbedder(scores),
            judge=judge, sem_judge=asyncio.Semaphore(4), bracket=BracketConfig(), keep=4,
        )
        self.assertEqual(task.meta["prefilter"]["kept"], [0, 2, 3, 4])
        self.assertTrue(task.meta["prefilter"]["scored"])
        self.assertEqual(winner.k, 4)                                   # tot nhat trong 4 ban giu
        judged = {t for pair in judge.calls for t in pair}
        self.assertEqual(judged, {"k0", "k2", "k3", "k4"})                # ban bi loai khong ton luot trong tai

    async def test_no_dino_scores_means_full_bracket(self):
        quality = {f"k{k}": q for k, q in enumerate([1, 9, 3, 4, 5, 6, 7, 8])}
        cands = [_cand(k) for k in range(8)]
        task = _task()
        winner = await _prefilter_bracket(
            task=task, leaf_tasks={c.k: _completed(c) for c in cands}, embedder=None,
            judge=FakeJudge(quality), sem_judge=asyncio.Semaphore(4), bracket=BracketConfig(), keep=4,
        )
        self.assertFalse(task.meta["prefilter"]["scored"])
        self.assertEqual(len(task.meta["prefilter"]["kept"]), 8)
        self.assertEqual(winner.k, 1)

    async def test_dead_candidates_are_never_kept(self):
        quality = {f"k{k}": q for k, q in enumerate([1, 2, 3, 4, 5, 6])}
        cands = [_cand(k, live=k not in (2, 5)) for k in range(6)]
        emb = FakeEmbedder({k: 1.0 - k / 10 for k in range(6)})
        task = _task()
        winner = await _prefilter_bracket(
            task=task, leaf_tasks={c.k: _completed(c) for c in cands}, embedder=emb,
            judge=FakeJudge(quality), sem_judge=asyncio.Semaphore(4), bracket=BracketConfig(), keep=3,
        )
        self.assertEqual(emb.asked, [0, 1, 3, 4])
        self.assertEqual(task.meta["prefilter"]["kept"], [0, 1, 3])
        self.assertEqual(winner.k, 3)


class _Report:
    def __init__(self) -> None:
        self.issues = []
        self.overall_score = 0.4
        self.matching_aspects = ["overall shape"]


class FakeCritic:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.seeds: list[int] = []

    async def critique(self, *, seed_override=None, temperature_override=None, **kw):
        if self.fail:
            raise RuntimeError("critic down")
        self.seeds.append(seed_override)
        return _Report()

    async def struct_critique(self, **kw):
        if self.fail:
            return None
        return {"structural_issues": [{"part": "leg", "should_be": "thin_member"}], "verdict": "legs too thick"}


class FakeCoder:
    def __init__(self, codes: dict[int, str], fail: bool = False, winner_grid: bytes = b"w") -> None:
        self.codes, self.fail, self.winner_grid = codes, fail, winner_grid
        self.thinking: list[bool] = []
        self.fixes = 0

    async def code_regen(self, task_id, *, image_bytes, image_mime, render_png, feedback, seed,
                         temperature, enable_thinking=True, max_tokens=20000, actor="coder#regen"):
        if self.fail:
            raise RuntimeError("coder down")
        assert "BUILD-TECHNIQUE" in feedback and render_png == self.winner_grid   # ve lai nhin luoi cua ban thang
        self.thinking.append(enable_thinking)
        return self.codes[int(actor.replace("coder#regen", ""))]

    async def code_regen_fix(self, task_id, *, js_code, error, seed, max_tokens=20000, actor=""):
        self.fixes += 1
        assert "PARSE_ERROR" in error
        return js_code.replace("BROKEN", "")


class FakeChecker:
    async def process(self, task, mode="sanity"):
        ok = "BROKEN" not in (task.js_code or "")
        task.js_valid, task.js_errors = ok, ([] if ok else ["PARSE_ERROR: unexpected token"])
        task.scene_json = {"object": 1} if ok and mode == "with_object" else None
        return task


class FakeRenderer:
    class config:                                       # noqa: N801 - mo phong RendererConfig
        judge_multiview = True

    async def process_with_judge_views(self, task, slot=None):
        tag = task.js_code.split("TAG:")[1].split()[0]
        task.rendered_png = tag.encode()
        return {n: b"w" for n in ALL_VIEWS}, {"front_left": b"g"}

    async def process(self, task, slot=None):
        task.rendered_png = task.js_code.split("TAG:")[1].split()[0].encode()
        return task


def _refine_task() -> PipelineTask:
    task = _task()
    w = Candidate(k=3, seed=3)
    w.js_code = "export default function generate(THREE) { return TAG:w }"
    w.rendered_png = b"w"
    w.judge_white_views = {n: b"w" for n in ALL_VIEWS}
    task.candidates, task.winner_k = [w], 3
    task.js_code, task.rendered_png = w.js_code, w.rendered_png
    return task


def _code(tag: str, broken: bool = False) -> str:
    return f"export default function generate(THREE) {{ {'BROKEN ' if broken else ''}return TAG:{tag} }}"


async def _run_refine(task, coder, critic, judge, **kw):
    sem = asyncio.Semaphore(8)
    return await refine_regen_stage(
        task, coder=coder, critic=critic, judge=judge, embedder=None, js_checker=FakeChecker(),
        renderer=FakeRenderer(), sem_coder=sem, sem_checker=sem, sem_renderer=sem, sem_judge=sem,
        sem_critic=sem, status={}, n_cand=4, accept_margin=0.25, thinking=True, max_tokens=20000,
        seed_offset=7300013, render_from_object=True, orientation=OrientationConfig(enabled=False), **kw,
    )


class RefineTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_the_regen_that_wins_by_the_most(self):
        quality = {"w": 5.0, "r0": 5.2, "r1": 6.0, "r2": 4.0, "r3": 5.5}   # r0 hon 0,2 < 0,25; r3 hong roi duoc sua
        coder = FakeCoder({0: _code("r0"), 1: _code("r1"), 2: _code("r2"), 3: _code("r3", broken=True)})
        task, judge = _refine_task(), FakeJudge(quality)
        info = await _run_refine(task, coder, FakeCritic(), judge)
        self.assertEqual(info["accepted"], 1)
        self.assertEqual(task.winner_k, REGEN_K_BASE + 1)
        self.assertEqual(task.js_code, _code("r1"))
        self.assertEqual(task.rendered_png, b"r1")
        self.assertEqual(coder.fixes, 1)
        self.assertEqual(coder.thinking, [True] * 4)
        passed = {c["c"]: c["passed"] for c in info["cands"] if "passed" in c}
        self.assertEqual(passed, {0: False, 1: True, 2: False, 3: True})
        self.assertTrue(all(left.startswith("r") and right == "w" for left, right in judge.calls))  # ban moi ben trai (A)
        self.assertEqual(sorted(c.k for c in task.candidates), [3, 1000, 1001, 1002, 1003])

    async def test_small_wins_keep_the_bracket_winner(self):
        quality = {"w": 5.0, "r0": 5.1, "r1": 5.2, "r2": 4.0, "r3": 3.0}
        coder = FakeCoder({i: _code(f"r{i}") for i in range(4)})
        task = _refine_task()
        info = await _run_refine(task, coder, FakeCritic(), FakeJudge(quality))
        self.assertIsNone(info["accepted"])
        self.assertEqual(task.winner_k, 3)
        self.assertEqual(task.js_code, _code("w"))

    async def test_coder_failure_is_swallowed(self):
        task = _refine_task()
        info = await _run_refine(task, FakeCoder({}, fail=True), FakeCritic(), FakeJudge({"w": 5.0}))
        self.assertIsNone(info["accepted"])
        self.assertEqual(task.js_code, _code("w"))
        self.assertEqual({c.get("dropped") for c in info["cands"]}, {"coder:RuntimeError"})

    async def test_no_critic_feedback_means_no_regen(self):
        coder = FakeCoder({i: _code(f"r{i}") for i in range(4)})
        task = _refine_task()
        info = await _run_refine(task, coder, FakeCritic(fail=True), FakeJudge({"w": 5.0}))
        self.assertIsNone(info["accepted"])
        self.assertEqual(coder.thinking, [])
        self.assertEqual(task.js_code, _code("w"))

    def test_s1_margin_sign(self):
        v = JudgeVerdict(winner="A", detail={"s1_slim": {"angles": [{"pen_a": 3, "pen_b": 5}, {"pen_a": 4, "pen_b": 5}]}})
        self.assertAlmostEqual(s1_margin(v), 1.5)
        self.assertIsNone(s1_margin(JudgeVerdict(winner="A")))


def _pipeline(refine: RefineConfig) -> Pipeline:
    return Pipeline(
        planner=None, coder=object(), critic=object(), judge=object(), js_checker=object(), renderer=object(),
        session_store=None, http_client=None, refine=refine,
    )


class RefineGateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._orig = orch_mod.refine_regen_stage

    async def asyncTearDown(self):
        orch_mod.refine_regen_stage = self._orig

    async def test_late_bracket_skips_refine(self):
        called = []

        async def fake(task, **kw):
            called.append(1)
        orch_mod.refine_regen_stage = fake
        task = _refine_task()
        task.started_at = time.time() - 10.0
        await _pipeline(RefineConfig(enabled=True, start_cutoff_s=5.0, deadline_s=8.0))._maybe_refine(task)
        self.assertEqual(called, [])
        self.assertIn("late", task.meta["refine"]["skipped"])

    async def test_retried_task_uses_the_batch_clock(self):
        called = []

        async def fake(task, **kw):
            called.append(1)
        orch_mod.refine_regen_stage = fake
        task = _refine_task()
        task.batch_started_at = time.time() - 10.0          # lo da chay 10 s
        task.started_at = time.time()                        # de vua duoc chay lai
        await _pipeline(RefineConfig(enabled=True, start_cutoff_s=5.0, deadline_s=8.0))._maybe_refine(task)
        self.assertEqual(called, [])
        self.assertIn("late", task.meta["refine"]["skipped"])

    async def test_deadline_cuts_refine_and_keeps_winner(self):
        async def slow(task, **kw):
            await asyncio.sleep(30)
            task.js_code = "SHOULD NOT APPEAR"
        orch_mod.refine_regen_stage = slow
        task = _refine_task()
        task.started_at = time.time()
        t0 = time.monotonic()
        await _pipeline(RefineConfig(enabled=True, start_cutoff_s=0.5, deadline_s=1.0))._maybe_refine(task)
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertIn("timeout", task.meta["refine"])
        self.assertEqual(task.js_code, _code("w"))

    async def test_refine_error_never_fails_the_task(self):
        async def boom(task, **kw):
            raise ValueError("bug")
        orch_mod.refine_regen_stage = boom
        task = _refine_task()
        task.started_at = time.time()
        await _pipeline(RefineConfig(enabled=True, start_cutoff_s=5.0, deadline_s=8.0))._maybe_refine(task)
        self.assertIn("ValueError", task.meta["refine"]["error"])
        self.assertEqual(task.js_code, _code("w"))

    async def test_disabled_does_nothing(self):
        task = _refine_task()
        await _pipeline(RefineConfig(enabled=False))._maybe_refine(task)
        self.assertNotIn("refine", task.meta)


class ConfigAndOutputTests(unittest.TestCase):
    def test_shipped_configuration(self):
        from config.settings import settings
        self.assertEqual(settings.actors.coder.ensemble_size, 36)
        self.assertEqual(settings.pipeline.prefilter_keep, 20)
        r = settings.pipeline.refine
        self.assertTrue(r.enabled)
        self.assertLess(r.start_cutoff_s, r.deadline_s)
        self.assertLess(r.deadline_s, settings.event_bus.task_deadline_s)
        self.assertEqual(settings.pipeline.seed_offset, 7300013)

    def test_refine_deadline_must_fit_task_deadline(self):
        import copy

        import yaml
        shipped = yaml.safe_load(open(_CFG))
        SettingsConf.model_validate(copy.deepcopy(shipped))             # doi chung: cau hinh nop phai hop le
        bad = copy.deepcopy(shipped)
        bad["pipeline"]["refine"]["deadline_s"] = 4000
        with self.assertRaisesRegex(Exception, "task_deadline_s"):
            SettingsConf.model_validate(bad)

    def test_regen_output_cleanup(self):
        n = SceneCoderAgent._normalize_regen_output
        body = "export default function generate(THREE) {\n  return new THREE.Group();\n}"
        self.assertEqual(n("<think>plan the parts</think>\n" + body), body)
        self.assertEqual(n("Here is the module:\n" + body), body)
        self.assertEqual(n("```javascript\n" + body + "\n```"), body)
        self.assertEqual(n(body), body)
        # luong sinh bai chinh giu nguyen: ham cu khong cat loi dan
        self.assertTrue(SceneCoderAgent._normalize_js_output("Here:\n" + body).startswith("Here:"))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------- tich hop: tron mot de qua Pipeline.run_one
class _FlowCoder(FakeCoder):
    """Coder gia cho ca luong: ung vien k -> 'TAG:k{k}', ban ve lai ci -> 'TAG:r{ci}'."""

    def __init__(self) -> None:
        super().__init__({i: _code(f"r{i}") for i in range(4)}, winner_grid=b"k3")
        self.seed = 0

    async def code(self, task_id, osd, image_bytes=None, image_mime="image/jpeg", *,
                   actor_override=None, seed_override=None, temperature_override=None):
        k = int(actor_override.split("#k")[1])
        return _code(f"k{k}")


class _FlowEmbedder(FakeEmbedder):
    async def embed_reference(self, image_bytes):
        return "refvec"

    async def build_candidate_embeddings(self, ref_vec, views):
        return None


class FlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._prep = orch_mod.prepare_inputs_stage

    async def asyncTearDown(self):
        orch_mod.prepare_inputs_stage = self._prep

    async def test_run_one_prefilter_bracket_then_refine(self):
        async def fake_prepare(task, **kw):
            task.image_bytes, task.osd = b"ref", None
        orch_mod.prepare_inputs_stage = fake_prepare
        from llm.session_store import SessionStore
        quality = {"k0": 2, "k1": 9, "k2": 4, "k3": 6, "k4": 5, "k5": 3,      # k1 tot nhat nhung DINO loai
                   "r0": 6.1, "r1": 7.5, "r2": 1, "r3": 6.2}
        scores = {0: 0.5, 1: 0.1, 2: 0.6, 3: 0.9, 4: 0.8, 5: 0.7}
        judge = FakeJudge(quality)
        p = Pipeline(
            planner=None, coder=_FlowCoder(), critic=FakeCritic(), judge=judge, embedder=_FlowEmbedder(scores),
            js_checker=FakeChecker(), renderer=FakeRenderer(), session_store=SessionStore(), http_client=None,
            use_planner=False, coder_multimodal=True, coder_ensemble_size=6, coder_ensemble_temperature=0.8,
            render_from_object=True, seed_offset=7300013, refinement_enabled=False, task_deadline_s=120,
            prefilter_keep=4, refine=RefineConfig(enabled=True, start_cutoff_s=60, deadline_s=90),
            coder_limit=8, js_checker_limit=8, renderer_limit=8, critic_limit=8, judge_limit=8,
        )
        task = await p.run_one(PipelineTask(stem="flow0", image_url="https://example.invalid/r.png", seed=11))
        self.assertFalse(task.failed, task.failure_reason)
        self.assertEqual(task.meta["prefilter"]["kept"], [2, 3, 4, 5])
        self.assertEqual(task.meta["refine"]["accepted"], 1)                 # r1 (7,5) hon k3 (6) 1,5 diem
        self.assertEqual(task.winner_k, REGEN_K_BASE + 1)
        self.assertEqual(task.js_code, _code("r1"))
        self.assertEqual(task.best_js_code, _code("r1"))                    # dau ra cuoi cua de
        self.assertEqual(task.best_rendered_png, b"r1")


# ---------------------------------------------------------------- ham that cua coder/critic voi client gia
class _Choice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class _FakeOpenAI:
    def __init__(self, content):
        self.content, self.kwargs = content, None
        outer = self

        class _Completions:
            async def create(self, **kw):
                outer.kwargs = kw
                return type("R", (), {"choices": [_Choice(outer.content)]})()
        self.chat = type("C", (), {"completions": _Completions()})()


class RealAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_struct_critique_parses_json_after_think(self):
        from config.settings import settings
        from modules.critic.agent import CriticAgent
        client = _FakeOpenAI('<think>x</think>{"structural_issues": [{"part": "rim", "should_be": "thin_member"}], "verdict": "rim too thick"}')
        critic = CriticAgent(client, settings=settings.actors.critic)
        out = await critic.struct_critique(task_id="t", image_bytes=b"i", image_mime="image/png", render_png=b"r",
                                           js_code="const rim = new THREE.Mesh(a, b);", seed=12, temperature=0.3)
        self.assertEqual(out["structural_issues"][0]["part"], "rim")
        self.assertEqual(client.kwargs["seed"], 12)
        self.assertEqual(client.kwargs["extra_body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("rim", client.kwargs["messages"][1]["content"][-1]["text"])

    async def test_code_regen_sends_reference_render_feedback_and_cleans_output(self):
        from config.settings import settings
        from llm.session_store import SessionStore
        body = "export default function generate(THREE) {\n  return new THREE.Group();\n}"
        coder = SceneCoderAgent(None, session_store=SessionStore(), settings=settings.actors.coder)
        seen = {}

        class _Session:
            async def run(self, msg):
                seen["msg"] = msg
                return "<think>parts first</think>\nModule below.\n" + body

        def build(task_id, actor, **kw):
            seen["kw"] = kw
            return _Session()
        coder._build_regen_session = build
        out = await coder.code_regen("t", image_bytes=b"ref", image_mime="image/png", render_png=b"grid",
                                     feedback="ISSUES ...", seed=5, temperature=0.7, enable_thinking=True,
                                     max_tokens=20000, actor="coder#regen0")
        self.assertEqual(out, body)
        self.assertEqual(seen["kw"], {"seed": 5, "temperature": 0.7, "enable_thinking": True, "max_tokens": 20000})
        kinds = [part["type"] for part in seen["msg"]]
        self.assertEqual(kinds, ["image_url", "image_url", "text"])
        self.assertIn("ISSUES ...", seen["msg"][2]["text"])
        self.assertNotIn("new THREE.", seen["msg"][2]["text"])               # khong dua ma cu
