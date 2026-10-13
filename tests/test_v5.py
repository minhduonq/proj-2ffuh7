"""Tests for the v5 additions on top of cddc197: rule-per-error checker repair, early duplicate fold, fresh-refine
(critic -> regenerate without old code) with its batch-clock gate, and the shipped configuration.

No pytest needed:
    in the image:  docker run --rm -v "$PWD/tests:/tests:ro" -w /workspace <image> python -m unittest discover -s /tests -p 'test_v5*.py'
    in the repo:   python -m unittest discover -s tests -p 'test_v5*.py'   (needs the packages of docker/requirements.lock.txt)
The real-checker tests run only where the checker's node_modules exist (the image), and are skipped elsewhere.
"""
from __future__ import annotations

import asyncio
import copy
import os
import shutil
import sys
import tempfile
import time
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SVC = os.path.join(_REPO, "pipeline_service")
if not os.path.isdir(_SVC):
    _SVC = "/workspace"                                   # tests mounted into the image
_CFG = os.path.join(_REPO, "configuration.yaml")
if not os.path.isfile(_CFG):
    _CFG = os.path.join(_SVC, "configuration.yaml")
sys.path.insert(0, _SVC)
os.chdir(tempfile.mkdtemp(prefix="w5test_"))              # logger_config writes logs/ under the CWD

import yaml  # noqa: E402

from config.settings import OrientationConfig, RefineConfig, SettingsConf  # noqa: E402
from llm.session_store import SessionStore  # noqa: E402
from modules.judge.schema import JudgeVerdict  # noqa: E402
from modules.scene_coder.agent import SceneCoderAgent  # noqa: E402
from pipeline import code_fix  # noqa: E402
from pipeline import orchestrator as orch_mod  # noqa: E402
from pipeline.orchestrator import Pipeline  # noqa: E402
from pipeline.refine import REGEN_K_BASE, refine_regen_stage, s1_margin  # noqa: E402
from pipeline.stages import _CandidateFactory  # noqa: E402
from pipeline.task import Candidate, PipelineTask  # noqa: E402

FRONT = ("front_left", "front_right", "front_below", "front_above")
ALL_VIEWS = FRONT + ("right", "back", "left", "top_down")


# ----------------------------------------------------------------------------------------------- fakes
class FakeJudge:
    """Decides from a hidden quality table (higher = better); S1 penalty = 9 - quality on all four angles."""

    def __init__(self, quality: dict[str, float]) -> None:
        self.quality = quality
        self.calls: list[tuple[str, str, bool]] = []
        self.seed = 0

    def early_stop_for_round(self, round_no):
        return False

    async def compare(self, *, task_id, match_label, reference_bytes, reference_mime, render_a, render_b,
                      white_views_a=None, white_views_b=None, gray_views_a=None, gray_views_b=None,
                      embeddings_a=None, embeddings_b=None, early_stop=False):
        assert reference_bytes and white_views_a and white_views_b
        a, b = render_a.decode(), render_b.decode()
        qa, qb = self.quality[a], self.quality[b]
        self.calls.append((a, b, early_stop))
        angles = [{"label": n, "pen_a": 9 - qa, "pen_b": 9 - qb} for n in FRONT]
        return JudgeVerdict(winner="A" if qa >= qb else "B", decided_by="S1", detail={"s1_slim": {"angles": angles}})


class _Report:
    overall_score = 0.4
    issues = [{"category": "shape", "description": "legs too thick"}]
    matching_aspects = ["colour"]


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


def _code(tag: str, broken: bool = False) -> str:
    return f"export default function generate(THREE) {{ {'BROKEN ' if broken else ''}return TAG:{tag} }}"


class FakeCoder:
    def __init__(self, codes: dict[int, str], fail: bool = False, winner_grid: bytes = b"w") -> None:
        self.codes, self.fail, self.winner_grid = codes, fail, winner_grid
        self.thinking: list[bool] = []
        self.fixes = 0
        self.seed = 0

    async def code_regen(self, task_id, *, image_bytes, image_mime, render_png, feedback, seed,
                         temperature, enable_thinking=True, max_tokens=20000, actor="coder#regen"):
        if self.fail:
            raise RuntimeError("coder down")
        assert "BUILD-TECHNIQUE" in feedback and render_png == self.winner_grid   # regen sees the winner's grid
        self.thinking.append(enable_thinking)
        return self.codes[int(actor.replace("coder#regen", ""))]

    async def code_regen_fix(self, task_id, *, js_code, error, seed, max_tokens=20000, actor=""):
        self.fixes += 1
        assert "PARSE_ERROR" in error
        return js_code.replace("BROKEN ", "")


class FakeChecker:
    """'BROKEN' -> an unfixable parse error; 'DUP' -> the real duplicate-declaration message (rule-fixable)."""

    def __init__(self) -> None:
        self.calls = 0

    async def process(self, task, mode="sanity"):
        self.calls += 1
        code = task.js_code or ""
        if "BROKEN" in code:
            task.js_valid, task.js_errors = False, ["PARSE_ERROR: Unexpected token (1:40)"]
        elif "const mat = 1;\nconst mat = 2;" in code:
            task.js_valid, task.js_errors = False, ["PARSE_ERROR: Identifier 'mat' has already been declared. (3:6)"]
        else:
            task.js_valid, task.js_errors = True, []
        task.scene_json = {"object": 1} if task.js_valid and mode == "with_object" else None
        return task


class FakeRenderer:
    class config:                                       # noqa: N801 - mimics RendererConfig
        judge_multiview = True

    async def process_with_judge_views(self, task, slot=None):
        task.rendered_png = task.js_code.split("TAG:")[1].split()[0].encode()
        return {n: b"w" for n in ALL_VIEWS}, {"front_left": b"g"}

    async def process(self, task, slot=None):
        task.rendered_png = task.js_code.split("TAG:")[1].split()[0].encode()
        return task


def _task() -> PipelineTask:
    t = PipelineTask(stem="stem0", image_url="https://example.invalid/ref.png", seed=11)
    t.image_bytes = b"ref"
    return t


def _refine_task() -> PipelineTask:
    task = _task()
    w = Candidate(k=3, seed=3)
    w.js_code = _code("w")
    w.rendered_png = b"w"
    w.judge_white_views = {n: b"w" for n in ALL_VIEWS}
    task.candidates, task.winner_k = [w], 3
    task.js_code, task.rendered_png = w.js_code, w.rendered_png
    return task


def _factory(task, coder=None, checker=None, code_fix_on=True):
    sem = asyncio.Semaphore(8)
    return _CandidateFactory(
        task, coder=coder, judge=FakeJudge({}), embedder=None, js_checker=checker or FakeChecker(),
        renderer=FakeRenderer(), sem_coder=sem, sem_checker=sem, sem_renderer=sem, status={}, osd=None,
        send_image=True, checker_mode="with_object", ensemble_temperature=0.7, seed_offset=7313013,
        orientation=OrientationConfig(enabled=False), code_fix=code_fix_on,
    )


# ----------------------------------------------------------------------------------------------- code_fix rules
GEN_AFTER = """export default function generate(THREE) {
  const root = new THREE.Group();
  root.add(makePart(0.5));
  return root;
}

function makePart(s) {
  return new THREE.Mesh(new THREE.BoxGeometry(s, s, s), new THREE.MeshStandardMaterial({ color: '#888888' }));
}
"""


class CodeFixRuleTests(unittest.TestCase):
    def test_parse_errors_splits_rule_and_detail(self):
        self.assertEqual(code_fix.parse_errors(["PARSE_ERROR: Identifier 'a' has already been declared. (3:6)", "X"]),
                         [{"rule": "PARSE_ERROR", "detail": "Identifier 'a' has already been declared. (3:6)"},
                          {"rule": "X", "detail": ""}])

    def test_duplicate_declaration_renames_from_that_line(self):
        src = "const mat = 1;\nuse(mat);\nconst mat = 2;\nuse(mat, obj.mat);"
        new, used = code_fix.fix_once(src, code_fix.parse_errors(
            ["PARSE_ERROR: Identifier 'mat' has already been declared. (3:6)"]))
        self.assertEqual(used, ["duplicate_declaration"])
        self.assertEqual(new, "const mat = 1;\nuse(mat);\nconst mat__2 = 2;\nuse(mat__2, obj.mat);")

    def test_three_outside_generate_moves_helpers_inside(self):
        new, used = code_fix.fix_once(GEN_AFTER, [{"rule": "THREE_AT_TOP_LEVEL",
                                                   "detail": "THREE referenced outside generate function at line 8"}])
        self.assertEqual(used, ["three_outside_generate"])
        self.assertTrue(new.startswith("export default function generate(THREE) {"))
        self.assertLess(new.index("function makePart"), new.index("const root"))   # helper now inside, before the body
        self.assertEqual(new.count("function makePart"), 1)
        self.assertEqual(new.rstrip()[-1], "}")
        body = new[new.index("{") + 1: new.rstrip().rindex("}")]
        self.assertIn("return root;", body)                       # original body kept, after the helper

    def test_braces_inside_strings_and_comments_are_ignored(self):
        src = 'export default function generate(THREE) {\n  const s = "}}";  // }\n  /* { */\n  return h();\n}\nfunction h() { return new THREE.Group(); }\n'
        new, used = code_fix.fix_once(src, [{"rule": "THREE_AT_TOP_LEVEL", "detail": "x"}])
        self.assertEqual(used, ["three_outside_generate"])
        self.assertIn('const s = "}}";', new)
        self.assertTrue(new.rstrip().endswith("}"))
        self.assertEqual(new.count("function h()"), 1)

    def test_broken_hex_colour(self):
        for line, err, want in [
            ("    color: 0xb9deds,", "Identifier directly after number. (1:18)", "    color: 0xb9dedd,"),
            ("    color: 0x39dff�,", "Unexpected character '�'. (1:18)", "    color: 0x39dfff,"),
        ]:
            new, used = code_fix.fix_once(line, [{"rule": "PARSE_ERROR", "detail": err}])
            self.assertEqual((new, used), (want, ["hex_colour"]))

    def test_const_rules(self):
        new, used = code_fix.fix_once("let a = 1;\nconst b;\n", [{"rule": "PARSE_ERROR",
                                                                  "detail": "Missing initializer in const declaration. (2:7)"}])
        self.assertEqual((new, used), ("let a = 1;\nlet b;\n", ["const_without_value"]))
        new, used = code_fix.fix_once("const a = 1;\na = 2;", [{"rule": "EXECUTION_THREW",
                                                              "detail": "Assignment to constant variable."}])
        self.assertEqual((new, used), ("let a = 1;\na = 2;", ["assignment_to_const"]))

    def test_misspelt_identifier(self):
        src = "const brand_word = 1;\nuse(brandWord);"
        new, used = code_fix.fix_once(src, [{"rule": "IDENTIFIER_NOT_ALLOWED", "detail": "brandWord at line 2"}])
        self.assertEqual(used, ["misspelt_identifier"])
        self.assertIn("use(brand_word)", new)

    def test_unrelated_failures_change_nothing(self):
        for f in ({"rule": "VERTEX_LIMIT_EXCEEDED", "detail": "450886 vertices"},
                  {"rule": "IDENTIFIER_NOT_ALLOWED", "detail": "frontCenter at line 3"},
                  {"rule": "EXECUTION_THREW", "detail": "Cannot read properties of undefined (reading 'x')"}):
            self.assertEqual(code_fix.fix_once("const a = 1;\nfoo(a);", [f]), ("const a = 1;\nfoo(a);", []))


class CodeFixInValidateTests(unittest.IsolatedAsyncioTestCase):
    DUP = _code("d").replace("{ return", "{\nconst mat = 1;\nconst mat = 2;\n return")

    async def test_rule_fix_rescues_a_checker_failure(self):
        task, checker = _task(), FakeChecker()
        cand = Candidate(k=2, seed=2)
        cand.js_code = self.DUP
        await _factory(task, checker=checker)._validate(cand)
        self.assertIsNone(cand.drop_reason)
        self.assertTrue(cand.js_valid)
        self.assertIn("const mat__2 = 2;", cand.js_code)
        self.assertEqual(cand.rendered_png, b"d")
        self.assertEqual(task.meta["code_fix"][2], {"rules": ["duplicate_declaration"], "passed": True})
        self.assertEqual(checker.calls, 2)

    async def test_disabled_or_unfixable_still_drops(self):
        task = _task()
        cand = Candidate(k=2, seed=2)
        cand.js_code = self.DUP
        await _factory(task, code_fix_on=False)._validate(cand)
        self.assertEqual(cand.drop_reason, "checker")
        self.assertEqual(cand.js_code, self.DUP)
        cand2 = Candidate(k=3, seed=3)
        cand2.js_code = _code("b", broken=True)
        await _factory(task)._validate(cand2)
        self.assertEqual(cand2.drop_reason, "checker")
        self.assertNotIn(3, task.meta.get("code_fix", {}))


def _checker_node_cwd() -> str | None:
    svc_mod = os.path.join(_SVC, "modules", "js_checker")
    for c in (os.environ.get("NODE_CWD", ""), "/workspace", svc_mod):
        if c and os.path.isdir(os.path.join(c, "node_modules", "three")) and os.path.isdir(os.path.join(c, "node_modules", "@babel")):
            return c
    return None


FIT = """function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const scale = 0.95 / maxDim;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}
"""


@unittest.skipIf(shutil.which("node") is None or _checker_node_cwd() is None, "checker node_modules not available")
class RealCheckerTests(unittest.IsolatedAsyncioTestCase):
    """The real BTC checker copy (with_object mode, as shipped) rejects these, and the rules make them pass."""

    async def _check(self, code: str) -> PipelineTask:
        from modules.js_checker.module import JSCheckerModule
        from modules.js_checker.settings import JSCheckerConfig
        os.environ.setdefault("NODE_CWD", _checker_node_cwd())
        t = PipelineTask(stem="real", image_url="x")
        t.js_code = code
        await JSCheckerModule(JSCheckerConfig()).process(t, mode="with_object")
        return t

    async def _assert_rescued(self, code: str, rule: str):
        t = await self._check(code)
        self.assertFalse(t.js_valid)
        self.assertTrue(any(e.startswith(rule) for e in t.js_errors), t.js_errors)
        fixed = code
        for _ in range(4):
            fixed, used = code_fix.fix_once(fixed, code_fix.parse_errors(t.js_errors))
            self.assertTrue(used, t.js_errors)
            t = await self._check(fixed)
            if t.js_valid:
                break
        self.assertTrue(t.js_valid, t.js_errors)

    async def test_three_outside_generate(self):
        code = ("export default function generate(THREE) {\n  const root = new THREE.Group();\n  root.add(part(0.4));\n"
                "  fitToUnitCube(THREE, root);\n  return root;\n}\n"
                "function part(s) {\n  return new THREE.Mesh(new THREE.BoxGeometry(s, s, s), "
                "new THREE.MeshStandardMaterial({ color: 0x8899aa }));\n}\n" + FIT)
        await self._assert_rescued(code, "THREE_AT_TOP_LEVEL")

    async def test_duplicate_declarations(self):
        code = ("export default function generate(THREE) {\n  const root = new THREE.Group();\n"
                "  const geo = new THREE.BoxGeometry(1, 1, 1);\n  const mat = new THREE.MeshStandardMaterial({ color: 0x336699 });\n"
                "  root.add(new THREE.Mesh(geo, mat));\n  const geo = new THREE.SphereGeometry(0.5, 16, 12);\n"
                "  root.add(new THREE.Mesh(geo, mat));\n  const mat = new THREE.MeshStandardMaterial({ color: 0x993366 });\n"
                "  root.add(new THREE.Mesh(geo, mat));\n  fitToUnitCube(THREE, root);\n  return root;\n\n" + FIT + "}\n")
        await self._assert_rescued(code, "PARSE_ERROR")

    async def test_broken_hex_colour(self):
        code = ("export default function generate(THREE) {\n  const root = new THREE.Group();\n"
                "  const mat = new THREE.MeshStandardMaterial({\n    color: 0xb9deds,\n    roughness: 0.5,\n  });\n"
                "  root.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mat));\n  fitToUnitCube(THREE, root);\n"
                "  return root;\n\n" + FIT + "}\n")
        await self._assert_rescued(code, "PARSE_ERROR")


# ----------------------------------------------------------------------------------------------- early duplicate fold
class _Gen:
    def __init__(self, code):
        self.code = code


class EarlyDedupTests(unittest.IsolatedAsyncioTestCase):
    async def _factory_with(self, codes: dict[int, str], unfinished: set[int] = frozenset()):
        task = _task()
        f = _factory(task)
        gates = {k: asyncio.Event() for k in codes}

        async def gen(k):
            if k in unfinished:
                await gates[k].wait()
            c = Candidate(k=k, seed=k)
            c.js_code = codes[k]
            return c
        f._coder_tasks = {k: asyncio.create_task(gen(k)) for k in codes}
        await asyncio.sleep(0)
        return f, gates

    async def test_duplicate_of_a_finished_candidate_skips_validation(self):
        checker = FakeChecker()
        f, _ = await self._factory_with({0: _code("a"), 1: _code("b"), 2: _code("a")})
        f.js_checker = checker
        self.assertIsNone(await f.leaf(2))
        self.assertEqual(f._coder_tasks[2].result().drop_reason, "duplicate_of_k0")
        self.assertEqual(checker.calls, 0)

    async def test_unfinished_earlier_coder_keeps_the_full_scan(self):
        f, gates = await self._factory_with({0: _code("a"), 1: _code("a")}, unfinished={0})
        leaf = asyncio.create_task(f.leaf(1))
        await asyncio.sleep(0.05)
        self.assertFalse(leaf.done())                    # waits for k0 inside the original full scan
        gates[0].set()
        self.assertIsNone(await leaf)
        self.assertEqual(f._coder_tasks[1].result().drop_reason, "duplicate_of_k0")

    async def test_unique_candidate_is_validated(self):
        f, _ = await self._factory_with({0: _code("a"), 1: _code("b")})
        cand = await f.leaf(1)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.rendered_png, b"b")


# ----------------------------------------------------------------------------------------------- refine
async def _run_refine(task, coder, critic, judge, **kw):
    sem = asyncio.Semaphore(8)
    return await refine_regen_stage(
        task, coder=coder, critic=critic, judge=judge, embedder=None, js_checker=FakeChecker(),
        renderer=FakeRenderer(), sem_coder=sem, sem_checker=sem, sem_renderer=sem, sem_judge=sem,
        sem_critic=sem, status={}, n_cand=4, accept_margin=0.25, thinking=True, max_tokens=20000,
        seed_offset=7313013, render_from_object=True, orientation=OrientationConfig(enabled=False), **kw,
    )


class RefineTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_the_regen_that_wins_by_the_most(self):
        quality = {"w": 5.0, "r0": 5.2, "r1": 6.0, "r2": 4.0, "r3": 5.5}   # r0 +0.2 < 0.25; r3 broken then fixed
        coder = FakeCoder({0: _code("r0"), 1: _code("r1"), 2: _code("r2"), 3: _code("r3", broken=True)})
        task, judge, critic = _refine_task(), FakeJudge(quality), FakeCritic()
        info = await _run_refine(task, coder, critic, judge)
        self.assertEqual(info["accepted"], 1)
        self.assertEqual(task.winner_k, REGEN_K_BASE + 1)
        self.assertEqual(task.js_code, _code("r1"))
        self.assertEqual(task.rendered_png, b"r1")
        self.assertEqual(coder.fixes, 1)
        self.assertEqual(coder.thinking, [True] * 4)
        passed = {c["c"]: c["passed"] for c in info["cands"] if "passed" in c}
        self.assertEqual(passed, {0: False, 1: True, 2: False, 3: True})
        self.assertTrue(all(a.startswith("r") and b == "w" and es is False for a, b, es in judge.calls))  # regen = A, full S1
        self.assertEqual(sorted(c.k for c in task.candidates), [3, 1000, 1001, 1002, 1003])
        self.assertEqual(len(set(critic.seeds)), 4)                          # one critic stream per regeneration

    async def test_small_wins_keep_the_bracket_winner(self):
        coder = FakeCoder({i: _code(f"r{i}") for i in range(4)})
        task = _refine_task()
        info = await _run_refine(task, coder, FakeCritic(), FakeJudge({"w": 5.0, "r0": 5.1, "r1": 5.2, "r2": 4.0, "r3": 3.0}))
        self.assertIsNone(info["accepted"])
        self.assertEqual((task.winner_k, task.js_code), (3, _code("w")))

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

    async def test_critic_seeds_are_prompt_scoped(self):
        seeds = []
        for stem in ("a", "b"):
            task, critic = _refine_task(), FakeCritic()
            task.stem = stem
            await _run_refine(task, FakeCoder({i: _code(f"r{i}") for i in range(4)}), critic,
                              FakeJudge({"w": 5.0, "r0": 1, "r1": 1, "r2": 1, "r3": 1}))
            seeds.append(sorted(critic.seeds))
        self.assertNotEqual(seeds[0], seeds[1])

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

    async def _gate(self, cfg, *, batch_age=None, task_age=0.0, stage=None):
        called = []

        async def fake(task, **kw):
            called.append(kw)
        orch_mod.refine_regen_stage = stage or fake
        task = _refine_task()
        task.started_at = time.time() - task_age
        if batch_age is not None:
            task.batch_started_at = time.time() - batch_age
        await _pipeline(cfg)._maybe_refine(task)
        return task, called

    async def test_late_batch_skips_refine(self):
        task, called = await self._gate(RefineConfig(enabled=True, start_cutoff_s=5.0, deadline_s=8.0), task_age=10.0)
        self.assertEqual(called, [])
        self.assertIn("late", task.meta["refine"]["skipped"])

    async def test_retried_task_uses_the_batch_clock(self):
        task, called = await self._gate(RefineConfig(enabled=True, start_cutoff_s=5.0, deadline_s=8.0), batch_age=10.0)
        self.assertEqual(called, [])
        self.assertIn("late", task.meta["refine"]["skipped"])

    async def test_early_task_runs_refine_with_config(self):
        cfg = RefineConfig(enabled=True, candidates=3, accept_margin=0.3, start_cutoff_s=5.0, deadline_s=8.0)
        _, called = await self._gate(cfg, batch_age=1.0)
        self.assertEqual(len(called), 1)
        self.assertEqual((called[0]["n_cand"], called[0]["accept_margin"]), (3, 0.3))

    async def test_deadline_cuts_refine_and_keeps_winner(self):
        async def slow(task, **kw):
            await asyncio.sleep(30)
            task.js_code = "SHOULD NOT APPEAR"
        t0 = time.monotonic()
        task, _ = await self._gate(RefineConfig(enabled=True, start_cutoff_s=0.5, deadline_s=1.0), stage=slow)
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertIn("timeout", task.meta["refine"])
        self.assertEqual(task.js_code, _code("w"))

    async def test_refine_error_never_fails_the_task(self):
        async def boom(task, **kw):
            raise ValueError("bug")
        task, _ = await self._gate(RefineConfig(enabled=True, start_cutoff_s=5.0, deadline_s=8.0), stage=boom)
        self.assertIn("ValueError", task.meta["refine"]["error"])
        self.assertEqual(task.js_code, _code("w"))

    async def test_disabled_does_nothing(self):
        task, called = await self._gate(RefineConfig(enabled=False))
        self.assertEqual(called, [])
        self.assertNotIn("refine", task.meta)

    async def test_startup_warmup_skips_refine(self):
        called = []

        async def fake(task, **kw):
            called.append(1)
        orch_mod.refine_regen_stage = fake
        task = _refine_task()
        task.stem, task.started_at = "warmup_task", time.time()
        await _pipeline(RefineConfig(enabled=True, start_cutoff_s=5.0, deadline_s=8.0))._maybe_refine(task)
        self.assertEqual(called, [])


# ----------------------------------------------------------------------------------------------- one prompt end to end
class _FlowCoder(FakeCoder):
    """Bracket candidate k -> 'TAG:k{k}' (k2 has a rule-fixable duplicate declaration); regeneration ci -> 'TAG:r{ci}'."""

    def __init__(self) -> None:
        super().__init__({i: _code(f"r{i}") for i in range(4)}, winner_grid=b"k3")

    async def code(self, task_id, osd, image_bytes=None, image_mime="image/jpeg", *,
                   actor_override=None, seed_override=None, temperature_override=None):
        k = int(actor_override.split("#k")[1])
        code = _code(f"k{k}")
        return code.replace("{ return", "{\nconst mat = 1;\nconst mat = 2;\n return") if k == 2 else code


class FlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._prep = orch_mod.prepare_inputs_stage

    async def asyncTearDown(self):
        orch_mod.prepare_inputs_stage = self._prep

    async def test_run_one_bracket_code_fix_then_refine(self):
        async def fake_prepare(task, **kw):
            task.image_bytes, task.osd = b"ref", None
        orch_mod.prepare_inputs_stage = fake_prepare
        quality = {"k0": 2, "k1": 4, "k2": 5, "k3": 6, "k4": 3, "k5": 1, "r0": 6.1, "r1": 7.5, "r2": 1, "r3": 6.2}
        p = Pipeline(
            planner=None, coder=_FlowCoder(), critic=FakeCritic(), judge=FakeJudge(quality), embedder=None,
            js_checker=FakeChecker(), renderer=FakeRenderer(), session_store=SessionStore(), http_client=None,
            use_planner=False, coder_multimodal=True, coder_ensemble_size=6, coder_ensemble_temperature=0.7,
            render_from_object=True, seed_offset=7313013, refinement_enabled=False, task_deadline_s=120,
            orientation=OrientationConfig(enabled=False), code_fix=True,
            refine=RefineConfig(enabled=True, start_cutoff_s=60, deadline_s=90),
            coder_limit=8, js_checker_limit=8, renderer_limit=8, critic_limit=8, judge_limit=8,
        )
        task = PipelineTask(stem="flow0", image_url="https://example.invalid/r.png", seed=11)
        task.batch_started_at = time.time()
        task = await p.run_one(task)
        self.assertFalse(task.failed, task.failure_reason)
        self.assertEqual(task.meta["code_fix"][2]["passed"], True)            # k2 rescued and entered the bracket
        self.assertEqual(task.meta["refine"]["accepted"], 1)                  # r1 (7.5) beats k3 (6) by 1.5
        self.assertEqual(task.winner_k, REGEN_K_BASE + 1)
        self.assertEqual(task.best_js_code, _code("r1"))                     # the prompt's final output
        self.assertEqual(task.best_rendered_png, b"r1")


# ----------------------------------------------------------------------------------------------- configuration + agents
class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.shipped = yaml.safe_load(open(_CFG))

    def test_shipped_configuration(self):
        s = SettingsConf.model_validate(copy.deepcopy(self.shipped))
        self.assertEqual(s.actors.coder.ensemble_size, 34)
        self.assertEqual(s.pipeline.seed_offset, 7313013)
        self.assertFalse(s.pipeline.refinement_enabled)                    # the old critic->patch loop stays off
        self.assertTrue(s.pipeline.code_fix)
        r = s.pipeline.refine
        self.assertTrue(r.enabled)
        self.assertEqual((r.candidates, r.accept_margin, r.start_cutoff_s, r.deadline_s), (4, 0.25, 1200, 1450))
        self.assertLess(r.deadline_s, s.event_bus.task_deadline_s)
        self.assertLess(r.deadline_s, s.pipeline.batch_time_budget - 120)
        coder = s.llm_clients["coder-instance"].vllm
        self.assertEqual((coder.model, coder.revision), ("computer-vision-ai-lab/Qwen-3.6-27B-AmberKite",
                                                         "14adb741b473bbd7bc0d02cacb27247d2cb707f6"))

    def test_refine_deadline_must_fit_task_and_batch_deadlines(self):
        bad = copy.deepcopy(self.shipped)
        bad["pipeline"]["refine"]["deadline_s"] = 2800
        with self.assertRaisesRegex(Exception, "task_deadline_s"):
            SettingsConf.model_validate(bad)
        bad = copy.deepcopy(self.shipped)
        bad["event_bus"]["task_deadline_s"] = 5000
        bad["pipeline"]["refine"]["deadline_s"] = 2900
        with self.assertRaisesRegex(Exception, "batch_time_budget"):
            SettingsConf.model_validate(bad)
        bad = copy.deepcopy(self.shipped)
        bad["pipeline"]["refine"]["start_cutoff_s"] = 1500
        with self.assertRaisesRegex(Exception, "greater than start_cutoff_s"):
            SettingsConf.model_validate(bad)

    def test_regen_output_cleanup(self):
        n = SceneCoderAgent._normalize_regen_output
        body = "export default function generate(THREE) {\n  return new THREE.Group();\n}"
        self.assertEqual(n("<think>plan the parts</think>\n" + body), body)
        self.assertEqual(n("Here is the module:\n" + body), body)
        self.assertEqual(n("```javascript\n" + body + "\n```"), body)
        self.assertEqual(n(body), body)
        self.assertTrue(SceneCoderAgent._normalize_js_output("Here:\n" + body).startswith("Here:"))   # main path unchanged


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
        self.assertEqual([part["type"] for part in seen["msg"]], ["image_url", "image_url", "text"])
        self.assertIn("ISSUES ...", seen["msg"][2]["text"])
        self.assertNotIn("new THREE.", seen["msg"][2]["text"])               # the old code is never sent


if __name__ == "__main__":
    unittest.main()
