"""Every text sent to the models. Edit freely; keep it deterministic (no clocks, no randomness)."""

from __future__ import annotations

from pathlib import Path

SPECIFICATION = (Path(__file__).resolve().parents[1] / "instructions.md").read_text(encoding="utf-8")

# How the renders are judged and what usually loses duels; see reports/round0-competitor-analysis.md.
GUIDANCE = """Modeling rules:
- Pose: Y is up and +Z points toward the judge's front cameras (30 degrees left and right of +Z, slightly below, and from above). Keep the pose the target shows: lying, standing, leaning, open or closed, and the arrangement of separate items. Do not stand up, straighten or tidy the object into a catalogue pose. The side of the object the target shows faces +Z: an object seen from its side (a car or boat in profile) runs along X; a three-quarter view keeps the object's front at +Z.
- When one face carries the object's identity (clock, sign, picture, medallion, coin, card, book cover, screen, mirror, wall fan), that face faces +Z, so the front views see the face and not a thin edge.
- Model only the object: no floor, pedestal, backdrop or box around it. Parts touch where they are attached; nothing floats unless it floats in the target. Build the back, sides and underside as complete solid geometry, never a flat facade.
- Thin parts (wires, rods, chains, straps, frames, blades) are at least about 1% of the object's size thick, or they vanish in the renders. Openings (cups, bowls, pots, rings, tubes) get real wall thickness: a lathe profile with outer wall, rim and inner wall, or a Shape with holes.
- Colors: hex values from the target's lit mid-tones. Keep its hue and saturation: do not lighten, grey out or pastelise colors; vivid parts stay vivid and dark parts stay dark. White parts use an off-white (#e6e6e6 to #f0f0f0, never #ffffff) with visible seams or darker trim; black parts use #1c1c1c to #2a2a2a, never #000000.
- Materials: MeshStandardMaterial, metalness 0 for everything that is not metal. Metal needs a light base color, because it mostly reflects a dim environment and a mid or dark base color renders as charcoal: chrome #d8d8d8 metalness 1.0 roughness 0.15; brushed steel #b8b8b8 metalness 0.85 roughness 0.45; gold or brass #d8b23a metalness 0.85 roughness 0.35.
- Glass and clear plastic: MeshPhysicalMaterial with transparent true, opacity 0.35-0.5, the target's tint as color and roughness 0.1, plus slightly darker rims or edges. Never use transmission: it is nearly invisible on the white background. Keep parts opaque unless the target clearly shows them transparent.
- Complexity: aim for at most 60,000 vertices, 120 draw calls and about 2,000 InstancedMesh instances in total; heavier scenes time out in the renderer.
- Never put a unary minus or plus directly before `**`: `-x ** 2` is a syntax error. Write `-(x ** 2)` or `-(x * x)`.
- Detail: write a thorough module, typically 6,000-8,000 tokens of code, and spend that length on what the target shows: every distinct part, accurate profiles and proportions, bevels, rims, seams and panel lines, buttons, screws, holes, feet and hinges, and printed patterns (a small procedural DataTexture or thin surface geometry). Use helper functions and loops so the length goes into detail, not repetition."""

# Leader of round 0 (reports/round0-competitor-analysis.md): how the judge looks at the module, and craft rules.
JUDGING = """How the result is judged: the module is rendered on a white background, and on a mid-gray one, by cameras on the +Z side looking at the origin: 30 degrees to either side of +Z at the object's height, the front from slightly below and the front from about 30 degrees above; then also from the right, the back, the left and straight above. A vision-language model compares each render with the target photo next to a rival model and prefers the one that looks more like the same object: overall shape and proportions, colors and materials, distinctive parts and details, and a complete, plausible object from every side.
Pose: place the object so that the cameras on the +Z side see it the way the target shows it: the same side facing the viewer, the same upright, lying or profile pose and the same left-right direction. Model only the object itself, never the floor, background, shadows or reflections."""

CRAFT = """Craft:
- Decompose the object into its real parts. Get the silhouette and proportions right first, then add the distinctive details (bevels, rims, seams, handles, holes, rivets, text-free markings).
- Use LatheGeometry for round profiles, ExtrudeGeometry (with bevel) for flat shapes with outlines, TubeGeometry for bent rods, cables and handles, and smooth sphere or capsule segments for organic forms; scale spheres non-uniformly for ellipsoids.
- Use small procedural DataTextures for visible patterns (wood grain, fabric, rust, print) when they matter.
- Keep surfaces closed; avoid thin floating planes, z-fighting and parts poking through each other."""

GENERATOR = (
    "You are an expert technical artist who rebuilds objects from a single reference photo as procedural Three.js "
    "models. Recreate the object in the target image as a complete deterministic Three.js ES module. Return only "
    "JavaScript, with no Markdown. Plan geometry, proportions, material colors and silhouette before writing. There "
    "are no tools. Follow the rules below; the module must satisfy the specification after them.\n\n"
    + JUDGING
    + "\n\n"
    + GUIDANCE
    + "\n\n"
    + CRAFT
    + "\n\n"
    + SPECIFICATION
)

DRAFT = "Reference photo attached. Write the complete Three.js module for this object."

REFINE = """Below the target are two sheets of renders of your current module. Compare them with the target and list to yourself the largest visible differences: shape and proportions, pose or orientation, colors and materials, missing or wrong parts, and sides, back or top that look missing or paper-thin. Then fix them and keep what already matches. Ignore differences caused only by the renders' background, lighting or automatic centering and scaling.
Return the complete improved module (only JavaScript), not a patch.

Current module:
```js
{source}
```"""

FRONT_SHEET = (
    "Front sheet: the current module from the four views the judge scores, on white, in a 2x2 grid: top-left "
    "slightly left of the front (camera turned 30 degrees towards +X), top-right slightly right of the front "
    "(30 degrees towards -X), bottom-left the front from slightly below, bottom-right the front from slightly above."
)

SIDE_SHEET = (
    "Side sheet: the same module on mid-gray, in a 2x2 grid: top-left camera on +X, top-right camera behind (-Z), "
    "bottom-left camera on -X, bottom-right camera directly above. The judge rejects a model whose sides, back or "
    "top look missing, paper-thin or unrelated to its front."
)

REPAIR = (
    "Repair this rejected module to satisfy the specification while preserving its reconstruction "
    "of the target. Fix every reported error. Return a complete module, not a patch.\n"
    "Errors:\n{errors}\nRejected source:\n{source}"
)

COMPACT = (
    "Your previous reply ran out of output tokens before the module was complete. Write a more compact "
    "module: reuse helpers and loops, avoid long literal arrays, and keep comments minimal."
)

# Stage S1 of the official judge (judge-service multi_stage.py), with each model shown as its front sheet.
JUDGE_SYSTEM = (
    "You are a specialized 3D model evaluation system.\n"
    "Analyze visual quality and prompt adherence with expert precision.\n"
    "Always respond with valid JSON only."
)

JUDGE_USER = (
    "You see two 3D models rendered from four angles around the front; each model is a 2x2 grid of the same object "
    "(30 degrees to either side, slightly below, above).\n"
    "The reference image shows the target object.\n\n"
    "Which model is a more faithful 3D reproduction of the reference?\n\n"
    "Penalty 0-10:\n"
    "0 = Perfect match to reference\n"
    "3 = Minor issues (slight shape differences, missing small details)\n"
    "5 = Moderate issues (wrong style, significant details missing)\n"
    "7 = Major issues (wrong category but related, e.g. chair vs stool)\n"
    "10 = Completely wrong object\n\n"
    'Output: {"penalty_1": <0-10>, "penalty_2": <0-10>, "issues": "<brief>"}'
)

JUDGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["penalty_1", "penalty_2", "issues"],
    "properties": {
        "penalty_1": {"type": "integer"},
        "penalty_2": {"type": "integer"},
        "issues": {"type": "string"},
    },
}
