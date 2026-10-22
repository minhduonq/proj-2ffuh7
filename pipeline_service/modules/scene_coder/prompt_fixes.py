"""Individually toggleable reference-fidelity prompt fixes.

Each fix targets one measured failure mode from the 128-sample baseline duel
analysis. They are separated so a run can enable exactly one and the effect on
that fix's own compliance metric can be attributed to it alone
(`tests/prompt_audit.py` measures each metric).

Select with the `PROMPT_FIXES` env var, comma-separated, e.g.

    PROMPT_FIXES=placement                 # verified baseline
    PROMPT_FIXES=placement,segments        # + one candidate under test
    PROMPT_FIXES=none                      # original prompt, no fixes

Default is `none`: no experimental rule survived duel evaluation, so the
shipped prompt is the original plus the baked-in metalness correction.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Rule blocks — inserted into CODER_SYSTEM_PROMPT between the material
# quick-reference and "Modeling strategy:". Numbering is applied at compose
# time so any subset reads correctly.
# --------------------------------------------------------------------------

_RULES: dict[str, str] = {}
_REMINDERS: dict[str, str] = {}

_RULES["color"] = """\
COLOR — the single biggest source of losses.
   - Declare every colour once, at the top of `generate`, as a named hex
     const (`const BODY = 0xc9a227;`), then reference those consts in
     materials. One const per colour region, reused by every part sharing it.
     Colours buried inline are the ones that end up inconsistent.
   - Read the colour off the REFERENCE, from the part's mid-tone — the large
     evenly-lit area of that part. Never sample a specular highlight (too
     light) or a shadow/contact edge (too dark). The renderer adds its own
     lighting; if you encode the highlight, the render comes out washed out.
   - Do not stylize, saturate, or "clean up" a colour. Cream is not white
     (`#f2ece0`, not `#ffffff`); near-black is not black (`#1c1c1c`, not
     `#000000`); a muted sage is not a vivid green. Matching a dull colour
     dully is a win; making it prettier is a loss.
   - CLEAR IS NOT A COLOUR. Clear glass, clear gemstones, clear acrylic,
     water, and ice take `color: 0xffffff` with `transmission`/`ior` doing
     the work. Tinting a colourless object — a blue gemstone where the
     reference is clear — is a guaranteed loss. Add a tint only when the
     reference visibly shows one.
   - The metal hex list in the material quick-reference is a fallback for
     NEUTRAL metals. If the reference metal is warm or coloured — brass,
     gold, copper, rose, blued steel, black anodized — tint `color` to that
     hue instead. Defaulting a brass fitting to `#d4d4d4` steel is a
     wrong-colour loss, not a safe choice.
   - Keep the palette tight. If the reference reads as three colours, use
     three. Inventing a fourth accent colour loses."""

_REMINDERS["color"] = """\
- Declare every colour as a named hex const at the top of `generate`, read
  from the reference's mid-tone for that part. Don't brighten, saturate, or
  tidy a colour, and don't tint anything the reference shows as clear."""

_RULES["placement"] = """\
PLACEMENT — details that exist but sit in the wrong place.
   - Derive positions, never type them twice. Declare the object's
     dimensions as consts first (`const bodyW = 0.4, bodyH = 0.6;`), then
     compute every child position from those consts. Hand-typed absolute
     offsets drift out of alignment as parts change.
   - Every part must physically meet what it belongs to. A handle touches
     the body at BOTH ends, a knob sits ON the face, a lid sits ON the rim.
     Check the reference for WHICH end, edge, or face a part attaches to —
     a correct handle on the wrong side is still a placement loss.
   - Repeated parts (keys, pearls, beads, slats, studs, dial markers) must
     be placed by index arithmetic from one origin and one pitch —
     `x = x0 + i * pitch`, or `angle = i / N * 2 * Math.PI` on a ring — so
     spacing is perfectly uniform. Placing them by a list of typed
     coordinates produces the misaligned rows the judge singles out.
   - Mirror-symmetric parts must use the SAME offset with opposite sign
     (`+armX` / `-armX`), never two separately typed numbers.
   - Row/ring counts must match the reference. Count them in the image
     before writing the loop bound."""

_REMINDERS["placement"] = """\
- Declare the object's dimensions as consts first, then compute every child
  position from them. Repeated parts (keys, beads, slats, markers) get placed
  by index arithmetic from one origin and one pitch, never by typed offsets."""

# Split out of the new prompt's rule 3: this half is a pure numeric floor with
# no perception component, so it is measured and attributed on its own.
_RULES["segments"] = """\
CURVED SURFACES MUST STAY SMOOTH.
   - NEVER let a smooth surface come out polygonal. Curved surfaces default
     to `radialSegments` 32 (cylinder/cone/circle), sphere 32x16, lathe
     segments 24+, tube radialSegments 12+, torus 16x32. Drop below those
     ONLY when the reference object is genuinely faceted (a cut gemstone, a
     hex nut, a low-poly design). Visible flat facets on something the
     reference shows as round is a proportion/shape loss."""

_REMINDERS["segments"] = """\
- Keep curved surfaces smooth: radialSegments 32, sphere 32x16, lathe 24+,
  torus 16x32, unless the reference object is genuinely faceted."""

# Rule 3 of the new prompt minus the segments half.
_RULES["proportion"] = """\
PROPORTION AND RESTRAINT — do not overdo detail.
   - Fix the overall silhouette first: read the object's width:height:depth
     ratio off the reference, declare it as consts, and size every part as a
     fraction of those. A correct silhouette with plain parts beats a
     detailed model with the wrong outline.
   - Fine detail stays fine. Stitching, piping, seams, trim, engraved lines,
     and beading read in the reference as thin lines: model them at 1-3% of
     the object's largest dimension. Scaling them up so they are "visible"
     is the exaggeration the judge penalises — if it is a hairline in the
     reference, it is a hairline in your model.
   - Secondary parts scale from the body, not from convenience. Handles,
     spouts, knobs, feet, and lids that look oversized against the body lose
     even when their shape is right.
   - Prefer fewer, better parts. Every part you add is another chance to be
     wrong about its colour, size, and position."""

_REMINDERS["proportion"] = """\
- Keep fine detail fine: stitching, piping, seams and trim at 1-3% of the
  largest dimension. Size secondary parts as a fraction of the body."""

_RULES["noinvent"] = """\
ONLY MODEL WHAT YOU CAN SEE — no invention.
   - Model the parts actually visible in the reference. If you are unsure
     whether something is there, leave it out: an omitted part costs less
     than an invented one, which the judge reports as an outright mismatch.
   - Do not add decorative extras the reference does not show — no finials,
     caps, buttons, spheres on handles, feet, rivets, or trim by default.
   - Exactly ONE object. Never add a second object, a stand, a base plate,
     a ground plane, or a prop next to it unless the reference clearly shows
     it as part of the same object.
   - For faces hidden from the camera, continue the visible form plainly.
     Do not invent new features on the unseen side."""

_REMINDERS["noinvent"] = """\
- Model only what you can see — one object, no invented finials, feet,
  rivets, props, or bases. When unsure a part exists, leave it out."""

_RULES["lettering"] = """\
LETTERING AND MARKINGS — write it right or write it abstract.
   - If the text in the reference is legible, transcribe it EXACTLY:
     character by character, same spelling, same case, same character count.
     Re-read the reference glyph by glyph before you write the string. A
     misspelling (`RHENNLDD` for `RHINNIOD`) reads to the judge as plainly
     wrong text and loses the duel outright.
   - If the text is NOT legible — too small, blurred, angled away — do NOT
     guess a word. Represent it as abstract marks: thin bars, dots, or a
     shallow plate at the right position, size, and colour. Unreadable marks
     in the right place beat an invented word.
   - Never mirror text. Shape/Extrude glyphs built in XY face +Z; if the
     lettering belongs on a face pointing -Z or -X, rotate the group so the
     glyphs still read left-to-right from outside the object. Reversed or
     upside-down text is a common and very visible loss.
   - Dial and gauge markings are counted, not eyeballed: a watch face has 12
     hour markers, a clock 12, a compass 4 cardinals. Count the markers and
     numerals in the reference, then place them with index arithmetic.
     Roman numerals must be the correct numerals in the correct order
     (I, II, III, IIII/IV, V, VI, VII, VIII, IX, X, XI, XII)."""

_REMINDERS["lettering"] = """\
- Text: transcribe legible lettering exactly, character by character; if it
  isn't legible, use abstract marks instead of a guessed word. Never mirror
  glyphs. Count dial markers and numerals against the reference."""

# Canonical order the rules are numbered in.
_ORDER = ["color", "placement", "segments", "proportion", "noinvent", "lettering"]

# Fixes applied as text substitutions rather than added rules.
# `metalness` is not a fix: binary metalness is baked into prompts.py, because
# the renderer lights with a real PMREM environment (see
# modules/renderer/render_service/object-page.html and competition doc 05).
_TRANSFORM_KEYS = ["fitcube", "segments_table"]

ALL_FIXES = _ORDER + _TRANSFORM_KEYS

_INTRO = """\
Reference fidelity rules (these decide the duel — read before modeling):

You are judged by a vision model comparing your render against the reference
image, side by side. It does not read your code and gives no credit for
effort. A part that is present but wrong loses to a part that is simple and
right.
"""


def _selected() -> list[str]:
    raw = os.environ.get("PROMPT_FIXES", "none").strip()
    if raw.lower() in ("", "none", "off"):
        return []
    if raw.lower() == "all":
        return list(ALL_FIXES)
    keys = [k.strip().lower() for k in raw.split(",") if k.strip()]
    unknown = [k for k in keys if k not in ALL_FIXES]
    if unknown:
        raise ValueError(
            f"PROMPT_FIXES has unknown fix(es) {unknown}; known: {ALL_FIXES}"
        )
    return keys


ACTIVE = _selected()


def fidelity_section() -> str:
    """The numbered rule block, or "" when no rule-type fix is active."""
    active = [k for k in _ORDER if k in ACTIVE]
    if not active:
        return ""
    body = "\n\n".join(
        f"{i}. {_RULES[k]}" for i, k in enumerate(active, start=1)
    )
    return "\n" + _INTRO + "\n" + body + "\n"


def reminder_bullets() -> str:
    """Per-fix bullets appended under "Reminders before you write:"."""
    active = [k for k in _ORDER if k in ACTIVE]
    if not active:
        return ""
    return "\n" + "\n".join(_REMINDERS[k] for k in active)


def apply_user_substitutions(text: str) -> str:
    """Substitutions that belong in the user templates, not the system prompt."""
    if "segments_table" in ACTIVE and _SEG_REMINDER_OLD in text:
        text = text.replace(_SEG_REMINDER_OLD, _SEG_REMINDER_NEW, 1)
    return text


# --------------------------------------------------------------------------
# Substitution fixes
# --------------------------------------------------------------------------

_FITCUBE_OLD = """\
6. Always include a fit-to-unit-cube normalization helper and call it
   before return. The helper MUST scale to `0.95 / maxDim` (not `1/maxDim`)
   so the object fills ~95% of the unit cube — smaller values leave the
   render mostly empty background."""

_FITCUBE_NEW = """\
6. Always end `generate` by normalizing the object. Copy this helper
   VERBATIM and call it immediately before `return` — do not rewrite it:

   ```javascript
   function fitToUnitCube(THREE, root) {
     const box = new THREE.Box3().setFromObject(root);
     const size = new THREE.Vector3();
     const center = new THREE.Vector3();
     box.getSize(size);
     box.getCenter(center);
     const maxDim = Math.max(size.x, size.y, size.z) || 1;
     const scale = 0.95 / maxDim;
     root.scale.setScalar(scale);
     // Recentering is NOT optional. Post-validation rejects any module whose
     // bounding box leaves [-0.5, 0.5] on ANY axis, and scaling alone leaves
     // the object wherever it happened to be built — BOUNDING_BOX_OUT_OF_RANGE
     // is the single most common failure there.
     root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
   }
   ```

   Two rules around the call:
   - Every mesh must already be added to the group BEFORE you call it;
     anything added afterwards escapes the normalization.
   - `0.95 / maxDim`, never `1 / maxDim` — the object should fill ~95% of the
     unit cube; smaller values leave the render mostly empty background."""


# `segments` as prose produced no measurable change (cylinder<32 stayed inside
# the control band). `metalness` — the same kind of numeric rule, but expressed
# as a row in the material lookup table — flipped 0% -> 93% compliance. This
# fix restates the identical segment floors in that proven table format to test
# whether the format, not the content, is what the model acts on.
_SEG_TABLE_ANCHOR = (
    "  generic / unsure            MeshStandardMaterial  metalness 0.0 roughness 0.7"
)

_SEG_TABLE = _SEG_TABLE_ANCHOR + """

Geometry segment quick-reference (apply to every curved primitive — pick exact
segment counts, don't improvise):

  CylinderGeometry            radialSegments 32
  ConeGeometry                radialSegments 32
  CircleGeometry              segments 32
  SphereGeometry              widthSegments 32   heightSegments 16
  TorusGeometry               radialSegments 16  tubularSegments 32
  LatheGeometry               segments 24
  TubeGeometry                tubularSegments 64 radialSegments 12
  RingGeometry                thetaSegments 32
  cut gem / hex nut / low-poly  use the reference's real facet count"""

_SEG_REMINDER_OLD = (
    "- Use the material normalization quick-reference from your system prompt\n"
    "  \u2014 don't improvise PBR values."
)
_SEG_REMINDER_NEW = (
    "- Use the material normalization quick-reference from your system prompt\n"
    "  \u2014 don't improvise PBR values.\n"
    "- Use the geometry segment quick-reference from your system prompt\n"
    "  \u2014 don't improvise segment counts."
)

_SUBS = {
    "fitcube": [(_FITCUBE_OLD, _FITCUBE_NEW)],
    "segments_table": [(_SEG_TABLE_ANCHOR, _SEG_TABLE)],
}


def apply_substitutions(text: str) -> str:
    """Apply the active substitution fixes to an assembled prompt string."""
    for key in _TRANSFORM_KEYS:
        if key not in ACTIVE:
            continue
        for old, new in _SUBS[key]:
            if old not in text:
                raise ValueError(
                    f"PROMPT_FIXES={key}: anchor text not found; prompts.py "
                    "changed and prompt_fixes.py needs updating"
                )
            text = text.replace(old, new)
    return text


# The `wrong_color` repair-playbook upgrade rides with the `color` fix, so the
# colour-const rule is enforced in repair rounds too and not undone there.
_REPAIR_COLOR_OLD = """\
- `wrong_color`        → change material `color:` to the hex from the
  description."""

_REPAIR_COLOR_NEW = """\
- `wrong_color`        → change material `color:` to the hex from the
  description. Update the named colour const if the part uses one, so every
  part sharing that region moves together. If the description says the part
  should be clear/colourless, set `color` to `0xffffff` and let
  transmission/ior carry it rather than tinting it."""


def apply_repair_substitutions(text: str) -> str:
    if "color" in ACTIVE and _REPAIR_COLOR_OLD in text:
        text = text.replace(_REPAIR_COLOR_OLD, _REPAIR_COLOR_NEW)
    return text


def describe() -> str:
    return ",".join(ACTIVE) if ACTIVE else "none"
