"""Turn model replies into checkable Three.js modules."""

from __future__ import annotations

import re
from pathlib import Path

NORMALIZATION = Path(__file__).with_name("normalize.js").read_text(encoding="utf-8")
# An object lying flat (height under half its footprint) is tilted 15 degrees so the front cameras see its top.
TILT = (
    "  const flat = new THREE.Box3().setFromObject(pivot).getSize(new THREE.Vector3());\n"
    "  if (flat.y < 0.5 * Math.max(flat.x, flat.z)) {\n"
    "    pivot.rotation.x = 0.261799;\n"
    "    root.updateMatrixWorld(true);\n"
    "  }"
)
MAX_SOURCE_BYTES = 1 << 20
FENCE = re.compile(r"```[A-Za-z]*[ \t]*\n([\s\S]*?)(?:\n```|\Z)")
EXPORT = re.compile(r"^[ \t]*export\s+default\s+", re.MULTILINE)
# Checks that would certainly fail the checker, caught before spending a render.
FORBIDDEN = {
    r"^[ \t]*import\b|\bimport\s*\(": "imports are forbidden",
    r"\bMath\.random\b": "Math.random is forbidden; derive values deterministically",
    r"\bnew\s+Date\b|\bDate\.now\b|\bperformance\.now\b": "time APIs are forbidden",
    r"\b(?:fetch|XMLHttpRequest|WebSocket|TextureLoader|ImageLoader|FileLoader)\b": "network access and loaders are forbidden",
    r"\beval\s*\(|\bnew\s+Function\b": "eval and Function are forbidden",
    r"\basync\s+function\b|\bawait\b": "generate must be synchronous",
}


def extract(reply: str) -> str:
    """Return the JavaScript in a reply: the longest fenced block with the export, or the bare text."""
    text = reply.strip()
    blocks = [block.strip() for block in FENCE.findall(text)]
    exported = [block for block in blocks if EXPORT.search(block)]
    return max(exported or blocks or [text], key=len)


def precheck(source: str) -> list[str]:
    """Static problems the checker would reject; an empty list means the source is worth rendering."""
    errors = []
    if len(source.encode()) > MAX_SOURCE_BYTES:
        errors.append("source exceeds 1 MiB")
    if len(EXPORT.findall(source)) != 1:
        errors.append("exactly one `export default function generate(THREE)` is required")
    for pattern, message in FORBIDDEN.items():
        if re.search(pattern, source, re.MULTILINE):
            errors.append(message)
    return errors


def normalize(source: str, *, tilt: bool = False) -> str:
    """Rename the default export and wrap it so the asset is centred and fitted inside the unit cube.

    Helpers may appear anywhere in the module. Sources without a recognisable export
    are returned unchanged and left for the checker to report. `tilt` adds the flat-object
    tilt, used only for the module that is returned, never for previews.
    """
    exported = EXPORT.search(source)
    if exported is None:
        return source
    rest = source[exported.end() :]
    function = re.match(r"function\s*(?:[A-Za-z_$][\w$]*)?\s*\(", rest)
    if function:
        renamed = "function __buildAsset(" + rest[function.end() :]
    else:
        renamed = "const __buildAsset = " + rest
    wrapper = NORMALIZATION.replace("__TILT__\n", TILT + "\n" if tilt else "")
    return wrapper.replace("__GENERATED_SOURCE__", source[: exported.start()] + renamed)


def placeholder(color: str) -> str:
    """A valid fallback module: a box in the target's dominant colour."""
    return (
        "export default function generate(THREE) {\n"
        "  const geometry = new THREE.BoxGeometry(0.8, 0.8, 0.8);\n"
        f"  const material = new THREE.MeshStandardMaterial({{ color: 0x{color}, roughness: 0.6 }});\n"
        "  const root = new THREE.Group();\n"
        "  root.add(new THREE.Mesh(geometry, material));\n"
        "  return root;\n"
        "}\n"
    )
