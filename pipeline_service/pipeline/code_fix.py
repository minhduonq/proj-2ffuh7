"""Deterministic, rule-per-error repair of candidates that FAILED the BTC checker.

Each rule fires only on the exact failure it targets (the checker reports one failure at a time, so the caller
re-checks after every pass). Measured 13/10 on 2,038 AmberKite candidates (lab/exp_13_10): 162 failed the
checker; these rules rescued 41 (25%) with no GPU cost, and the rescued candidates scored like an average
candidate of their prompt (not worse). Rules:

- duplicate declaration  (PARSE_ERROR "Identifier 'x' has already been declared (L:C)")
      -> rename the second declaration and every later use of x (not after '.') to x__2
- THREE outside generate (THREE_AT_TOP_LEVEL; helpers written AFTER `generate`)
      -> move all top-level code into the start of generate's body, where THREE is in scope
- broken hex colour      ("Identifier directly after number" / "Unexpected character" on a 0x literal)
      -> keep the hex digits, pad to 6
- const without value    ("Missing initializer in const declaration (L:C)") -> let on that line
- assignment to const    (EXECUTION_THREW "Assignment to constant variable") -> every const -> let
- misspelt identifier    (IDENTIFIER_NOT_ALLOWED "x at line L") -> closest declared name (difflib >= 0.8)

Pure stdlib; never raises (the caller treats any exception as "no fix").
"""
from __future__ import annotations

import difflib
import re

_DECL = re.compile(r"\b(?:const|let|var|function|class)\s+([A-Za-z_$][\w$]*)")
_GENERATE = re.compile(r"export\s+default\s+function\s+generate\s*\(\s*THREE\s*\)\s*\{")

FIXABLE_RULES = ("PARSE_ERROR", "THREE_AT_TOP_LEVEL", "IDENTIFIER_NOT_ALLOWED", "EXECUTION_THREW")


def parse_errors(js_errors: list[str]) -> list[dict]:
    """The worker's checker stores failures as 'RULE: detail' strings; split them back."""
    out = []
    for e in js_errors or []:
        rule, _, detail = str(e).partition(": ")
        out.append({"rule": rule.strip(), "detail": detail})
    return out


def _rename_after(lines: list[str], line_no: int, old: str, new: str) -> list[str]:
    pat = re.compile(r"(?<![\w$.])" + re.escape(old) + r"(?![\w$])")
    for i in range(line_no - 1, len(lines)):
        lines[i] = pat.sub(new, lines[i])
    return lines


def _match_brace(src: str, j: int) -> int:
    """Index of the '}' matching the '{' at j, skipping strings, templates and comments; -1 if unbalanced."""
    depth, k, n, mode = 0, j, len(src), None
    while k < n:
        c = src[k]
        if mode is None:
            if src.startswith("//", k):
                k = src.find("\n", k)
                k = n if k < 0 else k
                continue
            if src.startswith("/*", k):
                k = src.find("*/", k + 2)
                k = n if k < 0 else k + 2
                continue
            if c in "\"'`":
                mode = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return k
        else:
            if c == "\\":
                k += 2
                continue
            if c == mode:
                mode = None
        k += 1
    return -1


def move_top_level_into_generate(src: str) -> str:
    m = _GENERATE.search(src)
    if not m:
        return src
    open_at = m.end() - 1
    close_at = _match_brace(src, open_at)
    if close_at < 0:
        return src
    head, tail = src[: m.start()], src[close_at + 1:]
    if not (head.strip() or tail.strip()):
        return src
    return (src[m.start(): open_at + 1] + "\n" + head.strip("\n") + "\n" + tail.strip("\n") + "\n"
            + src[open_at + 1: close_at + 1] + "\n")


def _hexfix(mm: re.Match) -> str:
    digits = re.sub(r"[^0-9a-fA-F]", "", mm.group(1))[:6] or "0"
    return "0x" + digits.ljust(6, digits[-1])


def fix_once(src: str, failures: list[dict]) -> tuple[str, list[str]]:
    """Apply every rule matching `failures` once. Returns (new_source, rules_used)."""
    lines, used = src.split("\n"), []
    if any(f.get("rule") == "THREE_AT_TOP_LEVEL" for f in failures):
        moved = move_top_level_into_generate(src)
        if moved != src:
            lines = moved.split("\n")
            used.append("three_outside_generate")
    for f in failures:
        rule, det = f.get("rule", ""), f.get("detail", "") or ""
        m = re.search(r"Identifier '([A-Za-z_$][\w$]*)' has already been declared\.? \((\d+):", det)
        if m:
            old, line_no = m.group(1), int(m.group(2))
            if 1 <= line_no <= len(lines):
                lines = _rename_after(lines, line_no, old, old + "__2")
                used.append("duplicate_declaration")
            continue
        m = re.search(r"(?:Identifier directly after number|Unexpected character).*\((\d+):", det)
        if m and 1 <= int(m.group(1)) <= len(lines) and "0x" in lines[int(m.group(1)) - 1]:
            line_no = int(m.group(1))
            lines[line_no - 1] = re.sub(
                r"0x([0-9A-Za-z\u0080-￿]*[^\x00-\x7f0-9A-Za-z\s,;)}\]]*)", _hexfix, lines[line_no - 1])
            used.append("hex_colour")
            continue
        m = re.search(r"Missing initializer in const declaration\.? \((\d+):", det)
        if m:
            line_no = int(m.group(1))
            if 1 <= line_no <= len(lines):
                lines[line_no - 1] = re.sub(r"\bconst\b", "let", lines[line_no - 1], count=1)
                used.append("const_without_value")
            continue
        if "Assignment to constant variable" in det:
            lines = [re.sub(r"\bconst\b", "let", ln) for ln in lines]
            used.append("assignment_to_const")
            continue
        m = re.search(r"^([A-Za-z_$][\w$]*) at line (\d+)", det) if rule == "IDENTIFIER_NOT_ALLOWED" else None
        if m:
            bad = m.group(1)
            names = set(_DECL.findall("\n".join(lines)))
            close = difflib.get_close_matches(bad, list(names), n=1, cutoff=0.8)
            if close and close[0] != bad:
                pat = re.compile(r"(?<![\w$.])" + re.escape(bad) + r"(?![\w$])")
                lines = [pat.sub(close[0], ln) for ln in lines]
                used.append("misspelt_identifier")
    new = "\n".join(lines)
    return (new, used) if new != src else (src, [])
