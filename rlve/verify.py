"""Answer-extraction utilities shared by all verifiers.

Models are instructed to emit their final answer inside ``\\boxed{...}``.
We also accept ``<answer>...</answer>`` tags and, as a last resort, the last
number on the last non-empty line. Extraction is intentionally forgiving on
*format* but the per-environment ``_is_correct`` check is strict on *value*.
"""
from __future__ import annotations

import re
from typing import Optional

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def _extract_boxed(text: str) -> Optional[str]:
    """Return the content of the LAST ``\\boxed{...}`` with balanced braces."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    out = []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
    return None  # unbalanced


def extract_answer(text: str) -> Optional[str]:
    """Pull the model's final answer out of a completion string."""
    if not text:
        return None
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    m = list(_ANSWER_TAG.finditer(text))
    if m:
        return m[-1].group(1).strip()
    # Fallback: last non-empty line, stripped of trailing punctuation.
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line:
            return line.rstrip(".").strip()
    return None
