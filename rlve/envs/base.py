"""Base abstractions for verifiable environments.

A verifiable environment is the tuple E = (I, P, R) from the RLVE paper:
  - I : an input/instruction template (we keep this shared across envs, see
        ``rlve/data.py``; each env only emits the problem-specific question).
  - P : a *problem generator* ``generate(difficulty, rng)`` that procedurally
        samples a concrete problem at a given integer difficulty level d >= 0.
  - R : a *verifier* ``verify(completion, problem)`` that algorithmically scores
        a model completion, returning a reward and a binary correctness flag.

Every environment also exposes ``min_difficulty`` / ``max_difficulty`` so the
difficulty controllers know the valid range of d.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Optional

from rlve.verify import extract_answer


@dataclass
class Problem:
    """A single procedurally generated problem instance."""

    question: str            # problem-specific statement (the user turn)
    answer: str              # canonical gold answer string
    difficulty: int          # difficulty level d it was generated at
    env_name: str
    info: Dict = field(default_factory=dict)


@dataclass
class VerifyResult:
    reward: float            # continuous reward used for RL (here in [0, 1])
    correct: bool            # binary correctness used for difficulty stats
    extracted: Optional[str] # what we parsed out of the completion (for debugging)
    format_ok: bool          # whether the answer was in the requested format


class Environment:
    """Abstract base class for a verifiable environment."""

    name: str = "base"
    min_difficulty: int = 0
    max_difficulty: int = 10

    #: One-line description of the expected answer format, shown to the model.
    answer_spec: str = "your final answer"

    def generate(self, difficulty: int, rng: random.Random) -> Problem:
        raise NotImplementedError

    # --- verification -----------------------------------------------------
    def verify(self, completion: str, problem: Problem) -> VerifyResult:
        """Default verifier: extract the answer then defer to ``_is_correct``.

        Returns a binary-correctness reward in {0, 1} plus a small format bonus
        so that even wrong-but-well-formatted answers are distinguishable. The
        difficulty controllers only look at ``correct``.
        """
        extracted = extract_answer(completion)
        format_ok = extracted is not None
        correct = False
        if extracted is not None:
            try:
                correct = self._is_correct(extracted, problem)
            except Exception:
                correct = False
        reward = 1.0 if correct else (0.1 if format_ok else 0.0)
        return VerifyResult(reward=reward, correct=correct,
                            extracted=extracted, format_ok=format_ok)

    def _is_correct(self, extracted: str, problem: Problem) -> bool:
        """Compare a parsed answer string to the gold answer. Override per env."""
        return _norm(extracted) == _norm(problem.answer)

    # --- helpers ----------------------------------------------------------
    def clamp(self, d: int) -> int:
        return max(self.min_difficulty, min(self.max_difficulty, int(d)))


def _norm(s: str) -> str:
    return "".join(s.split()).strip().lower()


def parse_int(s: str) -> Optional[int]:
    """Best-effort parse of a (possibly comma-formatted, signed) integer."""
    s = s.strip().replace(",", "").replace(" ", "")
    if s.startswith("+"):
        s = s[1:]
    try:
        return int(s)
    except ValueError:
        # tolerate a trailing period or stray text containing one integer
        import re
        m = re.search(r"-?\d+", s)
        return int(m.group()) if m else None
