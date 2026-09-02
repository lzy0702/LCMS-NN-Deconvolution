"""Elemental formula arithmetic (integer or fractional counts, negative deltas allowed)."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping

from .elements import ISOTOPES, average_mass, monoisotopic_mass

_TOKEN = re.compile(r"([A-Z][a-z]?)(-?\d*\.?\d*)")


class Formula:
    """An elemental composition.

    Counts may be fractional (average compositions such as averagine) or negative (mass
    deltas such as Na-H). Immutable; arithmetic returns new instances.
    """

    __slots__ = ("_counts",)

    def __init__(self, counts: str | Mapping[str, float] | None = None):
        if counts is None:
            self._counts: dict[str, float] = {}
        elif isinstance(counts, str):
            self._counts = dict(Formula.parse(counts)._counts)
        else:
            self._counts = {k: float(v) for k, v in counts.items() if float(v) != 0.0}
        for el in self._counts:
            if el not in ISOTOPES:
                raise ValueError(f"Unknown element symbol: {el!r}")

    # ------------------------------------------------------------------ parsing
    @classmethod
    def parse(cls, text: str) -> Formula:
        text = text.strip().replace(" ", "")
        if not text:
            return cls()
        pos = 0
        counts: dict[str, float] = {}
        while pos < len(text):
            m = _TOKEN.match(text, pos)
            if not m or m.end() == pos:
                raise ValueError(f"Cannot parse formula {text!r} at position {pos}")
            el, num = m.group(1), m.group(2)
            if el not in ISOTOPES:
                raise ValueError(f"Unknown element symbol {el!r} in {text!r}")
            n = float(num) if num not in ("", "-") else (1.0 if num == "" else -1.0)
            counts[el] = counts.get(el, 0.0) + n
            pos = m.end()
        return cls(counts)

    # ------------------------------------------------------------------ accessors
    @property
    def counts(self) -> dict[str, float]:
        return dict(self._counts)

    def __getitem__(self, el: str) -> float:
        return self._counts.get(el, 0.0)

    def __iter__(self) -> Iterator[str]:
        return iter(self._counts)

    def __len__(self) -> int:
        return len(self._counts)

    def items(self):
        return self._counts.items()

    def __contains__(self, el: str) -> bool:
        return el in self._counts

    # ------------------------------------------------------------------ arithmetic
    def __add__(self, other: Formula) -> Formula:
        c = dict(self._counts)
        for k, v in other._counts.items():
            c[k] = c.get(k, 0.0) + v
        return Formula(c)

    def __sub__(self, other: Formula) -> Formula:
        c = dict(self._counts)
        for k, v in other._counts.items():
            c[k] = c.get(k, 0.0) - v
        return Formula(c)

    def __mul__(self, n: float) -> Formula:
        return Formula({k: v * n for k, v in self._counts.items()})

    __rmul__ = __mul__

    def __neg__(self) -> Formula:
        return self * -1.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Formula):
            return NotImplemented
        keys = set(self._counts) | set(other._counts)
        return all(abs(self[k] - other[k]) < 1e-9 for k in keys)

    def __hash__(self) -> int:
        return hash(tuple(sorted((k, round(v, 6)) for k, v in self._counts.items())))

    # ------------------------------------------------------------------ masses
    @property
    def mono_mass(self) -> float:
        return sum(n * monoisotopic_mass(el) for el, n in self._counts.items())

    @property
    def avg_mass(self) -> float:
        return sum(n * average_mass(el) for el, n in self._counts.items())

    @property
    def nominal_mass(self) -> int:
        return int(round(self.mono_mass))

    @property
    def is_integer(self) -> bool:
        return all(abs(v - round(v)) < 1e-9 for v in self._counts.values())

    def rounded(self) -> Formula:
        """Round every count to the nearest integer (used before isotope expansion)."""
        return Formula({k: float(round(v)) for k, v in self._counts.items()})

    def scaled_to_mass(self, target_mass: float, adjust: str = "H", average: bool = True) -> Formula:
        """Scale the composition so that its (average or mono) mass equals ``target_mass``.

        Counts are rounded to integers and the ``adjust`` element count is corrected so the
        rounded formula reproduces the target mass as closely as possible. This is the
        classical averagine construction.
        """
        m = self.avg_mass if average else self.mono_mass
        if m <= 0:
            raise ValueError("Cannot scale a formula with non-positive mass")
        f = (self * (target_mass / m)).rounded()
        el_mass = average_mass(adjust) if average else monoisotopic_mass(adjust)
        current = f.avg_mass if average else f.mono_mass
        delta = int(round((target_mass - current) / el_mass))
        counts = f.counts
        counts[adjust] = max(0.0, counts.get(adjust, 0.0) + delta)
        return Formula(counts)

    # ------------------------------------------------------------------ text
    def hill(self) -> str:
        """Hill-order string. Fractional counts are printed with 4 decimals."""
        parts = []
        keys = sorted(self._counts)
        ordered = []
        if "C" in self._counts:
            ordered.append("C")
            if "H" in self._counts:
                ordered.append("H")
            ordered += [k for k in keys if k not in ("C", "H")]
        else:
            ordered = keys
        for k in ordered:
            v = self._counts[k]
            if abs(v - round(v)) < 1e-9:
                iv = int(round(v))
                parts.append(k if iv == 1 else f"{k}{iv}")
            else:
                parts.append(f"{k}{v:.4f}")
        return "".join(parts)

    def __str__(self) -> str:
        return self.hill()

    def __repr__(self) -> str:
        return f"Formula({self.hill()!r})"
