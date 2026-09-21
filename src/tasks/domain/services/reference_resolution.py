"""Resolution of a human text reference against a set of owned rows.

People address projects and tasks by what they are called, and names collide.
A reference therefore resolves to exactly one of three results, and the caller
has to deal with each: one row, several candidates, or nothing. No result ever
picks one of several candidates on the caller's behalf.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

T = TypeVar("T")


@dataclass(frozen=True)
class Resolved(Generic[T]):
    """The reference points at exactly one row."""

    match: T


@dataclass(frozen=True)
class Ambiguous(Generic[T]):
    """The reference fits several rows equally well."""

    candidates: Tuple[T, ...]


@dataclass(frozen=True)
class Unmatched:
    """The reference fits no row."""


Resolution = Union[Resolved[T], Ambiguous[T], Unmatched]


def normalise_reference(text: Optional[str]) -> str:
    """Trimmed, case-folded form used on both sides of every comparison."""
    return (text or "").strip().casefold()


def resolve_reference(
    rows: Iterable[T],
    reference: Optional[str],
    *,
    primary_text: Callable[[T], str],
    searchable_texts: Callable[[T], Sequence[Optional[str]]],
) -> Resolution[T]:
    """Match ``reference`` against ``rows`` in two tiers.

    Tier one is a case-insensitive exact match on the primary text. Only when
    it finds nothing does tier two run: a case-insensitive substring match over
    every searchable text. The first tier that finds anything decides; more
    than one row in that tier is ambiguity, not a reason to try harder.
    """
    needle = normalise_reference(reference)
    if not needle:
        return Unmatched()

    candidates: List[T] = list(rows)
    winning_tier = [row for row in candidates if normalise_reference(primary_text(row)) == needle]
    if not winning_tier:
        winning_tier = [
            row
            for row in candidates
            if any(needle in normalise_reference(text) for text in searchable_texts(row))
        ]

    if not winning_tier:
        return Unmatched()
    if len(winning_tier) == 1:
        return Resolved(winning_tier[0])
    return Ambiguous(tuple(winning_tier))
