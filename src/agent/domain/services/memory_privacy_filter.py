"""Heuristics that keep secrets out of the long-term memory vault.

The system prompt already tells the agent not to store secrets or throwaway
remarks. This filter is the server-side backstop behind the ``remember`` tool. A
rejected write carries a machine-readable reason, so the model can tell the user
why nothing was stored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MemoryPrivacyVerdict:
    """Outcome of the privacy check. ``reason`` is set only on rejection."""

    allowed: bool
    reason: Optional[str] = None


# A fact containing any of these (compared in lower case) is treated as a secret.
_SECRET_KEYWORDS = (
    "password",
    "passcode",
    "pin code",
    "pin:",
    "cvv",
    "cvc",
    "one-time code",
    "verification code",
    "security question",
    "secret word",
    "private key",
    "secret key",
    "api key",
    "access token",
    "seed phrase",
    "recovery phrase",
)

# A payment-card number: 13 to 19 digits, written as one run or in groups
# separated by single spaces or dashes. Every repetition consumes one digit, so
# a match always holds at least 13 digits; shorter runs such as amounts and years
# never match.
_CARD_NUMBER_RE = re.compile(r"(?:\d[ -]?){13,19}")

_MIN_MEANINGFUL_LENGTH = 3


class MemoryPrivacyFilter:
    """Stateless gate for vault writes. No I/O, no randomness."""

    @staticmethod
    def check(content: str) -> MemoryPrivacyVerdict:
        """Decide whether ``content`` may be stored in the vault."""
        text = (content or "").strip()
        if len(text) < _MIN_MEANINGFUL_LENGTH:
            return MemoryPrivacyVerdict(allowed=False, reason="content_too_short")

        lowered = text.lower()
        if any(keyword in lowered for keyword in _SECRET_KEYWORDS):
            return MemoryPrivacyVerdict(allowed=False, reason="looks_like_secret")

        if _CARD_NUMBER_RE.search(text):
            return MemoryPrivacyVerdict(allowed=False, reason="looks_like_card_number")

        return MemoryPrivacyVerdict(allowed=True)
