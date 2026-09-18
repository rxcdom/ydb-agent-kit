"""Tests for the heuristics that guard writes to the memory vault."""
from __future__ import annotations

import pytest

from src.agent.domain.services.memory_privacy_filter import (
    MemoryPrivacyFilter,
    MemoryPrivacyVerdict,
)


@pytest.mark.parametrize(
    "content",
    [
        "Saving for a car, the goal is 15000 by December",
        "Works in IT and plans the week on Monday mornings",
        "Prefers a blunt tone without small talk",
        "Will close the loan in March 2027",  # a loan is not a secret
    ],
)
def test_allows_ordinary_durable_facts(content):
    assert MemoryPrivacyFilter.check(content) == MemoryPrivacyVerdict(allowed=True)


@pytest.mark.parametrize(
    "content",
    [
        "the password for the tax portal is qwerty",
        "my PIN code is 4321, keep the pin code",
        "card pin: 4321",
        "the one-time code from the text message is 123456",
        "verification code 778899",
        "the secret word at the bank is violet",
        "security question answer: first pet",
        "Passcode for the broker app is hunter2",
        "api key abc123 for the staging service",
        "access token for the build server",
        "the private key lives in the top drawer",
        "secret key of the signing service",
        "cvv 123",
        "CVC is 456",
        "seed phrase of the wallet: apple banana",
        "recovery phrase for the vault",
    ],
)
def test_rejects_secret_keywords(content):
    verdict = MemoryPrivacyFilter.check(content)
    assert verdict.allowed is False
    assert verdict.reason == "looks_like_secret"


@pytest.mark.parametrize(
    "content",
    [
        "card 1234567890123456",
        "remember 1234 5678 9012 3456",
        "4276-3800-1234-5678 is mine",
    ],
)
def test_rejects_card_number_shapes(content):
    verdict = MemoryPrivacyFilter.check(content)
    assert verdict.allowed is False
    assert verdict.reason == "looks_like_card_number"


@pytest.mark.parametrize("content", ["", "  ", "ok", None])
def test_rejects_empty_and_too_short(content):
    verdict = MemoryPrivacyFilter.check(content)
    assert verdict.allowed is False
    assert verdict.reason == "content_too_short"


def test_short_digit_runs_are_not_card_numbers():
    # Amounts and years must not trip the card heuristic.
    assert MemoryPrivacyFilter.check("The goal is 1500000 saved by the year 2027").allowed is True


def test_allowed_verdict_carries_no_reason():
    assert MemoryPrivacyFilter.check("Likes to review open tasks on Fridays").reason is None
