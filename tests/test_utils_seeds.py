"""Tests for reproducible seeding. Written before the implementation (TDD)."""

import random

from equibirth.utils.seeds import set_seeds


def test_set_seeds_makes_python_random_deterministic():
    set_seeds(42)
    first = [random.random() for _ in range(5)]
    set_seeds(42)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_set_seeds_returns_the_seed():
    assert set_seeds(123) == 123
