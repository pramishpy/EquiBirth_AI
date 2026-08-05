"""Tests for the logging helper. Written before the implementation (TDD)."""

import logging

from equibirth.utils.logging import get_logger


def test_get_logger_returns_logger_with_name():
    log = get_logger("equibirth.test")
    assert isinstance(log, logging.Logger)
    assert log.name == "equibirth.test"


def test_get_logger_is_idempotent_no_duplicate_handlers():
    get_logger("equibirth.dup")
    log = get_logger("equibirth.dup")
    assert len(log.handlers) <= 1
