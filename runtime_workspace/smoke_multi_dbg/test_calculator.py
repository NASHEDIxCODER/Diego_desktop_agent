#!/usr/bin/env python3
"""Tests for the generated calculator."""

from calculator import add, subtract, multiply, divide, evaluate


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 2) == 3


def test_multiply():
    assert multiply(3, 4) == 12


def test_divide():
    assert divide(8, 2) == 4


def test_divide_by_zero():
    try:
        divide(1, 0)
        assert False, "expected ZeroDivisionError"
    except ZeroDivisionError:
        pass


def test_evaluate():
    assert evaluate("2 + 3") == 5
    assert evaluate("10 * 4") == 40
    assert evaluate("9 / 3") == 3
    assert evaluate("7 - 1") == 6


def test_evaluate_rejects_garbage():
    try:
        evaluate("import os")
        assert False, "expected ValueError"
    except ValueError:
        pass
