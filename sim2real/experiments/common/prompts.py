"""Operator prompts for experiments that run on the real arm."""

from __future__ import annotations


class Quit(Exception):
    """The operator stopped the sequence."""


def ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        raise Quit() from None
