"""Detect text copied from the competition data, so it does not end up in a public repository.

The competition data may not be shared or redistributed. This repository is publicly readable, so source files, docs and tests
must not contain real transcripts. The check looks for (a) a whole transcript of 20+ characters and (b) any run of `window`
consecutive words that occurs in a transcript. Both are compared after lower-casing, Unicode NFC and keeping only word characters.
Very common phrases can trigger (b); the fix is to rephrase the example, which is the intent.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

WINDOW = 8
MIN_FULL_CHARS = 20


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFC", text).lower()))


def build_index(transcripts: Iterable[str], window: int = WINDOW) -> tuple[set[str], set[str]]:
    """(all `window`-word windows, whole normalised transcripts) of the given texts."""
    windows: set[str] = set()
    wholes: set[str] = set()
    for text in transcripts:
        norm = normalize(text)
        words = norm.split()
        windows.update(" ".join(words[i : i + window]) for i in range(len(words) - window + 1))
        if len(norm) >= MIN_FULL_CHARS:
            wholes.add(norm)
    return windows, wholes


def find_leaks(text: str, windows: set[str], wholes: set[str], window: int = WINDOW, limit: int = 3) -> list[str]:
    """Fragments of `text` that come from the indexed transcripts: whole transcripts first, then 8-word windows (at most `limit`)."""
    norm = normalize(text)
    found = [w for w in wholes if w in norm][:limit]
    if len(found) < limit:
        words = norm.split()
        for i in range(len(words) - window + 1):
            candidate = " ".join(words[i : i + window])
            if candidate in windows:
                found.append(candidate)
                if len(found) >= limit:
                    break
    return found
