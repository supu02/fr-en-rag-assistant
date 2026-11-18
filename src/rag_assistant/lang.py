"""Tiny French/English detector based on common words.

Good enough for our two languages, deterministic, no extra dependency.
Used to (1) pick the retrieval method and (2) check, in the evaluation,
that the answer is written in the same language as the question.
"""

from __future__ import annotations

import re

FR_WORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "est", "je", "j", "mon", "ma", "mes",
    "pour", "dans", "que", "qui", "quel", "quelle", "combien", "puis", "peut", "pas", "avec",
    "sur", "au", "aux", "il", "elle", "ce", "cette", "son", "sa", "ses", "en", "si", "ou", "mais",
    "doit", "dois", "ai", "suis", "à", "être", "avoir", "jusqu", "quand", "comment", "après",
}
EN_WORDS = {
    "the", "a", "an", "and", "is", "are", "i", "my", "me", "for", "in", "that", "what", "which",
    "how", "many", "much", "can", "do", "does", "not", "with", "on", "to", "of", "it", "this",
    "his", "her", "if", "or", "but", "must", "have", "has", "am", "be", "by", "until", "when",
    "after", "will", "should", "get", "from", "at", "your", "you",
}


def detect_language(text: str) -> str:
    """Return 'fr' or 'en' (ties go to 'fr', the corpus language)."""
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ]+", text.lower())
    fr = sum(w in FR_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    return "en" if en > fr else "fr"
