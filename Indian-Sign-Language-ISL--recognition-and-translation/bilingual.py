"""
Bilingual text support: Kannada script -> Roman transliteration -> ISL
alphabet finger-spelling.

This module is the text-processing heart of SignBridge AI's bilingual
feature. It provides:

  * ``detect_language(text)``       - "kn" (Kannada script) / "en" (Latin) /
                                      "mixed" when both scripts are present.
  * ``transliterate_kannada(text)`` - Kannada script -> ASCII Roman letters
                                      (rule-based IAST-like scheme).
  * ``prepare_for_isl(text)``       - one-call helper returning
                                      ``(roman_text, language)`` where
                                      ``roman_text`` is UPPERCASE A-Z (plus
                                      digits and spaces), ready for the
                                      existing ISL alphabet assets.

Scope notes (intentional):
  * This is NOT Kannada *Sign Language* recognition. The CNN recognizes ISL
    alphabet hand signs only; this module converts Kannada *text* into a
    Roman letter sequence that is then finger-spelled with the existing A-Z
    ISL alphabet images.
  * The Kannada-news dataset (kannada_dataset/) is unrelated NLP corpus
    material, kept separate from the CNN training data. It is used only as a
    source of example headlines, never as model data.
  * Transliteration is deterministic and rule-based - no external services.

Unsupported Kannada characters (rare/archaic glyphs such as U+0CF1/U+0CF2,
unmapped symbols) are skipped predictably: they contribute no letters but
never crash the pipeline. Whitespace of any kind (spaces, tabs, newlines,
zero-width joiners from copy-paste) is normalized to single spaces.

Transliteration scheme (IAST-like, ASCII output):

    Independent vowels   ಅ a    ಆ aa   ಇ i    ಈ ii   ಉ u    ಊ uu
                         ಎ e    ಏ ee   ಐ ai   ಒ o    ಓ oo   ಔ au
    Consonants carry an inherent 'a': ಕ ka, ಮ ma, ನ na, ರ ra ...
    Matras modify them:               ಕಿ ki, ಗು gu, ನೀ nii ...
    Virama (್) suppresses it:          ಕ್ k, ನ್ನ nna
    Anusvara/swara:                   ಂ m, ಃ h
    Kannada digits ೦-೯ map to 0-9.

Run this file directly for a small self-test covering English, Kannada and
mixed input.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

# Unicode block for Kannada: U+0C80 .. U+0CFF
_KANNADA_RE = re.compile(r"[\u0C80-\u0CFF]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# Invisible/zero-width characters commonly picked up from copy-paste
# (ZWSP, ZWNJ, ZWJ, BOM, soft hyphen, directional marks).
_INVISIBLE_RE = re.compile(
    r"[\u200B\u200C\u200D\uFEFF\u00AD\u2066-\u2069]"
)

# ---------------------------------------------------------------------------
# Kannada -> Roman transliteration tables
# ---------------------------------------------------------------------------

# Independent vowels
_VOWELS = {
    "ಅ": "a", "ಆ": "aa", "ಇ": "i", "ಈ": "ii", "ಉ": "u", "ಊ": "uu",
    "ಋ": "ru", "ೠ": "ruu", "ಎ": "e", "ಏ": "ee", "ಐ": "ai",
    "ಒ": "o", "ಓ": "oo", "ಔ": "au", "ಂ": "m", "ಃ": "h",
}

# Consonants (with inherent 'a')
_CONSONANTS = {
    "ಕ": "ka", "ಖ": "kha", "ಗ": "ga", "ಘ": "gha", "ಙ": "nga",
    "ಚ": "cha", "ಛ": "chha", "ಜ": "ja", "ಝ": "jha", "ಞ": "nya",
    "ಟ": "ta", "ಠ": "tha", "ಡ": "da", "ಢ": "dha", "ಣ": "na",
    "ತ": "ta", "ಥ": "tha", "ದ": "da", "ಧ": "dha", "ನ": "na",
    "ಪ": "pa", "ಫ": "pha", "ಬ": "ba", "ಭ": "bha", "ಮ": "ma",
    "ಯ": "ya", "ರ": "ra", "ಱ": "rra", "ಲ": "la", "ಳ": "la", "ೞ": "lla",
    "ವ": "va", "ಶ": "sha", "ಷ": "sha", "ಸ": "sa", "ಹ": "ha",
}

# Vowel signs (matras) - follow a consonant
_MATRAS = {
    "ಾ": "aa", "ಿ": "i", "ೀ": "ii", "ು": "u", "ೂ": "uu",
    "ೃ": "ru", "ೄ": "ruu", "ೆ": "e", "ೇ": "ee", "ೈ": "ai",
    "ೊ": "o", "ೋ": "oo", "ೌ": "au",
}

_VIRAMA = "\u0CCD"  # ್ halant: suppresses the inherent vowel 'a'

# Digits ೦-೯
_DIGITS = {
    "೦": "0", "೧": "1", "೨": "2", "೩": "3", "೪": "4",
    "೫": "5", "೬": "6", "೭": "7", "೮": "8", "೯": "9",
}

# Letters whose transliteration must be deduplicated when two identical
# consonants meet in a conjunct (ನ್ನ -> "nna", not "nanna").
_DOUBLED_CONSONANTS = {
    "ಕ", "ಖ", "ಗ", "ಘ", "ಚ", "ಛ", "ಜ", "ಝ", "ಟ", "ಠ", "ಡ", "ಢ", "ಣ",
    "ತ", "ಥ", "ದ", "ಧ", "ನ", "ಪ", "ಫ", "ಬ", "ಭ", "ಮ", "ಯ", "ರ", "ಲ",
    "ಳ", "ವ", "ಶ", "ಷ", "ಸ", "ಹ",
}


def detect_language(text: str) -> str:
    """
    Classify ``text`` by script.

    Returns:
        "kn"    - Kannada script characters present, no Latin letters.
        "mixed" - both Kannada and Latin letters present.
        "en"    - Latin letters only (or empty text).

    Invisible characters (zero-width spaces, BOM) are ignored; punctuation,
    digits and whitespace do not affect the verdict.
    """
    cleaned = _INVISIBLE_RE.sub("", text or "")
    has_kn = bool(_KANNADA_RE.search(cleaned))
    has_lat = bool(_LATIN_RE.search(cleaned))
    if has_kn and has_lat:
        return "mixed"
    if has_kn:
        return "kn"
    return "en"


def transliterate_kannada(text: str) -> str:
    """
    Convert Kannada script text to lowercase ASCII Roman letters.

    English letters, digits, spaces and basic punctuation pass through
    unchanged, so mixed text is safe. Unsupported Kannada characters
    (archaic glyphs, unmapped symbols) are skipped without error.

    Examples:
        "ನಮಸ್ಕಾರ" -> "namaskaara"
        "ಕನ್ನಡ"   -> "kannada"
        "ಗುರು"     -> "guru"
    """
    if not text:
        return ""

    # Strip invisible characters first so copy-paste artifacts cannot
    # break consonant+matra pairing.
    text = _INVISIBLE_RE.sub("", text)

    out: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch in _VOWELS:
            out.append(_VOWELS[ch])
            i += 1
        elif ch in _CONSONANTS:
            # Look ahead: consonant + optional matra / virama
            if i + 1 < n and text[i + 1] == _VIRAMA:
                # Dead consonant: bare form without inherent 'a'.
                # A following consonant repeats it: ನ್ನ -> "nna".
                bare = _CONSONANTS[ch][:-1]  # drop trailing 'a'
                j = i + 2
                if (
                    ch in _DOUBLED_CONSONANTS
                    and j < n
                    and text[j] == ch
                ):
                    out.append(bare * 2)
                    i = j + 1
                    # A matra may follow the cluster: ನ್ನಾ -> "nnaa"
                    if i < n and text[i] in _MATRAS:
                        out[-1] = out[-1] + _MATRAS[text[i]]
                        i += 1
                else:
                    out.append(bare)
                    i += 1  # consume only the virama; next char re-parsed
            elif i + 1 < n and text[i + 1] in _MATRAS:
                base = _CONSONANTS[ch][:-1]  # drop trailing 'a'
                out.append(base + _MATRAS[text[i + 1]])
                i += 2
            else:
                out.append(_CONSONANTS[ch])
                i += 1
        elif ch == _VIRAMA:
            # Standalone virama (rare) - skip
            i += 1
        elif ch in _MATRAS:
            # Matra without a base consonant - emit vowel approximation
            out.append(_MATRAS[ch].rstrip("a") or "a")
            i += 1
        elif ch in _DIGITS:
            out.append(_DIGITS[ch])
            i += 1
        else:
            # Spaces, punctuation, Latin text pass through unchanged
            out.append(ch)
            i += 1

    roman = "".join(out)
    # Normalize unicode form and collapse all whitespace to single spaces
    roman = unicodedata.normalize("NFKC", roman)
    roman = re.sub(r"\s+", " ", roman).strip()
    return roman


# ---------------------------------------------------------------------------
# Finger-spelling preparation
# ---------------------------------------------------------------------------

# Characters that can appear in finger-spelling output: A-Z, 0-9, space.
_FINGERSPELLABLE = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
)


def prepare_for_isl(text: str) -> tuple[str, str]:
    """
    One-call helper for the finger-spelling pipeline.

    Normalizes ``text`` (any script), transliterates Kannada if present, and
    returns ``(roman_text, language)``:

        ``roman_text``  UPPERCASE A-Z letters, digits and single spaces only -
                        directly usable with the existing ISL alphabet assets.
        ``language``    "en", "kn" or "mixed".

    Unsupported characters are dropped predictably; whitespace (including
    newlines/tabs/zero-width characters) collapses to single spaces.

    Examples:
        >>> prepare_for_isl("Hello")
        ('HELLO', 'en')
        >>> prepare_for_isl("ನಮಸ್ಕಾರ")
        ('NAMASKAARA', 'kn')
        >>> prepare_for_isl("Hello ನಮಸ್ಕಾರ")
        ('HELLO NAMASKAARA', 'mixed')
    """
    if not text:
        return "", "en"

    cleaned = _INVISIBLE_RE.sub("", text)
    language = detect_language(cleaned)
    if language in ("kn", "mixed"):
        roman = transliterate_kannada(cleaned)
    else:
        roman = cleaned

    # Keep only finger-spellable characters.
    filtered = "".join(ch for ch in roman if ch in _FINGERSPELLABLE)
    filtered = re.sub(r"\s+", " ", filtered).strip()
    return filtered.upper(), language


def isl_letter_sequence(roman_text: str) -> list[str]:
    """
    Split a romanized text into a readable A-Z letter sequence for display.

    Letters become single-character strings; word boundaries become the
    literal string ``"space"`` so the UI can render word gaps clearly.
    Digits are passed through (they are finger-spelled by name in practice).

    Example:
        >>> isl_letter_sequence("HI THERE")
        ['H', 'I', 'space', 'T', 'H', 'E', 'R', 'E']
    """
    return [
        ch if ch != " " else "space"
        for ch in (roman_text or "").upper()
        if ch in _FINGERSPELLABLE and (ch.isalnum() or ch == " ")
    ]


if __name__ == "__main__":
    import sys

    # Windows consoles often use cp1252 which cannot print Kannada glyphs.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

    samples = [
        "Hello",                    # English
        "ನಮಸ್ಕಾರ",                  # Kannada
        "ಕನ್ನಡ",                    # Kannada
        "ಗುರು",                     # Kannada (matra)
        "ನಮಸ್ಕಾರ Hello",            # Mixed
        "Hello  ಕನ್ನಡ!\tಒಳ್ಳೆಯದು",   # Mixed with punctuation + tabs
        "",                         # Empty
        "!!!",                      # Only unsupported characters
    ]
    for s in samples:
        lang = detect_language(s)
        roman, lang2 = prepare_for_isl(s)
        seq = isl_letter_sequence(roman)
        print(f"{s!r:30} lang={lang:5} roman={roman!r:20} seq={seq if len(seq) < 12 else str(seq[:10]) + '...'}")

    # Minimal self-checks
    assert prepare_for_isl("Hello") == ("HELLO", "en")
    assert prepare_for_isl("ನಮಸ್ಕಾರ") == ("NAMASKAARA", "kn")
    assert prepare_for_isl("ಕನ್ನಡ") == ("KANNADA", "kn")
    assert prepare_for_isl("Hello ನಮಸ್ಕಾರ") == ("HELLO NAMASKAARA", "mixed")
    assert prepare_for_isl("!!!") == ("", "en")
    assert isl_letter_sequence("HI") == ["H", "I"]
    assert isl_letter_sequence("HI THERE") == ["H", "I", "space", "T", "H", "E", "R", "E"]
    print("\nALL BILINGUAL SELF-CHECKS PASSED")
