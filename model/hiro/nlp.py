from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches

from .common import normalize
from .constants import (
    ALIASES,
    BODY_PART_PATTERNS,
    BODY_PART_WORD,
    BODY_PARTS,
    COORDINATORS,
    DURATION_PATTERNS,
    FUZZY_MIN_LABEL_LEN,
    FUZZY_THRESHOLD,
    NEGATION_CUES,
    NEGATION_FILLERS,
    NEGATION_WINDOW,
    NGRAM_STOP,
    POST_NEGATION,
    POST_NEGATION_WINDOW,
    SPELL_MIN_LEN,
    SPELL_THRESHOLD,
    STOPWORDS,
    WORD_PATTERN,
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    priority: int
    anchor: int = -1

    @property
    def negation_anchor(self) -> int:
        return self.start if self.anchor < 0 else self.anchor


def strip_durations(text: str) -> str:
    for pattern in DURATION_PATTERNS:
        text = pattern.sub(" ", text)
    return " ".join(text.split())


class SymptomExtractor:
    def __init__(self, labels: dict[str, str]) -> None:
        self.key_of = {label: key for key, label in labels.items()}
        self.labels = sorted(self.key_of, key=len, reverse=True)
        self.label_patterns = [(re.compile(rf"\b{re.escape(label)}\b"), label) for label in self.labels]
        self.alias_patterns = [
            (re.compile(rf"\b(?:{pattern})\b"), label) for pattern, label in ALIASES if label in self.key_of
        ]
        self.fuzzy_labels = [label for label in self.labels if len(label) >= FUZZY_MIN_LABEL_LEN]
        self.vocabulary = self._build_vocabulary()

    def _build_vocabulary(self) -> set[str]:
        words: set[str] = set()
        for label in self.labels:
            words.update(label.split())
        for pattern, label in ALIASES:
            if label in self.key_of:
                words.update(WORD_PATTERN.findall(pattern))
        words.update(BODY_PARTS)
        words.update(NEGATION_CUES)
        return {w for w in words if len(w) >= SPELL_MIN_LEN and w.isalpha()}

    def prepare(self, text: str) -> str:
        return self.correct_spelling(strip_durations(normalize(text)))

    def correct_spelling(self, text: str) -> str:
        out = []
        for word in text.split():
            if len(word) < SPELL_MIN_LEN or not word.isalpha() or word in self.vocabulary or word in STOPWORDS:
                out.append(word)
                continue
            match = get_close_matches(word, self.vocabulary, n=1, cutoff=SPELL_THRESHOLD)
            out.append(match[0] if match and match[0][0] == word[0] else word)
        return " ".join(out)

    def extract(self, text: str) -> tuple[list[str], list[str]]:
        text = self.prepare(text)
        tokens = list(WORD_PATTERN.finditer(text))
        spans = self._exact_spans(text) + self._alias_spans(text) + self._body_part_spans(text)
        spans += self._fuzzy_spans(tokens, {s.label for s in spans})
        chosen = self._resolve(spans)

        reported: list[str] = []
        denied: list[str] = []
        previous: tuple[Span, bool] | None = None
        for span in chosen:
            key = self.key_of[span.label]
            negated = self._negated(span, tokens)
            if not negated and previous and previous[1]:
                between = text[previous[0].end : span.negation_anchor].split()
                negated = bool(between) and all(w in COORDINATORS for w in between)
            previous = (span, negated)
            target = denied if negated else reported
            if key not in target:
                target.append(key)
        return [k for k in reported if k not in denied], denied

    def _exact_spans(self, text: str) -> list[Span]:
        return [Span(m.start(), m.end(), label, 3) for pattern, label in self.label_patterns for m in pattern.finditer(text)]

    def _alias_spans(self, text: str) -> list[Span]:
        return [Span(m.start(), m.end(), label, 2) for pattern, label in self.alias_patterns for m in pattern.finditer(text)]

    def _body_part_spans(self, text: str) -> list[Span]:
        spans = []
        for pattern in BODY_PART_PATTERNS:
            for m in pattern.finditer(text):
                parts = list(BODY_PART_WORD.finditer(text, m.start(), m.end()))
                for i, part in enumerate(parts):
                    label = BODY_PARTS[part.group(1)]
                    if label not in self.key_of:
                        continue
                    if i == 0:
                        spans.append(Span(m.start(), part.end() if len(parts) > 1 else m.end(), label, 2))
                    else:
                        spans.append(Span(part.start(), part.end(), label, 2, anchor=m.start()))
        return spans

    def _fuzzy_spans(self, tokens: list[re.Match], found: set[str]) -> list[Span]:
        spans = []
        words = [t.group() for t in tokens]
        for label in self.fuzzy_labels:
            if label in found:
                continue
            n = len(label.split())
            for i in range(len(words) - n + 1):
                chunk = words[i : i + n]
                if chunk[0] in NGRAM_STOP or chunk[-1] in NGRAM_STOP or min(len(w) for w in chunk) < 4:
                    continue
                if SequenceMatcher(None, " ".join(chunk), label).ratio() >= FUZZY_THRESHOLD:
                    spans.append(Span(tokens[i].start(), tokens[i + n - 1].end(), label, 1))
                    break
        return spans

    @staticmethod
    def _resolve(spans: list[Span]) -> list[Span]:
        chosen: list[Span] = []
        for span in sorted(spans, key=lambda s: (-(s.end - s.start), -s.priority, s.start)):
            if all(span.end <= c.start or span.start >= c.end for c in chosen):
                chosen.append(span)
        return sorted(chosen, key=lambda s: s.start)

    @staticmethod
    def _negated(span: Span, tokens: list[re.Match]) -> bool:
        anchor = span.negation_anchor
        before = [t.group() for t in tokens if t.end() <= anchor]
        for word in reversed(before[-NEGATION_WINDOW:]):
            if word in NEGATION_CUES:
                return True
            if word not in NEGATION_FILLERS:
                break
        after = [t.group() for t in tokens if t.start() >= span.end][:POST_NEGATION_WINDOW]
        return bool(after) and bool(POST_NEGATION.match(" ".join(after)))
