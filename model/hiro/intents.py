from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .common import Artifact, normalize
from .constants import (
    CHAR_NGRAMS,
    FALLBACK_TAG,
    INTENT_C,
    INTENT_CONFIDENCE_THRESHOLD,
    INTENT_MAX_ITER,
    INTENT_MODEL,
    INTENTS_JSON,
    WORD_NGRAMS,
)


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=CHAR_NGRAMS, sublinear_tf=True)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=WORD_NGRAMS, sublinear_tf=True)),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("clf", LogisticRegression(C=INTENT_C, max_iter=INTENT_MAX_ITER, class_weight="balanced")),
        ]
    )


@dataclass
class IntentClassifier(Artifact):
    artifact_path: ClassVar[Path] = INTENT_MODEL

    pipeline: Pipeline
    responses: dict[str, list[str]]
    tags: list[str]
    fingerprint: str = ""

    @classmethod
    def train(cls, intents_json: Path = INTENTS_JSON) -> "IntentClassifier":
        with open(intents_json, encoding="utf-8") as f:
            intents = json.load(f)["intents"]

        texts, labels, responses = [], [], {}
        for intent in intents:
            responses[intent["tag"]] = intent["responses"]
            for pattern in intent["patterns"]:
                texts.append(normalize(pattern))
                labels.append(intent["tag"])

        pipeline = build_pipeline()
        pipeline.fit(texts, labels)
        return cls(pipeline=pipeline, responses=responses, tags=sorted(set(labels)))

    def classify(self, text: str) -> tuple[str, float]:
        clean = normalize(text)
        if not clean:
            return FALLBACK_TAG, 1.0
        probabilities = self.pipeline.predict_proba([clean])[0]
        best = int(probabilities.argmax())
        confidence = float(probabilities[best])
        if confidence < INTENT_CONFIDENCE_THRESHOLD:
            return FALLBACK_TAG, confidence
        return str(self.pipeline.classes_[best]), confidence

    def reply(self, text: str, rng: random.Random | None = None) -> str:
        tag, _ = self.classify(text)
        options = self.responses.get(tag) or self.responses[FALLBACK_TAG]
        return (rng or random).choice(options)
