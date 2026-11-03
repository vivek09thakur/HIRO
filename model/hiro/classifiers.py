from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

from .common import Artifact, fingerprint, read_lookup
from .constants import (
    CV_FOLDS,
    DESCRIPTION_CSV,
    DISEASE_MODELS,
    NB_ALPHA,
    NO_DESCRIPTION,
    PRECAUTION_CSV,
    RANDOM_STATE,
    RF_ESTIMATORS,
    SCORE_TEMPERATURE,
    SVC_C,
    TARGET_COLUMN,
    TEST_CSV,
    TIE_BREAKER,
    TOP_CANDIDATES,
    TRAIN_CSV,
)


def build_models() -> dict[str, object]:
    return {
        "Random Forest": RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=RANDOM_STATE),
        "SVC": LinearSVC(C=SVC_C, random_state=RANDOM_STATE),
        "Naive Bayes": BernoulliNB(alpha=NB_ALPHA),
    }


def probabilities(model, x: np.ndarray) -> np.ndarray:
    if isinstance(model, RandomForestClassifier):
        x32 = x.astype(np.float32)
        total = np.zeros(model.n_classes_)
        for tree in model.estimators_:
            leaf = tree.tree_.predict(x32)[0]
            total += leaf / leaf.sum()
        return total / len(model.estimators_)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[0]
    scores = np.asarray(model.decision_function(x))[0] * SCORE_TEMPERATURE
    scores = scores - scores.max()
    e = np.exp(scores)
    return e / e.sum()


@dataclass
class DiseaseModels(Artifact):
    artifact_path: ClassVar[Path] = DISEASE_MODELS

    columns: list[str]
    classes: list[str]
    models: dict[str, object]
    profiles: dict[str, set[str]]
    descriptions: dict[str, str]
    precautions: dict[str, list[str]]
    fingerprint: str
    scores: dict[str, float] = field(default_factory=dict)

    @classmethod
    def current_fingerprint(cls) -> str:
        return fingerprint(TRAIN_CSV, DESCRIPTION_CSV, PRECAUTION_CSV)

    @classmethod
    def train(cls, train_csv: Path = TRAIN_CSV, evaluate: bool = False) -> "DiseaseModels":
        df = pd.read_csv(train_csv)
        columns = [c for c in df.columns if c != TARGET_COLUMN]
        X = df[columns].to_numpy(dtype=np.uint8)
        classes = sorted(df[TARGET_COLUMN].unique())
        y = np.array([classes.index(d) for d in df[TARGET_COLUMN]])

        models = build_models()
        scores = {}
        for name, model in models.items():
            if evaluate:
                scores[name] = float(cross_val_score(model, X, y, cv=CV_FOLDS).mean())
            model.fit(X, y)

        means = df.groupby(TARGET_COLUMN)[columns].mean()
        return cls(
            columns=columns,
            classes=classes,
            models=models,
            profiles={str(d): {c for c in columns if row[c] > 0} for d, row in means.iterrows()},
            descriptions={d: v[0] for d, v in read_lookup(DESCRIPTION_CSV).items() if v},
            precautions=read_lookup(PRECAUTION_CSV),
            fingerprint=cls.current_fingerprint(),
            scores=scores,
        )

    def evaluate(self, test_csv: Path = TEST_CSV) -> dict[str, float]:
        df = pd.read_csv(test_csv)
        X = df.reindex(columns=self.columns, fill_value=0).to_numpy(dtype=np.uint8)
        truth = df[TARGET_COLUMN].tolist()
        result = {}
        for name, model in self.models.items():
            pred = [self.classes[i] for i in model.predict(X)]
            result[name] = sum(p == t for p, t in zip(pred, truth)) / len(truth)
        votes = [self.predict_keys({c for c in self.columns if row[c]}) for _, row in df.iterrows()]
        result["Ensemble"] = sum(v["final"] == t for v, t in zip(votes, truth)) / len(truth)
        return result

    def predict_keys(self, keys: set[str]) -> dict:
        x = np.array([[int(c in keys) for c in self.columns]], dtype=np.uint8)
        votes: dict[str, str] = {}
        models: list[dict] = []
        total = np.zeros(len(self.classes))
        for name, model in self.models.items():
            proba = probabilities(model, x)
            best = int(proba.argmax())
            votes[name] = self.classes[best]
            models.append({"name": name, "prediction": self.classes[best], "probability": float(proba[best])})
            total += proba
        counts = Counter(votes.values())
        top = counts.most_common(1)[0][1]
        final = votes[TIE_BREAKER] if counts[votes[TIE_BREAKER]] == top else counts.most_common(1)[0][0]
        mean = total / len(self.models)
        order = np.argsort(mean)[::-1][:TOP_CANDIDATES]
        candidates = [{"disease": self.classes[i], "probability": float(mean[i])} for i in order if mean[i] > 0]
        for m in models:
            m["agrees"] = m["prediction"] == final
        return {"votes": votes, "final": final, "agreement": top, "models": models, "candidates": candidates}

    def description(self, disease: str) -> str:
        return self.descriptions.get(disease, NO_DESCRIPTION)

    def precautions_for(self, disease: str) -> list[str]:
        return self.precautions.get(disease, [])
