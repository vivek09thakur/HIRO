from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import ClassVar

import joblib


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def label_for(column: str) -> str:
    return " ".join(column.replace("_", " ").split()).lower()


def join_labels(labels: list[str]) -> str:
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def fingerprint(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_lookup(path: Path) -> dict[str, list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row[0]: [c.strip() for c in row[1:] if c.strip()] for row in csv.reader(f) if row}


class Artifact:
    artifact_path: ClassVar[Path]
    fingerprint: str = ""

    @classmethod
    def train(cls, **kwargs) -> "Artifact":
        raise NotImplementedError

    @classmethod
    def current_fingerprint(cls) -> str:
        return ""

    def save(self, path: Path | None = None) -> None:
        path = path or self.artifact_path
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: Path | None = None):
        path = path or cls.artifact_path
        if not path.exists():
            return None
        try:
            bundle = joblib.load(path)
        except Exception:
            return None
        if not isinstance(bundle, cls):
            return None
        try:
            expected = cls.current_fingerprint()
        except OSError:
            return bundle
        return bundle if not expected or bundle.fingerprint == expected else None

    @classmethod
    def load_or_train(cls):
        bundle = cls.load()
        if bundle is None:
            bundle = cls.train()
            try:
                bundle.save()
            except OSError:
                pass
        return bundle
