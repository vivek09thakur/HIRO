import csv
import hashlib
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from .core.companion import HEALTHCARE_COMPANION
from .nlp import SymptomExtractor

SERVER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = SERVER_DIR / "data"
MODEL_PATH = SERVER_DIR / "model" / "chat_model.h5"
CACHE_PATH = SERVER_DIR / "model" / "disease_models.joblib"

INTRO_MESSAGE = (
    "Hello, I am HIRO, a symptom checker. Describe what you are feeling and I will ask a few "
    "follow-up questions before suggesting a possible condition. This is not a medical diagnosis. "
    "Say \"that's all\" at any time to see the result."
)

MIN_SYMPTOMS = 3
MAX_QUESTIONS = 5
MAX_CANDIDATES = 12

YES_WORDS = {"yes", "y", "yeah", "yep", "yup", "ya", "sure", "correct", "right", "true", "indeed"}
NO_WORDS = {"no", "n", "nope", "nah", "not", "never", "none", "false"}
YES_PHRASES = ("i do", "i am", "i have")
NO_PHRASES = ("i don't", "i dont", "not really", "i do not", "i am not")
DONE_PATTERN = re.compile(
    r"\b(that'?s? (all|it)|that is (all|it)|nothing else|no more|no other|"
    r"done|finish(ed)?|enough|only (that|this|these))\b"
)


@dataclass
class Conversation:
    symptoms: list[str] = field(default_factory=list)
    denied: set[str] = field(default_factory=set)
    pending: str | None = None
    questions_asked: int = 0

    def reset(self) -> None:
        self.symptoms.clear()
        self.denied.clear()
        self.pending = None
        self.questions_asked = 0

    def add(self, key: str) -> bool:
        if key in self.symptoms:
            return False
        self.symptoms.append(key)
        self.denied.discard(key)
        return True

    def to_dict(self) -> dict:
        return {
            "symptoms": list(self.symptoms),
            "denied": sorted(self.denied),
            "pending": self.pending,
            "questions_asked": self.questions_asked,
        }


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def leading_answer(text: str) -> str | None:
    if text.startswith(NO_PHRASES):
        return "no"
    if text.startswith(YES_PHRASES):
        return "yes"
    first = text.split(" ", 1)[0] if text else ""
    if first in YES_WORDS:
        return "yes"
    if first in NO_WORDS:
        return "no"
    return None


def dataset_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in ("train.csv", "test.csv"):
        digest.update((DATA_DIR / name).read_bytes())
    return digest.hexdigest()


class HiroEngine:
    """Runs the legacy models behind a multi-turn symptom interview."""

    def __init__(self) -> None:
        self.hiro = HEALTHCARE_COMPANION(
            str(DATA_DIR / "train.csv"),
            str(DATA_DIR / "test.csv"),
            str(DATA_DIR / "description.csv"),
            str(DATA_DIR / "precaution.csv"),
            str(DATA_DIR / "intents.json"),
            str(MODEL_PATH),
        )
        # Legacy predict passes a bare ndarray to models fitted on a DataFrame.
        warnings.filterwarnings("ignore", message="X does not have valid feature names")

        fingerprint = dataset_fingerprint()
        cached = self._load_cache(fingerprint)
        if cached:
            self.hiro.encoder = cached["encoder"]
            self.hiro.svc_model = cached["svc"]
            self.hiro.nb_model = cached["nb"]
            self.hiro.rf_model = cached["rf"]
            self.hiro.data_dict = cached["data_dict"]
            self.profiles = cached["profiles"]
        else:
            # process_training_data() also runs 10-fold CV on three models purely to print scores.
            self.hiro.preprocess()
            self.hiro.build_robust_model()
            self.profiles = self._build_profiles()
            self._save_cache(fingerprint)

        self.labels: dict[str, str] = {
            key: " ".join(key.split()).lower() for key in self.hiro.data_dict["symptoms_index"]
        }
        self.extractor = SymptomExtractor(self.labels)

    @staticmethod
    def _load_cache(fingerprint: str) -> dict | None:
        if not CACHE_PATH.exists():
            return None
        try:
            bundle = joblib.load(CACHE_PATH)
        except Exception:
            return None
        return bundle if bundle.get("fingerprint") == fingerprint else None

    def _save_cache(self, fingerprint: str) -> None:
        bundle = {
            "fingerprint": fingerprint,
            "encoder": self.hiro.encoder,
            "svc": self.hiro.svc_model,
            "nb": self.hiro.nb_model,
            "rf": self.hiro.rf_model,
            "data_dict": self.hiro.data_dict,
            "profiles": self.profiles,
        }
        try:
            joblib.dump(bundle, CACHE_PATH, compress=3)
        except OSError:
            pass

    def _build_profiles(self) -> dict[str, set[str]]:
        df = pd.read_csv(DATA_DIR / "train.csv").dropna(axis=1)
        columns = [c for c in df.columns if c != "prognosis"]
        means = df.groupby("prognosis")[columns].mean()
        key_of = {c: " ".join(p.capitalize() for p in c.split("_")) for c in columns}
        return {
            str(disease): {key_of[c] for c in columns if row[c] > 0}
            for disease, row in means.iterrows()
        }

    def introduce(self) -> str:
        return INTRO_MESSAGE

    def load_state(self, state: dict | None) -> Conversation:
        """Rebuild a Conversation from client-supplied state, dropping anything unknown."""
        conv = Conversation()
        if not state:
            return conv
        for key in state.get("symptoms", []):
            if key in self.labels:
                conv.add(key)
        conv.denied = {k for k in state.get("denied", []) if k in self.labels} - set(conv.symptoms)
        pending = state.get("pending")
        conv.pending = pending if pending in self.labels else None
        conv.questions_asked = max(0, min(int(state.get("questions_asked", 0)), MAX_QUESTIONS))
        return conv

    def extract(self, text: str) -> tuple[list[str], list[str]]:
        return self.extractor.extract(text)

    def respond(self, conv: Conversation, user_input: str) -> dict:
        result = self._turn(conv, user_input)
        result["state"] = conv.to_dict()
        return result

    def _turn(self, conv: Conversation, user_input: str) -> dict:
        text = normalize(user_input)
        answer = leading_answer(text)
        done = bool(DONE_PATTERN.search(text))

        asked = conv.pending
        if asked and answer == "yes":
            conv.add(asked)
            conv.pending = None
        elif asked and answer == "no":
            conv.denied.add(asked)
            conv.pending = None

        reported, denied = self.extract(text)
        new_keys = [key for key in reported if conv.add(key)]
        for key in denied:
            if key not in conv.symptoms:
                conv.denied.add(key)
                if key == conv.pending:
                    conv.pending = None
        if new_keys:
            conv.pending = None

        if not conv.symptoms:
            return {"type": "chat", "reply": self.hiro.talk_to_user(user_input)}

        if done or conv.questions_asked >= MAX_QUESTIONS:
            return self.diagnose(conv)

        if conv.pending:
            return self._question(conv, conv.pending, prefix="I did not recognise any symptoms in that message.")

        ranking = self._rank(conv)
        if not ranking or self._confident(conv, ranking):
            return self.diagnose(conv)

        next_key = self._pick_question(conv, ranking)
        if next_key is None:
            return self.diagnose(conv)

        conv.pending = next_key
        conv.questions_asked += 1
        if new_keys:
            prefix = f"I have noted the following symptoms: {self._join(new_keys)}."
        elif asked and answer == "yes":
            prefix = f"Noted, {self.labels[asked]}."
        else:
            prefix = "Understood."
        return self._question(conv, next_key, prefix=prefix)

    def _question(self, conv: Conversation, key: str, prefix: str) -> dict:
        reply = (
            f"{prefix} Symptoms recorded so far: {self._join(conv.symptoms)}. "
            f"Are you also experiencing {self.labels[key]}?"
        )
        return {
            "type": "question",
            "reply": reply,
            "symptoms": [self.labels[k] for k in conv.symptoms],
            "asked": self.labels[key],
        }

    def _rank(self, conv: Conversation) -> list[tuple[float, int, str]]:
        reported = set(conv.symptoms)
        ranking = []
        for disease, profile in self.profiles.items():
            hits = len(reported & profile)
            if hits == 0:
                continue
            score = hits - 0.5 * len(conv.denied & profile)
            ranking.append((score, -len(profile), disease))
        ranking.sort(reverse=True)
        return ranking

    def _confident(self, conv: Conversation, ranking: list[tuple[float, int, str]]) -> bool:
        if len(conv.symptoms) < MIN_SYMPTOMS:
            return False
        return len(ranking) == 1 or ranking[0][0] - ranking[1][0] >= 1.0

    def _pick_question(self, conv: Conversation, ranking: list[tuple[float, int, str]]) -> str | None:
        top_score = ranking[0][0]
        candidates = [d for score, _, d in ranking if score >= top_score - 0.5][:MAX_CANDIDATES]
        if len(candidates) < 2:
            candidates = [d for _, _, d in ranking[:3]]
        leader = self.profiles[candidates[0]]
        known = set(conv.symptoms) | conv.denied

        pool: dict[str, int] = {}
        for disease in candidates:
            for key in self.profiles[disease] - known:
                pool[key] = pool.get(key, 0) + 1
        if not pool:
            return None

        n = len(candidates)
        # Prefer the symptom that splits the candidates most evenly, then one from the leader.
        return min(pool, key=lambda k: (abs(pool[k] - n / 2), k not in leader, self.labels[k]))

    def diagnose(self, conv: Conversation) -> dict:
        prediction = self.hiro.predict_disease_from_symptoms(",".join(conv.symptoms))
        if prediction is None:
            conv.reset()
            raise RuntimeError("Could not predict a disease from the collected symptoms")

        final = str(prediction["Final Prediction"])
        tests = {
            "Random Forest": str(prediction["Random Forest"]),
            "SVC": str(prediction["SVC"]),
            "Naive Bayes": str(prediction["Naive Bayes"]),
        }
        if len(conv.symptoms) < MIN_SYMPTOMS:
            confidence = "low"
        elif len(set(tests.values())) == 1:
            confidence = "high"
        else:
            confidence = "medium"

        result = {
            "type": "diagnosis",
            "symptoms": [self.labels[k] for k in conv.symptoms],
            "tests": tests,
            "final_prediction": final,
            "confidence": confidence,
            "description": self.hiro.get_description(final),
            "precautions": self._precautions(final),
        }
        conv.reset()
        return result

    def _join(self, keys: list[str]) -> str:
        return ", ".join(self.labels[k] for k in keys)

    def _precautions(self, disease: str) -> list[str]:
        with open(DATA_DIR / "precaution.csv", newline="") as f:
            for row in csv.reader(f):
                if row and row[0] == disease:
                    return [p.strip() for p in row[1:] if p.strip()]
        return []
