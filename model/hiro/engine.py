from __future__ import annotations

from dataclasses import dataclass, field

from .classifiers import DiseaseModels
from .common import join_labels, label_for, normalize
from .constants import (
    CANDIDATE_MARGIN,
    CONFIDENCE_MARGIN,
    DENIED_ONLY,
    DENIED_PENALTY,
    DONE_PATTERN,
    EMERGENCY_NOTICE,
    EMERGENCY_PATTERN,
    INTRO_MESSAGE,
    MAX_CANDIDATES,
    MAX_QUESTIONS,
    MIN_SYMPTOMS,
    MODEL_NAMES,
    NO_PHRASES,
    NO_SYMPTOMS_YET,
    NO_WORDS,
    NOTED_NEW,
    NOTED_NO,
    NOTED_YES,
    QUESTION,
    UNDERSTOOD,
    UNRECOGNISED,
    YES_PHRASES,
    YES_WORDS,
)
from .intents import IntentClassifier
from .nlp import SymptomExtractor


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


class HiroEngine:
    def __init__(self, models: DiseaseModels | None = None, intents: IntentClassifier | None = None) -> None:
        self.models = models or DiseaseModels.load_or_train()
        self.intents = intents or IntentClassifier.load_or_train()
        self.labels: dict[str, str] = {column: label_for(column) for column in self.models.columns}
        self.extractor = SymptomExtractor(self.labels)

    def introduce(self) -> str:
        return INTRO_MESSAGE

    def symptom_labels(self) -> list[str]:
        return sorted(self.labels.values())

    def load_state(self, state: dict | None) -> Conversation:
        conv = Conversation()
        if not state:
            return conv
        for key in state.get("symptoms") or []:
            if key in self.labels:
                conv.add(key)
        conv.denied = {k for k in state.get("denied") or [] if k in self.labels} - set(conv.symptoms)
        pending = state.get("pending")
        conv.pending = pending if pending in self.labels else None
        conv.questions_asked = max(0, min(int(state.get("questions_asked") or 0), MAX_QUESTIONS))
        return conv

    def extract(self, text: str) -> tuple[list[str], list[str]]:
        return self.extractor.extract(text)

    def respond(self, conv: Conversation, user_input: str) -> dict:
        result = self._turn(conv, user_input)
        if result["type"] != "diagnosis" and EMERGENCY_PATTERN.search(normalize(user_input)):
            result["reply"] = EMERGENCY_NOTICE + result["reply"]
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
            if done:
                return self._chat(NO_SYMPTOMS_YET)
            if denied:
                return self._chat(DENIED_ONLY.format(symptoms=self._join(denied)))
            return self._chat(self.intents.reply(user_input))

        if done or conv.questions_asked >= MAX_QUESTIONS:
            return self.diagnose(conv)

        if conv.pending:
            return self._question(conv, conv.pending, UNRECOGNISED)

        ranking = self._rank(conv)
        if not ranking or self._confident(conv, ranking):
            return self.diagnose(conv)

        next_key = self._pick_question(conv, ranking)
        if next_key is None:
            return self.diagnose(conv)

        conv.pending = next_key
        conv.questions_asked += 1
        if new_keys:
            prefix = NOTED_NEW.format(symptoms=self._join(new_keys))
        elif asked and answer == "yes":
            prefix = NOTED_YES.format(symptom=self.labels[asked])
        elif asked and answer == "no":
            prefix = NOTED_NO.format(symptom=self.labels[asked])
        else:
            prefix = UNDERSTOOD
        return self._question(conv, next_key, prefix)

    @staticmethod
    def _chat(reply: str) -> dict:
        return {"type": "chat", "reply": reply}

    def _question(self, conv: Conversation, key: str, prefix: str) -> dict:
        return {
            "type": "question",
            "reply": QUESTION.format(prefix=prefix, symptoms=self._join(conv.symptoms), asked=self.labels[key]),
            "symptoms": [self.labels[k] for k in conv.symptoms],
            "asked": self.labels[key],
            "insight": self._insight(conv, self.models.predict_keys(set(conv.symptoms))),
        }

    def _insight(self, conv: Conversation, prediction: dict) -> dict:
        return {
            "symptom_count": len(conv.symptoms),
            "agreement": prediction["agreement"],
            "model_count": len(MODEL_NAMES),
            "models": prediction["models"],
            "candidates": prediction["candidates"],
        }

    def _rank(self, conv: Conversation) -> list[tuple[float, int, str]]:
        reported = set(conv.symptoms)
        ranking = []
        for disease, profile in self.models.profiles.items():
            hits = len(reported & profile)
            if hits == 0:
                continue
            score = hits - DENIED_PENALTY * len(conv.denied & profile)
            ranking.append((score, -len(profile), disease))
        ranking.sort(reverse=True)
        return ranking

    def _confident(self, conv: Conversation, ranking: list[tuple[float, int, str]]) -> bool:
        if len(conv.symptoms) < MIN_SYMPTOMS:
            return False
        return len(ranking) == 1 or ranking[0][0] - ranking[1][0] >= CONFIDENCE_MARGIN

    def _pick_question(self, conv: Conversation, ranking: list[tuple[float, int, str]]) -> str | None:
        top_score = ranking[0][0]
        candidates = [d for score, _, d in ranking if score >= top_score - CANDIDATE_MARGIN][:MAX_CANDIDATES]
        if len(candidates) < 2:
            candidates = [d for _, _, d in ranking[:3]]
        leader = self.models.profiles[candidates[0]]
        known = set(conv.symptoms) | conv.denied

        pool: dict[str, int] = {}
        for disease in candidates:
            for key in self.models.profiles[disease] - known:
                pool[key] = pool.get(key, 0) + 1
        if not pool:
            return None

        n = len(candidates)
        return min(pool, key=lambda k: (abs(pool[k] - n / 2), k not in leader, self.labels[k]))

    def diagnose(self, conv: Conversation) -> dict:
        prediction = self.models.predict_keys(set(conv.symptoms))
        final = prediction["final"]
        if len(conv.symptoms) < MIN_SYMPTOMS:
            confidence = "low"
        elif prediction["agreement"] == len(MODEL_NAMES):
            confidence = "high"
        else:
            confidence = "medium"

        result = {
            "type": "diagnosis",
            "symptoms": [self.labels[k] for k in conv.symptoms],
            "tests": prediction["votes"],
            "final_prediction": final,
            "confidence": confidence,
            "description": self.models.description(final),
            "precautions": self.models.precautions_for(final),
            "insight": self._insight(conv, prediction),
        }
        conv.reset()
        return result

    def _join(self, keys: list[str]) -> str:
        return join_labels([self.labels[k] for k in keys])
