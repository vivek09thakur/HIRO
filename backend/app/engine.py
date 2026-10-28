import csv
from pathlib import Path

from .core.companion import HEALTHCARE_COMPANION

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
MODEL_PATH = BACKEND_DIR / "model" / "chat_model.h5"

INTRO_MESSAGE = (
    "Hello there!, I am HIRO, your healthcare chatbot. "
    "I can help you diagnose your disease based on your symptoms. "
    "Let's start with your problem"
)


class HiroEngine:
    """Wraps the legacy CLI companion so each UI message runs the same turn loop."""

    def __init__(self) -> None:
        self.hiro = HEALTHCARE_COMPANION(
            str(DATA_DIR / "train.csv"),
            str(DATA_DIR / "test.csv"),
            str(DATA_DIR / "description.csv"),
            str(DATA_DIR / "precaution.csv"),
            str(DATA_DIR / "intents.json"),
            str(MODEL_PATH),
        )
        self.hiro.process_training_data()
        self.hiro.build_robust_model()

    def introduce(self) -> str:
        return INTRO_MESSAGE

    def respond(self, user_input: str) -> dict:
        extracted = self.hiro.extract_symptoms(user_input)
        if not extracted:
            return {"type": "chat", "reply": str(self.hiro.talk_to_user(user_input))}

        prediction = self.hiro.predict_disease_from_symptoms(extracted)
        if prediction is None:
            raise RuntimeError("Could not predict a disease from the extracted symptoms")

        final = str(prediction["Final Prediction"])
        return {
            "type": "diagnosis",
            "symptoms": [s.strip() for s in extracted.split(",") if s.strip()],
            "tests": {
                "Random Forest": str(prediction["Random Forest"]),
                "SVC": str(prediction["SVC"]),
                "Naive Bayes": str(prediction["Naive Bayes"]),
            },
            "final_prediction": final,
            "description": self.hiro.get_description(final),
            "precautions": self._precautions(final),
        }

    def _precautions(self, disease: str) -> list[str]:
        with open(DATA_DIR / "precaution.csv", newline="") as f:
            for row in csv.reader(f):
                if row and row[0] == disease:
                    return [p.strip() for p in row[1:] if p.strip()]
        return []
