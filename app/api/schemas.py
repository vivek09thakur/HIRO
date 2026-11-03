from typing import Literal

from pydantic import BaseModel, Field


class IntroResponse(BaseModel):
    message: str


class ConversationState(BaseModel):
    symptoms: list[str] = Field(default_factory=list, max_length=64)
    denied: list[str] = Field(default_factory=list, max_length=128)
    pending: str | None = None
    questions_asked: int = Field(default=0, ge=0, le=50)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    state: ConversationState | None = None


class ChatReply(BaseModel):
    type: Literal["chat"]
    reply: str
    state: ConversationState


class ModelReading(BaseModel):
    name: str
    prediction: str
    probability: float = Field(ge=0, le=1)
    agrees: bool


class Candidate(BaseModel):
    disease: str
    probability: float = Field(ge=0, le=1)


class Insight(BaseModel):
    symptom_count: int
    agreement: int
    model_count: int
    models: list[ModelReading]
    candidates: list[Candidate]


class QuestionReply(BaseModel):
    type: Literal["question"]
    reply: str
    symptoms: list[str]
    asked: str
    insight: Insight
    state: ConversationState


class DiagnosisReply(BaseModel):
    type: Literal["diagnosis"]
    symptoms: list[str]
    tests: dict[str, str]
    final_prediction: str
    confidence: Literal["low", "medium", "high"]
    description: str
    precautions: list[str]
    insight: Insight
    state: ConversationState
