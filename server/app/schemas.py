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


class QuestionReply(BaseModel):
    type: Literal["question"]
    reply: str
    symptoms: list[str]
    asked: str
    state: ConversationState


class DiagnosisReply(BaseModel):
    type: Literal["diagnosis"]
    symptoms: list[str]
    tests: dict[str, str]
    final_prediction: str
    confidence: Literal["low", "medium", "high"]
    description: str
    precautions: list[str]
    state: ConversationState


class Hospital(BaseModel):
    name: str
    kind: str
    distance_km: float
    address: str | None = None
    phone: str | None = None
    emergency: bool = False
    lat: float
    lon: float
    directions_url: str
