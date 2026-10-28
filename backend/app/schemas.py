from typing import Literal

from pydantic import BaseModel, Field


class IntroResponse(BaseModel):
    message: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatReply(BaseModel):
    type: Literal["chat"]
    reply: str


class DiagnosisReply(BaseModel):
    type: Literal["diagnosis"]
    symptoms: list[str]
    tests: dict[str, str]
    final_prediction: str
    description: str
    precautions: list[str]
