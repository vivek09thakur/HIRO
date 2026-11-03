import sys
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from .common import InferenceGate, Overloaded
from .constants import (
    APP_TITLE,
    INFERENCE_WORKERS,
    MAX_QUEUE,
    MSG_BUSY,
    MSG_CHAT_FAILED,
    MSG_EMPTY,
    RETRY_AFTER_SECONDS,
    ROOT_DIR,
    WEB_DIR,
)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.hiro import HiroEngine  # noqa: E402

from .schemas import ChatReply, ChatRequest, DiagnosisReply, IntroResponse, QuestionReply  # noqa: E402

gate = InferenceGate(INFERENCE_WORKERS, MAX_QUEUE)


@lru_cache(maxsize=1)
def get_engine() -> HiroEngine:
    return HiroEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_engine()
    yield


app = FastAPI(title=APP_TITLE, lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/intro", response_model=IntroResponse)
def intro() -> dict:
    return {"message": get_engine().introduce()}


@app.get("/api/symptoms", response_model=list[str])
def symptoms() -> list[str]:
    return get_engine().symptom_labels()


@app.post("/api/chat", response_model=ChatReply | QuestionReply | DiagnosisReply)
async def chat(payload: ChatRequest) -> dict:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail=MSG_EMPTY)
    engine = get_engine()
    conv = engine.load_state(payload.state.model_dump() if payload.state else None)
    try:
        return await gate.run(engine.respond, conv, message)
    except Overloaded:
        raise HTTPException(status_code=503, detail=MSG_BUSY, headers={"Retry-After": str(RETRY_AFTER_SECONDS)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=MSG_CHAT_FAILED.format(reason=exc)) from exc


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
