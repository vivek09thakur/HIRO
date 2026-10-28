from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .engine import HiroEngine
from .hospitals import nearby_hospitals
from .schemas import ChatReply, ChatRequest, DiagnosisReply, Hospital, IntroResponse, QuestionReply


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = HiroEngine()
    yield


app = FastAPI(title="HIRO Healthcare Companion", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/intro", response_model=IntroResponse)
def intro(request: Request) -> dict:
    return {"message": request.app.state.engine.introduce()}


@app.get("/api/symptoms", response_model=list[str])
def symptoms(request: Request) -> list[str]:
    return sorted(request.app.state.engine.labels.values())


@app.get("/api/hospitals", response_model=list[Hospital])
async def hospitals(
    lat: float = Query(ge=-90, le=90),
    lon: float = Query(ge=-180, le=180),
    radius: int = Query(default=5000, ge=500, le=25000),
) -> list[dict]:
    try:
        return await nearby_hospitals(lat, lon, radius)
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Hospital lookup is unavailable right now.") from exc


@app.post("/api/chat", response_model=ChatReply | QuestionReply | DiagnosisReply)
async def chat(payload: ChatRequest, request: Request) -> dict:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    engine = request.app.state.engine
    conv = engine.load_state(payload.state.model_dump() if payload.state else None)
    try:
        return await run_in_threadpool(engine.respond, conv, message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ERROR OCCURED => {exc}") from exc
