from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from hello_rag_agent import get_service


app = FastAPI(title="Hello Agents RAG API", version="2.0.0")
service = get_service()


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    user_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    user_id: str
    message_count: int


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "hello-agents-rag",
        "knowledge_base": service.knowledge_stats(),
    }


def _run_chat(request: ChatRequest) -> ChatResponse:
    try:
        response, session_id = service.ask(request.query, request.session_id, user_id=request.user_id)
        history = service.get_history(session_id)
        return ChatResponse(
            response=response,
            session_id=session_id,
            user_id=service.get_user_id(session_id) or request.user_id or f"session:{session_id}",
            message_count=len(history),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return _run_chat(request)


@app.post("/api/query", response_model=ChatResponse)
def query(request: ChatRequest):
    return _run_chat(request)


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    try:
        stream, session_id = service.stream_ask(
            request.query,
            request.session_id,
            user_id=request.user_id,
        )
        resolved_user_id = service.get_user_id(session_id) or request.user_id or f"session:{session_id}"
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    def event_stream():
        header = {
            "session_id": session_id,
            "user_id": resolved_user_id,
        }
        yield f"event: meta\ndata: {json.dumps(header, ensure_ascii=False)}\n\n"

        response_parts: list[str] = []
        for chunk in stream:
            response_parts.append(chunk)
            payload = {"text": chunk}
            yield f"event: chunk\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        history = service.get_history(session_id)
        done_payload = {
            "response": "".join(response_parts),
            "session_id": session_id,
            "user_id": resolved_user_id,
            "message_count": len(history),
        }
        yield f"event: done\ndata: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class ResetRequest(BaseModel):
    session_id: str


@app.post("/api/session/reset")
def reset_session(request: ResetRequest):
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    service.reset_session(request.session_id)
    return {"status": "ok", "session_id": request.session_id}


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    return {
        "session_id": session_id,
        "user_id": service.get_user_id(session_id),
        "history": service.get_history(session_id),
    }


@app.get("/api/knowledge/stats")
def knowledge_stats():
    return service.knowledge_stats()
