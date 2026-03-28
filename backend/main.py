"""
FastAPI app — rate limiting, restricted CORS, request size cap, secure headers.
"""
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

load_dotenv()

from model import FIREInput, StartChatRequest, ChatRequest
from calculator import calculate_fire
from ai_advisor import get_ai_roadmap, _fallback_roadmap
from chat_advisor import get_chat_response
from database import (
    save_plan,
    create_chat_session,
    save_message,
    get_chat_history,
    get_user_sessions,
    update_session_activity,
    delete_user_data,
)

# ── Rate limiter ────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/day"])

app = FastAPI(
    title="FIRE Path Planner API",
    description="AI-powered FIRE planning for Indians",
    version="3.2",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS — restrict to known origins ───────────────────────
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

# ── Request body size limit (1 MB) ─────────────────────────
@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 1_048_576:
        raise HTTPException(status_code=413, detail="Request body too large.")
    return await call_next(request)

# ── In-memory session → fire_summary store ─────────────────
_session_summaries: dict = {}

# ── Static frontend ────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root():
    return RedirectResponse(url="/app/firststep_page.html")


@app.get("/app/{page_name}")
def serve_page(page_name: str):
    # Prevent path traversal
    if ".." in page_name or "/" in page_name:
        raise HTTPException(status_code=400, detail="Invalid page name.")
    file_path = os.path.join(FRONTEND_DIR, page_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Page not found.")


# ── FIRE Plan ──────────────────────────────────────────────
@app.post("/fire-plan")
@limiter.limit("5/minute")
def create_fire_plan(request: Request, data: FIREInput):
    try:
        if data.fire_target_age <= data.age:
            raise HTTPException(status_code=400, detail="FIRE target age must be greater than current age.")
        if data.monthly_expenses >= data.monthly_income:
            raise HTTPException(status_code=400, detail="Expenses must be less than income.")

        calc_results = calculate_fire(data)

        _ROADMAP_TIMEOUT = 150.0
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(get_ai_roadmap, data, calc_results)
            try:
                ai_roadmap = fut.result(timeout=_ROADMAP_TIMEOUT)
            except FuturesTimeout:
                ai_roadmap = _fallback_roadmap(calc_results, data)
            except Exception:
                ai_roadmap = _fallback_roadmap(calc_results, data)

        plan_id, user_id = save_plan(data, calc_results, ai_roadmap)

        return {
            "status": "success",
            "user_id": str(user_id),
            "plan_id": str(plan_id),
            "language": data.language,
            "user": data.name,
            "results": calc_results,
            "ai_roadmap": ai_roadmap,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error.")


# ── Chat ───────────────────────────────────────────────────
@app.post("/chat/start")
@limiter.limit("10/minute")
def start_chat(request: Request, req: StartChatRequest):
    try:
        session_id = create_chat_session(
            user_id=req.user_id,
            language=req.language.value,
            plan_id=req.plan_id,
        )
        if req.fire_summary:
            _session_summaries[str(session_id)] = req.fire_summary[:2000]
        return {"status": "success", "session_id": str(session_id)}
    except Exception:
        raise HTTPException(status_code=500, detail="Could not start chat session.")


@app.post("/chat/message")
@limiter.limit("20/minute")
def send_message(request: Request, req: ChatRequest):
    try:
        history = get_chat_history(req.session_id, limit=20)
        fire_summary = _session_summaries.get(req.session_id, "")

        ai_reply, retrieved_context = get_chat_response(
            user_message=req.message,
            chat_history=history,
            language=req.language.value,
            fire_summary=fire_summary,
        )

        save_message(req.session_id, "user",      req.message, req.language.value)
        save_message(req.session_id, "assistant", ai_reply,    req.language.value, retrieved_context)
        update_session_activity(req.session_id)

        return {"status": "success", "reply": ai_reply, "session_id": req.session_id}
    except Exception:
        raise HTTPException(status_code=500, detail="Could not process message.")


@app.get("/chat/history/{session_id}")
@limiter.limit("30/minute")
def get_history(request: Request, session_id: str):
    try:
        messages = get_chat_history(session_id, limit=100)
        return {"session_id": session_id, "messages": messages, "count": len(messages)}
    except Exception:
        raise HTTPException(status_code=500, detail="Could not fetch history.")


@app.get("/chat/sessions/{user_id}")
@limiter.limit("10/minute")
def get_sessions(request: Request, user_id: str):
    try:
        sessions = get_user_sessions(user_id)
        return {"user_id": user_id, "sessions": sessions, "count": len(sessions)}
    except Exception:
        raise HTTPException(status_code=500, detail="Could not fetch sessions.")


# ── GDPR-style delete ──────────────────────────────────────
@app.delete("/user/{user_id}")
@limiter.limit("3/minute")
def delete_user(request: Request, user_id: str):
    try:
        delete_user_data(user_id)
        return {"status": "success", "message": "All user data deleted."}
    except Exception:
        raise HTTPException(status_code=500, detail="Could not delete user data.")


# ── Knowledge search ───────────────────────────────────────
@app.get("/search")
@limiter.limit("10/minute")
def search_knowledge_base(request: Request, query: str):
    if len(query) > 500:
        raise HTTPException(status_code=400, detail="Query too long.")
    try:
        from rag.retriever import retrieve_relevant_context
        context = retrieve_relevant_context(query, k=4)
        return {"query": query, "context": context}
    except Exception:
        raise HTTPException(status_code=500, detail="Search unavailable.")
