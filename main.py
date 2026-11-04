"""
FastAPI Backend - Chatbot Địa đạo Củ Chi
API endpoints cho frontend và xử lý AI.
"""
import asyncio
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Đảm bảo app package trong path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings
from app import database, rag, llm, agent

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Thread pool để chạy LLM inference không block event loop
# Chỉ 1 worker để tránh 2 inference đồng thời gây tràn bộ nhớ (8GB RAM + 4GB VRAM)
executor = ThreadPoolExecutor(max_workers=1)


# ── Lifespan ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo các module khi start server."""
    logger.info("=" * 60)
    logger.info("KHỞI ĐỘNG CHATBOT ĐỊA ĐẠO CỦ CHI")
    logger.info("=" * 60)

    # 1. Kiểm tra database
    logger.info("[1/3] Kiểm tra kết nối PostgreSQL...")
    db_ok = database.init_database()
    if not db_ok:
        logger.warning("⚠ Không thể kết nối PostgreSQL. Lịch sử hội thoại sẽ không được lưu.")

    # 2. Xây dựng FAISS index
    logger.info("[2/3] Xây dựng FAISS index...")
    try:
        rag.build_index()
        logger.info("✓ FAISS index sẵn sàng.")
    except Exception as e:
        logger.error(f"✗ Lỗi FAISS index: {e}")

    # 3. Tải LLM (chạy trong background để không chặn startup)
    logger.info("[3/3] Bắt đầu tải Qwen2.5-3B-Instruct...")
    try:
        llm.load_model()
        logger.info("✓ LLM đã sẵn sàng.")
    except Exception as e:
        logger.error(f"✗ Lỗi tải LLM: {e}")

    logger.info("=" * 60)
    logger.info("CHATBOT SẴN SÀNG PHỤC VỤ")
    logger.info(f"Truy cập: http://localhost:{settings.PORT}")
    logger.info("=" * 60)

    yield

    logger.info("Đang tắt server...")


# ── FastAPI App ─────────────────────────────────────────────────────
app = FastAPI(
    title="Chatbot Địa đạo Củ Chi",
    description="Agentic AI Chatbot hỗ trợ tham quan Địa đạo Củ Chi",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Câu hỏi của người dùng")
    session_id: str | None = Field(None, description="ID phiên chat (tự tạo nếu để trống)")


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: list | None = None
    agent_trace: dict | None = None


class SessionResponse(BaseModel):
    id: str
    title: str | None
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    sources: str | None
    sequence_number: int
    created_at: str


class SessionCreateRequest(BaseModel):
    title: str | None = None


# ── API Endpoints ───────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái backend."""
    return {
        "status": "ok",
        "llm_loaded": llm.is_model_loaded(),
        "database": database.init_database() if settings.DB_PASSWORD else False,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint chính: nhận câu hỏi, xử lý bằng Agent, trả câu trả lời.
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    # Tạo hoặc lấy session
    session_id = request.session_id
    if session_id:
        existing = database.get_session_by_id(session_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Phiên chat không tồn tại.")
    else:
        session = database.create_session()
        session_id = session.id

    try:
        # Lưu tin nhắn user
        database.save_message(session_id, "user", message)
    except Exception as e:
        logger.warning(f"Không thể lưu tin nhắn user: {e}")

    try:
        # Chạy Agent trong thread pool để không block event loop
        loop = asyncio.get_event_loop()
        response_text, trace, sources = await loop.run_in_executor(
            executor,
            agent.process_query,
            message,
            session_id,
        )
    except Exception as e:
        logger.error(f"Lỗi xử lý Agent: {e}")
        response_text = "Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý. Vui lòng thử lại."
        trace = agent.AgentTrace(user_query=message)
        sources = None

    try:
        # Lưu tin nhắn assistant
        database.save_message(
            session_id, "assistant", response_text,
            sources=sources,
            tool_trace=trace.to_json() if trace else None,
        )
    except Exception as e:
        logger.warning(f"Không thể lưu tin nhắn assistant: {e}")

    # Parse sources cho response
    sources_list = None
    if sources:
        try:
            sources_list = json.loads(sources)
        except json.JSONDecodeError:
            pass

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        sources=sources_list,
        agent_trace=trace.to_dict() if trace else None,
    )


@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest = None):
    """Tạo phiên chat mới."""
    try:
        title = request.title if request else None
        session = database.create_session(title=title)
        return SessionResponse(
            id=session.id,
            title=session.title,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
        )
    except Exception as e:
        logger.error(f"Lỗi tạo session: {e}")
        raise HTTPException(status_code=500, detail="Không thể tạo phiên chat mới.")


@app.get("/api/sessions", response_model=list[SessionResponse])
async def list_sessions():
    """Liệt kê các phiên chat."""
    try:
        sessions = database.list_sessions()
        return [
            SessionResponse(
                id=s.id,
                title=s.title,
                created_at=s.created_at.isoformat(),
                updated_at=s.updated_at.isoformat(),
            )
            for s in sessions
        ]
    except Exception as e:
        logger.error(f"Lỗi liệt kê sessions: {e}")
        raise HTTPException(status_code=500, detail="Không thể lấy danh sách phiên chat.")


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Xóa một phiên chat."""
    ok = database.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Phiên chat không tồn tại.")
    return {"status": "ok"}


@app.get("/api/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def get_session_messages(session_id: str):
    """Lấy lịch sử tin nhắn của phiên chat."""
    session = database.get_session_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Phiên chat không tồn tại.")

    try:
        messages = database.get_messages(session_id)
        return [
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                sources=m.sources,
                sequence_number=m.sequence_number,
                created_at=m.created_at.isoformat(),
            )
            for m in messages
        ]
    except Exception as e:
        logger.error(f"Lỗi lấy messages: {e}")
        raise HTTPException(status_code=500, detail="Không thể lấy lịch sử tin nhắn.")


# ── Serve Frontend ──────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    """Phục vụ giao diện chatbot."""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Chatbot Địa đạo Củ Chi</h1><p>index.html không tìm thấy.</p>")


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        workers=1,
    )
