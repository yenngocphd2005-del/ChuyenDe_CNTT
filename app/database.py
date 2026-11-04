"""
Module Database - SQLAlchemy models và kết nối PostgreSQL
"""
import uuid
import logging
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Text, DateTime, Integer, ForeignKey,
    create_engine, text,
)
from sqlalchemy.orm import (
    DeclarativeBase, relationship, sessionmaker, Session,
)

from app.config import settings

logger = logging.getLogger(__name__)


# ── Base ────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Models ──────────────────────────────────────────────────────────
class ChatSession(Base):
    """Phiên hội thoại."""
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(200), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    messages = relationship(
        "ChatMessage", back_populates="session",
        order_by="ChatMessage.sequence_number",
        cascade="all, delete-orphan",
    )


class ChatMessage(Base):
    """Tin nhắn trong phiên hội thoại."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user' hoặc 'assistant'
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)       # JSON string: nguồn tài liệu đã dùng
    tool_trace = Column(Text, nullable=True)    # JSON string: trace Agentic
    sequence_number = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session = relationship("ChatSession", back_populates="messages")


# ── Engine & Session ────────────────────────────────────────────────
_engine = None
_SessionLocal = None


def get_engine():
    """Lấy hoặc tạo SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
    return _engine


def get_session() -> Session:
    """Tạo database session mới."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


def init_database():
    """Tạo tất cả các bảng nếu chưa tồn tại."""
    engine = get_engine()
    try:
        # Kiểm tra kết nối
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Kết nối PostgreSQL thành công.")

        # Tạo bảng
        Base.metadata.create_all(engine)
        logger.info("Đã tạo/kiểm tra các bảng database.")
        return True
    except Exception as e:
        logger.error(f"Lỗi kết nối database: {e}")
        return False


# ── CRUD Operations ─────────────────────────────────────────────────
def create_session(title: str = None) -> ChatSession:
    """Tạo phiên chat mới."""
    session = get_session()
    try:
        chat_session = ChatSession(title=title)
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)
        logger.info(f"Tạo phiên chat mới: {chat_session.id}")
        return chat_session
    except Exception as e:
        session.rollback()
        logger.error(f"Lỗi tạo phiên chat: {e}")
        raise
    finally:
        session.close()


def save_message(
    session_id: str,
    role: str,
    content: str,
    sources: str = None,
    tool_trace: str = None,
) -> ChatMessage:
    """Lưu tin nhắn vào phiên chat."""
    session = get_session()
    try:
        # Tính sequence_number tiếp theo
        last_msg = (
            session.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.sequence_number.desc())
            .first()
        )
        next_seq = (last_msg.sequence_number + 1) if last_msg else 1

        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources,
            tool_trace=tool_trace,
            sequence_number=next_seq,
        )
        session.add(message)

        # Cập nhật updated_at cho session
        chat_session = session.query(ChatSession).filter(ChatSession.id == session_id).first()
        if chat_session:
            chat_session.updated_at = datetime.now(timezone.utc)
            # Tự tạo title từ câu hỏi đầu tiên
            if not chat_session.title and role == "user":
                chat_session.title = content[:100]

        session.commit()
        session.refresh(message)
        return message
    except Exception as e:
        session.rollback()
        logger.error(f"Lỗi lưu tin nhắn: {e}")
        raise
    finally:
        session.close()


def get_messages(session_id: str, limit: int = 50) -> list[ChatMessage]:
    """Lấy lịch sử tin nhắn theo phiên."""
    session = get_session()
    try:
        messages = (
            session.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.sequence_number.asc())
            .limit(limit)
            .all()
        )
        # Detach from session for use outside
        for msg in messages:
            session.expunge(msg)
        return messages
    finally:
        session.close()


def get_recent_messages(session_id: str, limit: int = 10) -> list[dict]:
    """Lấy N tin nhắn gần nhất dưới dạng dict."""
    session = get_session()
    try:
        messages = (
            session.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.sequence_number.desc())
            .limit(limit)
            .all()
        )
        return [
            {"role": m.role, "content": m.content}
            for m in reversed(messages)
        ]
    finally:
        session.close()


def list_sessions(limit: int = 50) -> list[ChatSession]:
    """Liệt kê các phiên chat."""
    session = get_session()
    try:
        sessions = (
            session.query(ChatSession)
            .order_by(ChatSession.updated_at.desc())
            .limit(limit)
            .all()
        )
        for s in sessions:
            session.expunge(s)
        return sessions
    finally:
        session.close()


def delete_session(session_id: str) -> bool:
    """Xóa phiên chat."""
    session = get_session()
    try:
        chat_session = session.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not chat_session:
            return False
        session.delete(chat_session)
        session.commit()
        logger.info(f"Đã xóa phiên chat: {session_id}")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Lỗi xóa phiên chat: {e}")
        return False
    finally:
        session.close()


def get_session_by_id(session_id: str) -> ChatSession | None:
    """Lấy thông tin phiên chat theo ID."""
    session = get_session()
    try:
        chat_session = session.query(ChatSession).filter(ChatSession.id == session_id).first()
        if chat_session:
            session.expunge(chat_session)
        return chat_session
    finally:
        session.close()
