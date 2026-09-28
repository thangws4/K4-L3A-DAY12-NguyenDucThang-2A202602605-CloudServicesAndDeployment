"""Agent service — điểm ráp nối của cả lab (CP1, CP3, CP4).

Luồng một request tới /ask:

    client ──► verify_api_key ──► rate_limiter ──► cost_guard
                                                       │
                              store.get_history ◄──────┘
                                       │
                                    ask_llm
                                       │
                              store.append × 2 ──► cost_guard.record ──► log_event
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi import Path as PathParam
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .auth import verify_api_key
from .config import get_settings
from .cost_guard import CostGuard
from .lifecycle import lifecycle
from .llm import ask_llm
from .logging_utils import log_event
from .rate_limiter import RateLimiter
from .store import ConversationIndex, ConversationStore, ThreadStore, get_redis_client

SERVICE_NAME = "day12-agent"
SERVICE_VERSION = "1.0.0"
CHAT_PAGE = Path(__file__).parent / "static" / "chat.html"
# Không chứa ":" để khóa thread:<user>:<conversation> không bao giờ trùng nhau
CONVERSATION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"


# ─────────────────────────────────────────────────────────────
# Providers — CHO SẴN
# Tách ra thành hàm để test có thể thay bằng Redis giả qua
# app.dependency_overrides, và để kết nối Redis chỉ tạo khi thật sự cần.
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_store() -> ConversationStore:
    return ConversationStore(get_redis_client())


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter:
    return RateLimiter(get_redis_client(), get_settings().rate_limit_per_minute)


@lru_cache(maxsize=1)
def get_cost_guard() -> CostGuard:
    return CostGuard(get_redis_client(), get_settings().monthly_budget_usd)


@lru_cache(maxsize=1)
def get_threads() -> ThreadStore:
    return ThreadStore(get_redis_client())


@lru_cache(maxsize=1)
def get_conversation_index() -> ConversationIndex:
    return ConversationIndex(get_redis_client())


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """CHO SẴN — chạy lúc app khởi động và lúc tắt."""
    lifecycle.install()
    log_event("service_started", service=SERVICE_NAME, version=SERVICE_VERSION)
    yield
    log_event("service_stopped", service=SERVICE_NAME)


app = FastAPI(title="Day 12 Production Agent", version=SERVICE_VERSION, lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Không gửi → dùng lịch sử chung của user như luồng CP3/CP4 ban đầu
    conversation_id: str | None = Field(default=None, pattern=CONVERSATION_ID_PATTERN)


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


# ─────────────────────────────────────────────────────────────
# Giao diện chat — trang tĩnh, không chứa secret; API key do người dùng
# nhập trên trình duyệt và gửi kèm mỗi request tới /ask.
# ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
@app.get("/ask", include_in_schema=False)  # mở /ask trên trình duyệt cũng ra trang chat; POST /ask vẫn là API
def chat_page():
    return FileResponse(CHAT_PAGE, media_type="text/html")


# ─────────────────────────────────────────────────────────────
# Health & readiness
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness probe — process còn sống không?

    TODO (CP1 + CP4):
      - Đang tắt dần (``lifecycle.shutting_down``) → trả
        ``JSONResponse(status_code=503, content={"status": "shutting_down"})``
      - Bình thường → ``{"status": "ok", "service": SERVICE_NAME,
        "version": SERVICE_VERSION}`` (mặc định FastAPI trả 200).

    Endpoint này phải **nhẹ**: không gọi Redis, không query DB. Nó chỉ trả
    lời câu hỏi "có cần restart container này không?". Nếu nó phụ thuộc
    Redis, Redis chết một nhịp là cả cụm container bị restart theo.
    """
    if lifecycle.shutting_down:
        return JSONResponse(status_code=503, content={"status": "shutting_down"})
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.get("/ready")
def ready(store: ConversationStore = Depends(get_store)):
    """Readiness probe — đã sẵn sàng nhận traffic chưa?

    TODO (CP4):
      - Đang tắt dần → 503 ``{"status": "shutting_down"}``
      - ``store.ping()`` False → 503 ``{"status": "not ready", "redis": False}``
      - Ngược lại → ``{"status": "ready", "redis": True}``

    Khác /health ở chỗ: endpoint này ĐƯỢC PHÉP kiểm tra dependency. Load
    balancer dùng nó để quyết định có đẩy request vào instance này không.
    """
    if lifecycle.shutting_down:
        return JSONResponse(status_code=503, content={"status": "shutting_down"})
    if not store.ping():
        return JSONResponse(status_code=503, content={"status": "not ready", "redis": False})
    return {"status": "ready", "redis": True}


# ─────────────────────────────────────────────────────────────
# Endpoint chính
# ─────────────────────────────────────────────────────────────
@app.post("/ask")
def ask(
    payload: AskRequest,
    user_id: str = Depends(verify_api_key),
    store: ConversationStore = Depends(get_store),
    limiter: RateLimiter = Depends(get_rate_limiter),
    guard: CostGuard = Depends(get_cost_guard),
    threads: ThreadStore = Depends(get_threads),
    conversations: ConversationIndex = Depends(get_conversation_index),
):
    """Hỏi agent một câu.

    TODO (CP3 + CP4) — làm ĐÚNG THỨ TỰ sau:
      1. ``limiter.check(user_id)``           → 429 nếu gọi quá nhanh
      2. ``guard.check(user_id)``             → 402 nếu hết ngân sách
      3. ``history = store.get_history(user_id)``
      4. ``result = ask_llm(payload.question, history)``
      5. ``store.append(user_id, "user", payload.question)`` và
         ``store.append(user_id, "assistant", result["answer"])``
      6. ``guard.record(user_id, result["cost_usd"])``
      7. ``log_event("ask_completed", user_id=user_id,
         tokens_in=result["tokens_in"], tokens_out=result["tokens_out"],
         cost_usd=result["cost_usd"])``
      8. trả về::

            {
                "answer": result["answer"],
                "user_id": user_id,
                "history_length": len(history),
                "cost_usd": result["cost_usd"],
                "tokens": {"in": result["tokens_in"], "out": result["tokens_out"]},
            }

    Vì sao check trước rồi mới gọi LLM? Vì tiền mất ở bước gọi LLM. Chặn sau
    khi đã gọi thì bạn vừa trả tiền vừa trả lỗi.

    ``user_id`` do ``verify_api_key`` trả về, nên request không có API key
    hợp lệ sẽ dừng ở 401 trước khi chạm vào bất cứ dòng nào ở đây.
    """
    # Chặn trước khi gọi LLM — tiền mất ở bước gọi LLM
    limiter.check(user_id)
    guard.check(user_id)

    conversation_id = payload.conversation_id
    if conversation_id:
        history_store, history_key = threads, ThreadStore.thread_id(user_id, conversation_id)
    else:
        history_store, history_key = store, user_id

    history = history_store.get_history(history_key)
    result = ask_llm(payload.question, history)

    history_store.append(history_key, "user", payload.question)
    history_store.append(history_key, "assistant", result["answer"])
    guard.record(user_id, result["cost_usd"])

    if conversation_id:
        # Câu hỏi đầu tiên làm tiêu đề; quá giới hạn thì bỏ cuộc cũ nhất
        for stale in conversations.touch(user_id, conversation_id, payload.question):
            threads.clear(ThreadStore.thread_id(user_id, stale))

    log_event(
        "ask_completed",
        user_id=user_id,
        tokens_in=result["tokens_in"],
        tokens_out=result["tokens_out"],
        cost_usd=result["cost_usd"],
        model=result.get("model", "mock"),
    )
    return {
        "answer": result["answer"],
        "user_id": user_id,
        "history_length": len(history),
        "cost_usd": result["cost_usd"],
        "tokens": {"in": result["tokens_in"], "out": result["tokens_out"]},
        "model": result.get("model", "mock"),
        "conversation_id": conversation_id,
    }


# ─────────────────────────────────────────────────────────────
# Danh sách cuộc trò chuyện — thanh lịch sử bên trái trang chat
# ─────────────────────────────────────────────────────────────
ConversationId = PathParam(pattern=CONVERSATION_ID_PATTERN)


def _require_conversation(conversations: ConversationIndex, user_id: str, conversation_id: str) -> dict:
    meta = conversations.get(user_id, conversation_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return meta


@app.get("/conversations")
def list_conversations(
    user_id: str = Depends(verify_api_key),
    conversations: ConversationIndex = Depends(get_conversation_index),
):
    return {"user_id": user_id, "conversations": conversations.list(user_id)}


@app.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str = ConversationId,
    user_id: str = Depends(verify_api_key),
    threads: ThreadStore = Depends(get_threads),
    conversations: ConversationIndex = Depends(get_conversation_index),
):
    meta = _require_conversation(conversations, user_id, conversation_id)
    messages = threads.get_history(ThreadStore.thread_id(user_id, conversation_id))
    return {**meta, "messages": messages}


@app.patch("/conversations/{conversation_id}")
def rename_conversation(
    body: RenameRequest,
    conversation_id: str = ConversationId,
    user_id: str = Depends(verify_api_key),
    conversations: ConversationIndex = Depends(get_conversation_index),
):
    _require_conversation(conversations, user_id, conversation_id)
    conversations.rename(user_id, conversation_id, body.title)
    return conversations.get(user_id, conversation_id)


@app.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str = ConversationId,
    user_id: str = Depends(verify_api_key),
    threads: ThreadStore = Depends(get_threads),
    conversations: ConversationIndex = Depends(get_conversation_index),
):
    _require_conversation(conversations, user_id, conversation_id)
    threads.clear(ThreadStore.thread_id(user_id, conversation_id))
    conversations.remove(user_id, conversation_id)
    log_event("conversation_deleted", user_id=user_id, conversation_id=conversation_id)
    return {"id": conversation_id, "deleted": True}


@app.get("/history")
def get_history(
    user_id: str = Depends(verify_api_key),
    store: ConversationStore = Depends(get_store),
):
    """Lịch sử hội thoại của user — trang chat tải lại khi refresh."""
    return {"user_id": user_id, "messages": store.get_history(user_id)}


@app.delete("/history")
def clear_history(
    user_id: str = Depends(verify_api_key),
    store: ConversationStore = Depends(get_store),
):
    """Xóa lịch sử hội thoại của user — nút "Cuộc trò chuyện mới" trên trang chat."""
    store.clear(user_id)
    log_event("history_cleared", user_id=user_id)
    return {"user_id": user_id, "cleared": True}


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
