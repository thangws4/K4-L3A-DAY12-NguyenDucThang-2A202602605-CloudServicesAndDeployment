"""Gọi LLM thật qua OpenRouter, rơi về mock LLM khi chưa có API key.

OpenRouter dùng API tương thích OpenAI (``/chat/completions``) và trả luôn
chi phí thật của lượt gọi trong ``usage.cost`` — cost guard dùng con số đó
thay vì ước lượng.

Hàm ``ask_llm`` giữ đúng interface của ``utils.mock_llm.ask_llm`` nên
``/ask`` không cần biết đang dùng provider nào.
"""

from __future__ import annotations

import httpx
from fastapi import HTTPException, status

from utils import mock_llm

from .config import get_settings
from .logging_utils import log_event

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI thân thiện, trả lời bằng tiếng Việt trừ khi người dùng "
    "hỏi bằng ngôn ngữ khác. Trả lời ngắn gọn, chính xác; không bịa thông tin."
)


def ask_llm(question: str, history: list[dict] | None = None) -> dict:
    """Trả về dict gồm answer, tokens_in, tokens_out, cost_usd."""
    settings = get_settings()
    if not settings.openrouter_api_key:
        return mock_llm.ask_llm(question, history)
    return ask_openrouter(question, history or [])


def _estimate_cost(tokens_in: int, tokens_out: int) -> float:
    """Dự phòng khi provider không trả ``usage.cost`` — dùng thang giá của mock."""
    return round(
        tokens_in / 1000 * mock_llm.PRICE_INPUT_PER_1K
        + tokens_out / 1000 * mock_llm.PRICE_OUTPUT_PER_1K,
        8,
    )


def ask_openrouter(
    question: str,
    history: list[dict],
    client: httpx.Client | None = None,
) -> dict:
    """Một lượt chat completion trên OpenRouter.

    ``client`` để test truyền vào transport giả; bình thường để None.
    Lỗi phía provider → 502, để /ask không ghi nhận chi phí cho lượt hỏng.
    """
    settings = get_settings()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": t["role"], "content": t["content"]} for t in history]
    messages.append({"role": "user", "content": question})

    payload = {
        "model": settings.openrouter_model,
        "messages": messages,
        "max_tokens": settings.llm_max_tokens,
        "usage": {"include": True},
    }
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "X-Title": "day12-agent",
    }

    http = client or httpx.Client(timeout=settings.llm_timeout_seconds)
    try:
        response = http.post(OPENROUTER_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"] or ""
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as err:
        detail = getattr(getattr(err, "response", None), "status_code", None)
        log_event(
            "llm_error",
            level="error",
            provider="openrouter",
            model=settings.openrouter_model,
            error=type(err).__name__,
            upstream_status=detail,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from err
    finally:
        if client is None:
            http.close()

    usage = data.get("usage") or {}
    tokens_in = int(usage.get("prompt_tokens") or 0)
    tokens_out = int(usage.get("completion_tokens") or 0)
    cost = usage.get("cost")
    cost_usd = float(cost) if cost is not None else _estimate_cost(tokens_in, tokens_out)

    return {
        "answer": answer.strip(),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost_usd,
        "model": data.get("model", settings.openrouter_model),
    }
