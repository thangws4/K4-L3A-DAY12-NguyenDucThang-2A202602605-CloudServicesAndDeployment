"""OpenRouter adapter + trang chat — không gọi mạng thật (httpx.MockTransport).

Chạy: pytest tests/test_llm.py -v
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import HTTPException


@pytest.fixture
def settings_with_key(monkeypatch):
    """Settings có OPENROUTER_API_KEY giả, không đọc .env của máy."""
    from app import llm
    from app.config import Settings

    settings = Settings(
        _env_file=None,
        openrouter_api_key="or-test-key",
        openrouter_model="test/model",
        llm_max_tokens=123,
    )
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    return settings


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _ok_body(content="Xin chào!", cost=0.00042):
    usage = {"prompt_tokens": 12, "completion_tokens": 5}
    if cost is not None:
        usage["cost"] = cost
    return {
        "model": "test/model",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": usage,
    }


class TestOpenRouter:
    def test_khong_co_key_thi_dung_mock(self, monkeypatch):
        from app import llm
        from app.config import Settings

        monkeypatch.setattr(llm, "get_settings", lambda: Settings(_env_file=None))
        result = llm.ask_llm("Docker là gì?", [])
        assert result["answer"] and result["cost_usd"] > 0

    def test_gui_dung_request(self, settings_with_key):
        from app.llm import OPENROUTER_URL, ask_openrouter

        seen = {}

        def handler(request: httpx.Request):
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["Authorization"]
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_ok_body())

        history = [
            {"role": "user", "content": "câu cũ"},
            {"role": "assistant", "content": "trả lời cũ"},
        ]
        ask_openrouter("câu mới", history, client=_client(handler))

        body = seen["body"]
        assert seen["url"] == OPENROUTER_URL
        assert seen["auth"] == "Bearer or-test-key"
        assert body["model"] == "test/model"
        assert body["max_tokens"] == 123
        assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "user"]
        assert body["messages"][-1]["content"] == "câu mới"

    def test_doc_answer_token_va_chi_phi_that(self, settings_with_key):
        from app.llm import ask_openrouter

        result = ask_openrouter(
            "hi", [], client=_client(lambda r: httpx.Response(200, json=_ok_body()))
        )
        assert result == {
            "answer": "Xin chào!",
            "tokens_in": 12,
            "tokens_out": 5,
            "cost_usd": pytest.approx(0.00042),
            "model": "test/model",
        }

    def test_thieu_cost_thi_uoc_luong(self, settings_with_key):
        from app.llm import ask_openrouter

        result = ask_openrouter(
            "hi", [], client=_client(lambda r: httpx.Response(200, json=_ok_body(cost=None)))
        )
        assert result["cost_usd"] > 0, "không có usage.cost vẫn phải tính chi phí"

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(500, json={"error": "boom"}),
            httpx.Response(401, json={"error": "bad key"}),
            httpx.Response(200, json={"choices": []}),
        ],
    )
    def test_loi_provider_thi_502(self, settings_with_key, response):
        from app.llm import ask_openrouter

        with pytest.raises(HTTPException) as err:
            ask_openrouter("hi", [], client=_client(lambda r: response))
        assert err.value.status_code == 502

    def test_loi_provider_khong_tinh_tien(self, client, fake_redis, auth_headers, monkeypatch):
        """LLM lỗi → /ask trả 502 và KHÔNG ghi nhận chi phí, không lưu lịch sử."""
        from app import main
        from app.cost_guard import CostGuard

        def hong(question, history):
            raise HTTPException(status_code=502, detail="LLM provider error")

        monkeypatch.setattr(main, "ask_llm", hong)
        response = client.post("/ask", json={"question": "x"}, headers=auth_headers)
        assert response.status_code == 502
        assert CostGuard(fake_redis, 10.0).spent("sv-test") == 0.0


class TestChatUi:
    def test_trang_chat(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "/ask" in response.text

    def test_history_can_api_key(self, client):
        assert client.get("/history").status_code == 401
        assert client.delete("/history").status_code == 401

    def test_doc_va_xoa_lich_su(self, client, auth_headers):
        client.post("/ask", json={"question": "Xin chào"}, headers=auth_headers)
        messages = client.get("/history", headers=auth_headers).json()["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]

        assert client.delete("/history", headers=auth_headers).status_code == 200
        assert client.get("/history", headers=auth_headers).json()["messages"] == []
