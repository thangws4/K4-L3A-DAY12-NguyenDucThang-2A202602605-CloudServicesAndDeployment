"""Nhiều cuộc trò chuyện cho mỗi user — nguồn dữ liệu của thanh lịch sử.

Chạy: pytest tests/test_conversations.py -v
"""

from __future__ import annotations

import pytest

CID_A = "conv-aaaaaaaa"
CID_B = "conv-bbbbbbbb"


@pytest.fixture
def chat(client, fake_redis):
    """Client có ThreadStore + ConversationIndex chạy trên Redis giả."""
    from app import main
    from app.store import ConversationIndex, ThreadStore

    main.app.dependency_overrides[main.get_threads] = lambda: ThreadStore(fake_redis)
    main.app.dependency_overrides[main.get_conversation_index] = lambda: ConversationIndex(fake_redis)
    return client


def ask(chat, headers, question, cid=None):
    body = {"question": question}
    if cid:
        body["conversation_id"] = cid
    response = chat.post("/ask", json=body, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


class TestConversationIndex:
    def test_moi_nhat_len_dau_va_giu_tieu_de_dau_tien(self, fake_redis):
        from app.store import ConversationIndex

        index = ConversationIndex(fake_redis)
        index.touch("u1", CID_A, "Câu hỏi đầu của A", now=100)
        index.touch("u1", CID_B, "Câu hỏi đầu của B", now=200)
        index.touch("u1", CID_A, "Câu hỏi thứ hai của A", now=300)

        rows = index.list("u1")
        assert [r["id"] for r in rows] == [CID_A, CID_B]
        assert rows[0]["title"] == "Câu hỏi đầu của A"

    def test_tieu_de_dai_bi_cat_gon(self, fake_redis):
        from app.store import ConversationIndex

        index = ConversationIndex(fake_redis)
        index.touch("u1", CID_A, "  rất   dài " * 30)
        title = index.list("u1")[0]["title"]
        assert len(title) <= 60 and title.endswith("…") and "  " not in title

    def test_vuot_gioi_han_thi_bo_cuoc_cu_nhat(self, fake_redis):
        from app.store import ConversationIndex

        index = ConversationIndex(fake_redis, max_per_user=2)
        index.touch("u1", "conv-00000001", "1", now=1)
        index.touch("u1", "conv-00000002", "2", now=2)
        stale = index.touch("u1", "conv-00000003", "3", now=3)

        assert stale == ["conv-00000001"]
        assert [r["id"] for r in index.list("u1")] == ["conv-00000003", "conv-00000002"]

    def test_moi_user_danh_sach_rieng(self, fake_redis):
        from app.store import ConversationIndex

        index = ConversationIndex(fake_redis)
        index.touch("u1", CID_A, "của u1")
        assert index.list("u2") == []


class TestConversationApi:
    def test_moi_cuoc_co_lich_su_rieng(self, chat, auth_headers):
        assert ask(chat, auth_headers, "A1", CID_A)["history_length"] == 0
        assert ask(chat, auth_headers, "B1", CID_B)["history_length"] == 0
        second = ask(chat, auth_headers, "A2", CID_A)
        assert second["history_length"] == 2, "cuộc A chỉ thấy lịch sử của A"
        assert second["conversation_id"] == CID_A

    def test_khong_gui_conversation_id_van_nhu_cu(self, chat, auth_headers):
        ask(chat, auth_headers, "câu trong luồng cũ")
        assert ask(chat, auth_headers, "câu tiếp")["history_length"] == 2
        assert chat.get("/conversations", headers=auth_headers).json()["conversations"] == []

    def test_danh_sach_va_chi_tiet(self, chat, auth_headers):
        ask(chat, auth_headers, "Docker là gì?", CID_A)
        ask(chat, auth_headers, "Redis là gì?", CID_B)

        rows = chat.get("/conversations", headers=auth_headers).json()["conversations"]
        assert [(r["id"], r["title"]) for r in rows] == [(CID_B, "Redis là gì?"), (CID_A, "Docker là gì?")]

        detail = chat.get(f"/conversations/{CID_A}", headers=auth_headers).json()
        assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
        assert detail["messages"][0]["content"] == "Docker là gì?"

    def test_doi_ten(self, chat, auth_headers):
        ask(chat, auth_headers, "câu hỏi", CID_A)
        response = chat.patch(f"/conversations/{CID_A}", json={"title": "Tên mới"}, headers=auth_headers)
        assert response.status_code == 200
        assert chat.get("/conversations", headers=auth_headers).json()["conversations"][0]["title"] == "Tên mới"

    def test_xoa(self, chat, auth_headers, fake_redis):
        from app.store import ThreadStore

        ask(chat, auth_headers, "câu hỏi", CID_A)
        assert chat.delete(f"/conversations/{CID_A}", headers=auth_headers).status_code == 200
        assert chat.get("/conversations", headers=auth_headers).json()["conversations"] == []
        assert not fake_redis.exists(ThreadStore._key(ThreadStore.thread_id("sv-test", CID_A)))
        assert chat.get(f"/conversations/{CID_A}", headers=auth_headers).status_code == 404

    def test_khong_xem_duoc_cuoc_cua_user_khac(self, chat, auth_headers):
        ask(chat, auth_headers, "bí mật", CID_A)
        other = {**auth_headers, "X-User-Id": "nguoi-khac"}
        assert chat.get(f"/conversations/{CID_A}", headers=other).status_code == 404

    @pytest.mark.parametrize("bad", ["ngan", "co:hai-cham", "x" * 65])
    def test_conversation_id_sai_dinh_dang_thi_422(self, chat, auth_headers, bad):
        response = chat.post("/ask", json={"question": "hi", "conversation_id": bad}, headers=auth_headers)
        assert response.status_code == 422

    def test_can_api_key(self, chat):
        assert chat.get("/conversations").status_code == 401
        assert chat.get(f"/conversations/{CID_A}").status_code == 401
        assert chat.delete(f"/conversations/{CID_A}").status_code == 401
