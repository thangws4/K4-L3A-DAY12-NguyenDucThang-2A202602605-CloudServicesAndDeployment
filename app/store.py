"""CP4 — Stateless: state sống ngoài process.

Nếu lịch sử hội thoại nằm trong một dict trong RAM, thì khi scale lên 3
instance, user hỏi câu 1 vào instance A và câu 2 vào instance B sẽ thấy agent
"mất trí nhớ". Container còn bị restart bất cứ lúc nào. Vì vậy state phải
nằm ở nơi mọi instance cùng nhìn thấy: Redis.
"""

from __future__ import annotations

import json
import time

import redis

from .config import get_settings

HISTORY_MAX_MESSAGES = 20
HISTORY_TTL_SECONDS = 7 * 24 * 3600
CONVERSATIONS_MAX_PER_USER = 50


def get_redis_client(url: str | None = None):
    """CHO SẴN — tạo client Redis từ URL.

    ``fake://`` trả về Redis giả chạy trong RAM, dùng khi máy bạn chưa có
    Docker. Tiện cho lúc học, nhưng KHÔNG dùng khi deploy: nó vẫn là state
    trong process, đúng cái mà CP4 đang tìm cách loại bỏ.
    """
    url = url or get_settings().redis_url
    if url.startswith("fake://"):
        import fakeredis

        return fakeredis.FakeRedis(decode_responses=True)
    return redis.from_url(url, decode_responses=True)


class ConversationStore:
    """Lưu lịch sử hội thoại của từng user trong Redis List."""

    def __init__(self, client) -> None:
        self.client = client

    @staticmethod
    def _key(user_id: str) -> str:
        """CHO SẴN."""
        return f"history:{user_id}"

    def ping(self) -> bool:
        """Redis có trả lời không? Dùng cho endpoint /ready.

        TODO (CP4): gọi ``self.client.ping()`` trong try/except.
        Trả ``True`` nếu thành công, ``False`` nếu có bất kỳ Exception nào
        (mất mạng, sai mật khẩu, Redis chưa khởi động...).
        """
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def append(self, user_id: str, role: str, content: str) -> None:
        """Ghi thêm một lượt vào lịch sử.

        TODO (CP4):
          1. ``self.client.rpush(key, json.dumps({"role": role, "content": content},
             ensure_ascii=False))``
          2. ``self.client.ltrim(key, -HISTORY_MAX_MESSAGES, -1)`` — chỉ giữ
             ``HISTORY_MAX_MESSAGES`` message gần nhất, nếu không prompt sẽ
             phình vô hạn và tiền token cũng vậy.
          3. ``self.client.expire(key, HISTORY_TTL_SECONDS)`` — hội thoại cũ
             tự hết hạn, khỏi phải dọn tay.
        """
        key = self._key(user_id)
        message = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        self.client.rpush(key, message)
        self.client.ltrim(key, -HISTORY_MAX_MESSAGES, -1)
        self.client.expire(key, HISTORY_TTL_SECONDS)

    def get_history(self, user_id: str) -> list[dict]:
        """Đọc lịch sử hội thoại, cũ nhất trước.

        TODO (CP4): ``self.client.lrange(key, 0, -1)`` rồi ``json.loads``
        từng phần tử. Chưa có gì → trả về list rỗng.
        """
        raw = self.client.lrange(self._key(user_id), 0, -1)
        return [json.loads(item) for item in raw]

    def clear(self, user_id: str) -> None:
        """CHO SẴN — xóa lịch sử của một user."""
        self.client.delete(self._key(user_id))


class ThreadStore(ConversationStore):
    """Lịch sử của từng cuộc trò chuyện (một user có nhiều cuộc).

    Khóa ``thread:<user_id>:<conversation_id>`` tách biệt với ``history:<user_id>``
    của ConversationStore, nên luồng cũ (không có conversation_id) không đổi.
    conversation_id không chứa ``:`` nên hai cặp (user, cuộc) khác nhau không
    bao giờ trùng khóa.
    """

    @staticmethod
    def thread_id(user_id: str, conversation_id: str) -> str:
        return f"{user_id}:{conversation_id}"

    @staticmethod
    def _key(thread_id: str) -> str:
        return f"thread:{thread_id}"


class ConversationIndex:
    """Danh mục các cuộc trò chuyện của một user — nguồn dữ liệu cho thanh lịch sử.

    - ``conversations:<user_id>``      ZSET conversation_id → thời điểm cập nhật
    - ``conversation_titles:<user_id>`` HASH conversation_id → tiêu đề
    """

    def __init__(self, client, max_per_user: int = CONVERSATIONS_MAX_PER_USER) -> None:
        self.client = client
        self.max_per_user = max_per_user

    @staticmethod
    def _list_key(user_id: str) -> str:
        return f"conversations:{user_id}"

    @staticmethod
    def _title_key(user_id: str) -> str:
        return f"conversation_titles:{user_id}"

    @staticmethod
    def make_title(text: str, limit: int = 60) -> str:
        title = " ".join(text.split())
        return title if len(title) <= limit else title[: limit - 1].rstrip() + "…"

    def touch(self, user_id: str, conversation_id: str, title: str, now: float | None = None) -> list[str]:
        """Đưa cuộc trò chuyện lên đầu danh sách (tạo mới nếu chưa có).

        Trả về các conversation_id cũ bị đẩy ra khỏi danh sách vì vượt
        ``max_per_user`` — caller xóa luôn lịch sử của chúng.
        """
        now = now if now is not None else time.time()
        list_key, title_key = self._list_key(user_id), self._title_key(user_id)
        self.client.hsetnx(title_key, conversation_id, self.make_title(title))
        self.client.zadd(list_key, {conversation_id: now})
        stale = self.client.zrange(list_key, 0, -(self.max_per_user + 1))
        if stale:
            self.client.zrem(list_key, *stale)
            self.client.hdel(title_key, *stale)
        self.client.expire(list_key, HISTORY_TTL_SECONDS)
        self.client.expire(title_key, HISTORY_TTL_SECONDS)
        return list(stale)

    def list(self, user_id: str) -> list[dict]:
        """Các cuộc trò chuyện, mới cập nhật nhất trước."""
        rows = self.client.zrevrange(self._list_key(user_id), 0, self.max_per_user - 1, withscores=True)
        if not rows:
            return []
        ids = [conversation_id for conversation_id, _ in rows]
        titles = self.client.hmget(self._title_key(user_id), ids)
        return [
            {"id": conversation_id, "title": title or "Cuộc trò chuyện", "updated_at": score}
            for (conversation_id, score), title in zip(rows, titles)
        ]

    def get(self, user_id: str, conversation_id: str) -> dict | None:
        score = self.client.zscore(self._list_key(user_id), conversation_id)
        if score is None:
            return None
        title = self.client.hget(self._title_key(user_id), conversation_id)
        return {"id": conversation_id, "title": title or "Cuộc trò chuyện", "updated_at": score}

    def rename(self, user_id: str, conversation_id: str, title: str) -> bool:
        if self.get(user_id, conversation_id) is None:
            return False
        self.client.hset(self._title_key(user_id), conversation_id, self.make_title(title))
        return True

    def remove(self, user_id: str, conversation_id: str) -> bool:
        removed = self.client.zrem(self._list_key(user_id), conversation_id)
        self.client.hdel(self._title_key(user_id), conversation_id)
        return bool(removed)
