"""Giữ test chạy offline với mock LLM, kể cả khi .env có OPENROUTER_API_KEY.

pytest nạp file này trước ``tests/conftest.py``. Đặt biến rỗng ở đây thì
``load_dotenv`` (không ghi đè biến đã có) và pydantic-settings (ưu tiên biến
môi trường hơn file .env) đều bỏ qua key trong .env — test không gọi
OpenRouter thật, không tốn tiền và không phụ thuộc rate limit của provider.
"""

import os

os.environ["OPENROUTER_API_KEY"] = ""
