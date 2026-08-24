"""
Retry với exponential backoff cho lỗi rate-limit tạm thời (429 / quota exceeded).

Lý do cần file này: free tier của Gemini giới hạn ~100 request/phút cho cả chat
và embedding. Batch embedding 107 chunks của knowledge base một lúc là đủ để vượt
ngưỡng đó ngay lần build vectorstore đầu tiên. Đây không phải lỗi trong code của
lab — nó là giới hạn thật của nhà cung cấp — nên cách xử lý đúng là retry có chờ,
không phải sửa logic RAG.

Cách dùng:
    from utils.retry import call_with_backoff
    result = call_with_backoff(lambda: llm.invoke(question))
"""
import time


TRANSIENT_MARKERS = (
    "429", "quota", "rate limit", "rate_limit", "resourceexhausted",
    "resource exhausted", "503", "unavailable", "internal error",
)


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in TRANSIENT_MARKERS)


def call_with_backoff(fn, max_attempts: int = 6, base_delay: float = 8.0,
                       max_delay: float = 60.0, label: str = ""):
    """
    Gọi fn() (không tham số — dùng lambda/closure để bọc call thật). Nếu lỗi có
    dấu hiệu rate-limit/tạm thời, chờ theo exponential backoff rồi thử lại. Lỗi
    không thuộc dạng tạm thời (auth sai, input sai, ...) được raise ngay lập tức
    để không lãng phí thời gian retry vô ích.
    """
    delay = base_delay
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not _is_transient(e) or attempt == max_attempts:
                raise
            tag = f" [{label}]" if label else ""
            print(f"  ⏳ Rate limit{tag} (lần {attempt}/{max_attempts}), "
                  f"chờ {delay:.0f}s rồi thử lại... ({str(e)[:80]})")
            time.sleep(delay)
            delay = min(delay * 2, max_delay)

    raise last_exc
