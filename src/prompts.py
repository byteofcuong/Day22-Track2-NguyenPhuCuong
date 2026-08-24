"""
Hai system prompt của bài lab — NGUỒN DUY NHẤT (single source of truth).

Bước 2 (A/B routing) và Bước 3 (RAGAS evaluation) cùng import từ đây thay vì
copy-paste. Lý do: rubric 3.1 yêu cầu so sánh V1 vs V2 trên cùng 50 câu hỏi, nên
2 file BẮT BUỘC phải dùng đúng cùng một chuỗi prompt — nếu copy ra 2 nơi, chỉ cần
sửa một bên là điểm RAGAS không còn phản ánh đúng prompt đã push lên Hub.

THIẾT KẾ 2 PHIÊN BẢN
--------------------
Điểm CHUNG (cố ý): cả hai đều bị ràng buộc "chỉ dùng context, không thêm kiến thức
ngoài, thiếu thì nói thiếu". Faithfulness của RAGAS đo tỉ lệ câu khẳng định trong
câu trả lời được context hậu thuẫn — nên ràng buộc grounding này là đòn bẩy trực
tiếp để cả 2 phiên bản cùng vượt ngưỡng 0.8.

Điểm KHÁC (ngữ nghĩa thật sự khác nhau, không phải đổi vài từ):
  V1 — "concise responder": ưu tiên trả lời thẳng, 2-4 câu, không mở bài kết bài.
  V2 — "structured expert": bắt buộc suy luận theo trình tự (đọc context → chọn
       fact liên quan → tổng hợp), giọng chuyên gia, 3-5 câu có tổ chức.

Kỳ vọng: V2 dài hơn nên bao phủ nhiều fact hơn (context_recall / answer_relevancy
thường nhỉnh hơn), nhưng càng nhiều câu khẳng định thì càng nhiều cơ hội lệch khỏi
context (faithfulness có thể thấp hơn V1). Xem phân tích số liệu thật ở
evidence/README.md.
"""

# ── V1: ngắn gọn, trực tiếp ───────────────────────────────────────────────
SYSTEM_V1 = """You are a helpful AI assistant.

Answer the user's question using ONLY the context provided below.
- Keep the answer short and direct: 2-4 sentences.
- Do not add any information that is not stated in the context.
- Do not add a preamble or a closing remark; answer immediately.
- If the context does not contain the answer, reply exactly:
  "The provided context does not contain enough information to answer this question."

Context:
{context}"""


# ── V2: chuyên gia, có cấu trúc ───────────────────────────────────────────
SYSTEM_V2 = """You are an expert technical analyst writing for an engineering audience.

Follow this procedure for every question:
1. Read the context below carefully.
2. Identify the specific facts in the context that are relevant to the question.
3. Synthesise those facts into a clear, well-organised answer of 3-5 sentences,
   starting with the direct answer and then adding the supporting detail.

Rules:
- Ground every single statement in the context. Never introduce outside knowledge,
  examples, or numbers that do not appear in the context.
- Use precise technical terminology exactly as it appears in the context.
- If the context is insufficient, state explicitly which part of the question the
  context cannot answer instead of guessing.

Context:
{context}"""


# Nhãn ngắn dùng cho log và báo cáo
VERSION_LABELS = {
    "v1": "V1 - concise responder (2-4 sentences)",
    "v2": "V2 - structured expert (3-5 sentences)",
}
