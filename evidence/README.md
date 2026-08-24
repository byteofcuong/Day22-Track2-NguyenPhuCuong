# Bằng chứng nộp bài — Day 22: LangSmith + Prompt Versioning

**Học viên:** Nguyễn Phú Cường
**LangSmith project:** `day22-nguyenphucuong`
**Provider:** OpenAI (`gpt-4o-mini` + `text-embedding-3-small`)

---

## Danh sách tệp bằng chứng

| Tệp | Nhiệm vụ | Nội dung |
|---|---|---|
| `01_langsmith_traces.png` | 1 | Giao diện LangSmith hiển thị ≥ 50 traces `rag-query` |
| `01_rag_pipeline_log.txt` | 1 | Console log Bước 1 — 50/50 câu hỏi, 0 lỗi *(bổ sung)* |
| `02_prompt_hub.png` | 2 | Prompt Hub hiển thị 2 phiên bản `cuong-rag-v1` / `cuong-rag-v2` |
| `02_ab_routing_log.txt` | 2 | Console log A/B routing — 50 truy vấn có nhãn v1/v2 |
| `03_ragas_scores.png` | 3 | Terminal hiển thị bảng so sánh V1 vs V2 |
| `03_ragas_report.json` | 3 | Bản sao của `data/ragas_report.json` |
| `03_ragas_console.txt` | 3 | Console log đầy đủ Bước 3 *(bổ sung)* |
| `04_pii_demo_log.txt` | 4 | Console log PII detector — 6 test case |
| `04_json_demo_log.txt` | 4 | Console log JSON formatter — 5 test case |

---

## Kết quả từng nhiệm vụ

### Nhiệm vụ 1 — RAG Pipeline với LangSmith

- Knowledge base 30 KB → **107 chunks** (`chunk_size=500`, `chunk_overlap=50`), index bằng FAISS.
- RAG chain dựng bằng LCEL: `{"context": retriever | format_docs, "question": RunnablePassthrough()} | RAG_PROMPT | llm | StrOutputParser()`.
- Retriever nằm **bên trong** chain nên LangSmith ghi lại bước truy xuất như một child run —
  mỗi trace chứa đủ câu hỏi, context đã truy xuất và câu trả lời (tiêu chí 1.4).
- `@traceable(name="rag-query", tags=["rag", "step1"])`.
- **Kết quả: 50/50 traces, 0 lỗi.**

### Nhiệm vụ 2 — Prompt Hub & A/B Routing

- Hai prompt được push lên Hub:
  - `cuong-rag-v1` → https://smith.langchain.com/prompts/cuong-rag-v1/fe2e71c7
  - `cuong-rag-v2` → https://smith.langchain.com/prompts/cuong-rag-v2/8e67cc70
- Lúc chạy, **cả 2 prompt đều được pull thật từ Hub** (log ghi `[source=HUB]`, không phải
  local fallback) — xem dòng `Nguồn prompt: cuong-rag-v1=HUB | cuong-rag-v2=HUB`.
- Routing tất định bằng `int(hashlib.md5(request_id).hexdigest(), 16) % 2`.
  Dùng MD5 thay vì `hash()` built-in vì Python randomise hash của `str` theo từng process
  (`PYTHONHASHSEED`) — `hash()` sẽ cho kết quả khác nhau giữa các lần chạy, phá vỡ đúng tính
  chất tất định mà đề bài yêu cầu.
- Self-check in ngay trong log: mỗi `request_id` cho cùng một version 5/5 lần → PASS.
- **Phân bố: V1 = 19 câu, V2 = 31 câu (tổng 50).** Cả hai version đều nhận truy vấn.

### Nhiệm vụ 3 — RAGAS Evaluation

Xem `03_ragas_report.json` và phần **Phân tích V1 vs V2** bên dưới.

### Nhiệm vụ 4 — Guardrails AI Validators

**PII Detector** — `@register_validator(name="custom/pii-detector")`, phát hiện 4 loại PII
bằng regex (EMAIL, SSN, CREDIT_CARD, PHONE):

- **Kết quả: 5/6 test case bị redact**, case `Clean` giữ nguyên đúng như kỳ vọng.
- Thứ tự regex có chủ đích: SSN và CREDIT_CARD được xét **trước** PHONE, vì pattern PHONE
  cũng khớp một phần chuỗi số thẻ tín dụng — nếu để PHONE chạy trước, số thẻ bị dán nhãn sai.
- `\b` đặt **sau** `\(?` trong pattern PHONE, nếu không thì `(555) 867-5309` sẽ bị bỏ sót dấu
  mở ngoặc và để lại `([PHONE_REDACTED]`.

**JSON Formatter** — tự sửa 3 loại lỗi: gỡ markdown fences, đổi nháy đơn → nháy kép,
xoá dấu phẩy thừa.

- **Kết quả: 4/5 case cho ra JSON dùng được**, case `Truly invalid` rơi vào JSON lỗi dự phòng
  `{"error": "invalid_json", ...}` đúng yêu cầu tiêu chí 4.7.

> **Ghi chú kỹ thuật quan trọng:** với `guardrails-ai 0.11`, `PassResult(value_override=...)`
> **không có tác dụng** — `Validator.override_value_on_pass` mặc định là `False`
> (`guardrails/validator_base.py:98`) nên đầu ra trả về nguyên văn input, tức là PII **không hề
> bị che** dù validator log báo đã redact. Cách đúng về mặt ngữ nghĩa Guardrails: phát hiện PII
> **là** một validation failure, và `OnFailAction.FIX` chính là cơ chế thay đầu ra bằng
> `fix_value`. Cả hai validator đều dùng `FailResult(fix_value=...)`, đã kiểm chứng bằng thực
> nghiệm cho ra `validated_output` đã redact thật.

---

## Phân tích V1 vs V2

*(điền sau khi Bước 3 hoàn tất)*

---

## Cách tái tạo kết quả

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
cp .env.example .env      # điền LANGCHAIN_API_KEY + OPENAI_API_KEY
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8   # bắt buộc trên Windows (script in emoji)
cd src && python run_all.py
```
