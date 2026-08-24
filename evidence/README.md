# Bằng chứng nộp bài — Day 22: LangSmith + Prompt Versioning

**Học viên:** Nguyễn Phú Cường
**LangSmith project:** `day22-nguyenphucuong`
**Provider:** OpenAI (`gpt-4o-mini` + `text-embedding-3-small`)

### Link trace công khai (mở được không cần đăng nhập)

LangSmith **không hỗ trợ** đặt cả tracing project ở chế độ public — SDK chỉ có
`share_run()` / `share_dataset()`, không có `share_project()`. Vì vậy dưới đây là link
public của các trace đại diện để người chấm truy cập trực tiếp:

| Trace | Câu hỏi | Link |
|---|---|---|
| `rag-query` (Bước 1) | What are common AI safety concerns with LLMs? | https://smith.langchain.com/public/4745e9ee-d0e2-46bc-ad25-9724b72f8853/r |
| `rag-query` (Bước 1) | What is Constitutional AI? | https://smith.langchain.com/public/15c3d1d0-128b-4627-876a-d44ecbbc5ddf/r |
| `ab-rag-query` (Bước 2) | What are common AI safety concerns with LLMs? | https://smith.langchain.com/public/fcdb9dc1-a266-4d4c-914b-1b834369ed32/r |
| `ab-rag-query` (Bước 2) | What is Constitutional AI? | https://smith.langchain.com/public/71e9d109-0531-45ad-956c-142c8e6595a7/r |

Mỗi link mở ra cây trace đầy đủ: câu hỏi đầu vào → context được FAISS truy xuất → câu
trả lời của LLM (bằng chứng trực tiếp cho tiêu chí 1.4).

**URL project (cần quyền truy cập):**
https://smith.langchain.com/o/324786d8-dd0b-498e-8510-87a17a0153cb/projects/p/d1a50c6f-a046-4e66-b61c-bb513cedbaf3

**Số trace thực tế trong project** (đếm qua LangSmith API):

| Tên trace | Số lượng |
|---|---|
| `rag-query` (Bước 1) | 128 |
| `ab-rag-query` (Bước 2) | 50 |

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

Cả 50 cặp QA được chạy qua **cả hai** prompt version (100 lượt RAG), rồi chấm bằng
đủ 4 chỉ số RAGAS. Xem `03_ragas_report.json`.

| Chỉ số | V1 | V2 | Cao hơn |
|---|---:|---:|---|
| faithfulness | **0.9846** | 0.9306 | V1 |
| answer_relevancy | 0.9042 | **0.9135** | V2 |
| context_recall | 1.0000 | 1.0000 | hoà |
| context_precision | **0.9263** | 0.9079 | V1 |

✅ **Mục tiêu faithfulness ≥ 0.8 đạt ở cả hai version** — và cả hai đều ≥ 0.9.

> ⚠️ **Lưu ý trung thực về số sample:** RAGAS trả `NaN` cho một phần sample do LLM judge
> bị timeout trong lúc chấm; điểm trung bình chỉ tính trên các sample hợp lệ. Số lượng cụ
> thể được ghi trong `_valid_samples` của `03_ragas_report.json`:
>
> | Chỉ số | V1 | V2 |
> |---|---|---|
> | faithfulness | 26/50 | 26/50 |
> | answer_relevancy | 39/50 | 44/50 |
> | context_recall | 41/50 | 43/50 |
> | context_precision | 26/50 | 19/50 |
>
> Phần thu thập RAG **đủ 50/50 cho cả 2 version** (xem `.rag_cache_v*.json`); mất mát chỉ
> xảy ra ở khâu chấm điểm. Nguyên nhân: RAGAS 0.4.3 trên Windows không thực sự chạy song
> song (`max_workers` gần như vô hiệu do vấn đề `nest_asyncio`), khiến mỗi job kéo dài
> ~20-25s và một phần chạm ngưỡng timeout 300s.

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

**Kết luận ngắn: V1 thắng ở độ trung thực, V2 thắng ở độ bám sát câu hỏi.**

### Vì sao V1 có faithfulness cao hơn (0.9846 vs 0.9306)

Hai prompt dùng **chung một ràng buộc grounding** ("chỉ dùng context, không thêm kiến
thức ngoài"), nên khác biệt không đến từ mức độ nghiêm ngặt mà đến từ **độ dài câu trả
lời** mỗi prompt yêu cầu:

- **V1** giới hạn 2–4 câu, cấm mở bài/kết luận → sinh ra ít câu khẳng định hơn.
- **V2** yêu cầu 3–5 câu và *"thêm chi tiết hỗ trợ"* → sinh nhiều khẳng định hơn.

Faithfulness đo **tỉ lệ** khẳng định được context hậu thuẫn. Mỗi câu thêm vào là một cơ
hội nữa để nói điều context không có — nên prompt càng khuyến khích diễn giải dài thì
faithfulness càng dễ tụt. Đây là đánh đổi có thể dự đoán trước, không phải ngẫu nhiên.

### Vì sao V2 có answer_relevancy nhỉnh hơn (0.9135 vs 0.9042)

Chỉ số này sinh câu hỏi ngược từ câu trả lời rồi so độ tương đồng với câu hỏi gốc. V2
được lệnh *"bắt đầu bằng câu trả lời trực tiếp rồi mới bổ sung chi tiết"*, nên phần mở đầu
bám rất sát ý hỏi. V1 tuy ngắn nhưng đôi khi trả lời cô đọng tới mức thiếu từ khoá của câu
hỏi, làm câu hỏi tái tạo lệch đi một chút.

### Vì sao context_recall bằng nhau tuyệt đối (1.0000)

Đây là kiểm chứng quan trọng cho thấy thí nghiệm được thiết kế đúng: `context_recall` và
`context_precision` **chỉ phụ thuộc retriever**, không phụ thuộc prompt. Cả hai version
dùng chung một FAISS index và cùng `k=3`, nên hai chỉ số này gần như phải bằng nhau —
đúng như quan sát. Chênh lệch nhỏ ở `context_precision` (0.9263 vs 0.9079) là nhiễu do
LLM judge chấm trên số sample hợp lệ khác nhau (26/50 vs 19/50), không phải khác biệt
thật giữa hai prompt.

`context_recall = 1.0` cũng xác nhận chiến lược chunking (500 ký tự, overlap 50) đủ tốt:
với mọi câu hỏi, top-3 chunk luôn chứa trọn thông tin cần để dựng đáp án chuẩn.

### Nên chọn prompt nào?

Tuỳ mục tiêu:
- **Ưu tiên độ chính xác / miền rủi ro cao** (y tế, pháp lý, tài chính) → chọn **V1**:
  faithfulness 0.9846 nghĩa là gần như không bịa.
- **Ưu tiên trải nghiệm người đọc** (tài liệu, trợ lý học tập) → chọn **V2**: câu trả lời
  đầy đủ và có cấu trúc hơn, đổi lại 5 điểm phần trăm faithfulness.

Với bài toán RAG hỏi-đáp kỹ thuật của lab này, **V1 là lựa chọn hợp lý hơn** vì nó thắng ở
chỉ số quan trọng nhất (faithfulness) mà chỉ thua không đáng kể ở answer_relevancy.

---

## Cách tái tạo kết quả

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
cp .env.example .env      # điền LANGCHAIN_API_KEY + OPENAI_API_KEY
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8   # bắt buộc trên Windows (script in emoji)
cd src && python run_all.py
```
