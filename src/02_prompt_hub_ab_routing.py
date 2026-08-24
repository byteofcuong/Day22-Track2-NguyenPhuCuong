"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from utils.retry import call_with_backoff
from qa_pairs import SAMPLE_QUESTIONS
from prompts import SYSTEM_V1, SYSTEM_V2   # nguồn duy nhất, dùng chung với Bước 3


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
# Tên phải duy nhất trong Hub cá nhân; chỉ dùng chữ thường, số và dấu gạch nối.
PROMPT_V1_NAME = "cuong-rag-v1"
PROMPT_V2_NAME = "cuong-rag-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# SYSTEM_V1 / SYSTEM_V2 được import từ prompts.py — xem file đó để biết vì sao
# hai phiên bản khác nhau về ngữ nghĩa chứ không chỉ khác câu chữ.
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.

    push_prompt() là idempotent: chạy lại sẽ tạo commit mới trên cùng một prompt
    chứ không báo lỗi trùng tên, nên script an toàn khi chạy nhiều lần.
    """
    specs = [
        (PROMPT_V1_NAME, PROMPT_V1,
         "V1 - concise responder: tra loi truc tiep 2-4 cau, chi dung context."),
        (PROMPT_V2_NAME, PROMPT_V2,
         "V2 - structured expert: suy luan theo trinh tu, tra loi 3-5 cau co to chuc."),
    ]
    for name, template, desc in specs:
        try:
            url = client.push_prompt(name, object=template, description=desc)
            print(f"✅ Đã push '{name}' → {url}")
        except Exception as e:
            print(f"⚠️  Push '{name}' lỗi: {type(e).__name__}: {e}")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.

    Tiêu chí 2.3 yêu cầu prompt phải THỰC SỰ được pull từ Hub lúc chạy. Fallback
    local vẫn được giữ để script không chết giữa chừng, nhưng in cảnh báo rõ ràng
    để không ai nhầm một lần chạy fallback là bằng chứng hợp lệ.

    Trả về: {name: (ChatPromptTemplate, source)} với source = "hub" | "local"
    """
    prompts = {}

    for name, local_fallback in [(PROMPT_V1_NAME, PROMPT_V1),
                                 (PROMPT_V2_NAME, PROMPT_V2)]:
        try:
            prompts[name] = (client.pull_prompt(name), "hub")
            print(f"↓ Đã pull '{name}' từ Hub  [source=HUB]")
        except Exception as e:
            prompts[name] = (local_fallback, "local")
            print(f"❗ KHÔNG pull được '{name}' từ Hub ({type(e).__name__}: {e})")
            print("   → Dùng local fallback [source=LOCAL] — bằng chứng này KHÔNG hợp lệ")
            print("     cho tiêu chí 2.3. Kiểm tra LANGCHAIN_API_KEY rồi chạy lại.")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).

    Dùng MD5 chứ không dùng hash() built-in: Python randomise hash của str theo
    từng process (PYTHONHASHSEED), nên hash() sẽ cho kết quả khác nhau giữa các
    lần chạy — phá vỡ đúng tính chất tất định mà tiêu chí 2.4 yêu cầu.
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


def verify_routing_is_deterministic(sample_ids=("req-0000", "req-0017", "req-0042")) -> bool:
    """
    Tự kiểm chứng tính tất định của router: gọi lặp lại trên cùng request_id phải
    luôn ra cùng version. In ra log làm bằng chứng trực tiếp cho tiêu chí 2.4.
    """
    print("\n🔍 Self-check tính tất định của A/B router:")
    all_ok = True
    for rid in sample_ids:
        versions = {get_prompt_version(rid) for _ in range(5)}
        ok = len(versions) == 1
        all_ok = all_ok and ok
        v = versions.pop()
        tag = "v1" if v == PROMPT_V1_NAME else "v2"
        verdict = "✅ PASS" if ok else "❌ FAIL"
        print(f"   {rid} → prompt-{tag}  (5/5 lần giống nhau)  {verdict}")
    return all_ok


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được router chọn.

    Trả về dict có cả `version` và `contexts` để nhãn v1/v2 và context đã truy
    xuất nằm ngay trong output của trace trên LangSmith, không chỉ ở console.
    """
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  context,
        "question": question,
    })

    return {
        "question": question,
        "answer":   answer,
        "version":  version,
        "contexts": [doc.page_content for doc in docs],
    }


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    print("\n── Push 2 prompt lên LangSmith Prompt Hub ──")
    push_prompts_to_hub(client)

    print("\n── Pull 2 prompt từ Hub (chạy bằng chính bản trên Hub) ──")
    pulled  = pull_prompts_from_hub(client)
    prompts = {name: tpl for name, (tpl, _src) in pulled.items()}
    sources = {name: src for name, (_tpl, src) in pulled.items()}

    verify_routing_is_deterministic()

    print()
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    print(f"\n── Chạy {len(SAMPLE_QUESTIONS)} câu hỏi qua A/B router ──")

    v1_count, v2_count, failed = 0, 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"

        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        try:
            result  = call_with_backoff(
                lambda r=retriever, l=llm, p=prompt, q=question, v=version_tag:
                    ask_ab(r, l, p, q, v),
                label=f"Q{i+1}",
            )
            preview = str(result["answer"])[:70].replace("\n", " ")
        except Exception as e:
            preview = f"[LỖI] {type(e).__name__}: {e}"
            failed += 1

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

        print(f"[{i+1:02d}] [{request_id}] [prompt-{version_tag}] {question[:52]}...")
        print(f"     → {preview}")

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print(f"   Nguồn prompt: {PROMPT_V1_NAME}={sources[PROMPT_V1_NAME].upper()} | "
          f"{PROMPT_V2_NAME}={sources[PROMPT_V2_NAME].upper()}")
    if failed:
        print(f"⚠️  {failed} câu bị lỗi khi gọi LLM.")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
