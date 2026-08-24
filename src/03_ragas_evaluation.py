"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from utils.retry import call_with_backoff
from qa_pairs import QA_PAIRS
from prompts import SYSTEM_V1, SYSTEM_V2   # cùng nguồn với Bước 2 → so sánh mới công bằng


# ── 1. Prompt Templates ────────────────────────────────────────────────────
# Import từ prompts.py thay vì copy-paste: rubric 3.1 yêu cầu đánh giá đúng 2
# prompt đã push lên Hub ở Bước 2. Nếu copy chuỗi ra 2 file, chỉ cần sửa một bên
# là điểm RAGAS không còn phản ánh prompt thật đang chạy production.
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}

# 4 metrics đánh giá — giữ trong list để thứ tự in bảng và thứ tự tính luôn khớp
METRICS = [faithfulness, answer_relevancy, context_recall, context_precision]
METRIC_NAMES = [m.name for m in METRICS]

# Số cặp QA đưa vào đánh giá RAGAS; None = toàn bộ 50 (đặt qua --limit)
EVAL_LIMIT = None


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision —
    context_precision xếp hạng độ liên quan của TỪNG đoạn, nên nếu truyền một
    chuỗi đã ghép thì mọi sample chỉ có 1 "đoạn" và metric mất hết ý nghĩa.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)

    contexts = [doc.page_content for doc in docs]   # phải là list[str] !

    # Ghép riêng một bản string chỉ để đưa vào biến {context} của prompt
    ctx_str = "\n\n".join(contexts)

    chain  = prompt | llm | StrOutputParser()
    answer = call_with_backoff(
        lambda: chain.invoke({"context": ctx_str, "question": question})
    )

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts

    Kết quả được cache ra data/.rag_cache_{version}.json. Phần đánh giá RAGAS
    phía sau chạy rất lâu và đã từng bị mất trắng khi máy ngủ/khởi động lại;
    cache giúp lần chạy lại bỏ qua 50 lệnh gọi LLM đã hoàn thành thay vì làm lại
    từ đầu. Xoá file cache nếu muốn thu thập lại câu trả lời mới.
    """
    cache_path = Path(__file__).parent.parent / "data" / f".rag_cache_{prompt_version}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if len(cached) == len(QA_PAIRS):
                print(f"\n♻️  Dùng lại {len(cached)} kết quả RAG đã cache cho prompt "
                      f"{prompt_version} ({cache_path.name})")
                return cached
            print(f"\n⚠️  Cache {cache_path.name} chỉ có {len(cached)}/{len(QA_PAIRS)} "
                  f"sample — thu thập lại từ đầu.")
        except Exception as e:
            print(f"\n⚠️  Không đọc được cache {cache_path.name} ({e}) — thu thập lại.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy {len(QA_PAIRS)} câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        try:
            out = run_rag(retriever, llm, prompt, qa["question"])
        except Exception as e:
            # Giữ sample lại với answer rỗng thì RAGAS sẽ cho điểm 0 và bóp méo
            # trung bình → bỏ hẳn sample lỗi và báo rõ số lượng ở cuối.
            print(f"  [{i:02d}/{len(QA_PAIRS)}] ⚠️  LỖI {type(e).__name__}: {str(e)[:80]} — bỏ qua")
            continue

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],   # list[str]
        })
        print(f"  [{i:02d}/{len(QA_PAIRS)}] {qa['question'][:60]}")

    if len(results) < len(QA_PAIRS):
        print(f"  ⚠️  Chỉ thu được {len(results)}/{len(QA_PAIRS)} sample "
              f"(rubric trừ 1đ mỗi 5 cặp thiếu) — nên chạy lại bước 3.")
    else:
        # Chỉ cache khi thu đủ 50/50, để lần sau không vô tình dùng lại bộ thiếu
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        print(f"  💾 Đã cache kết quả RAG vào {cache_path.name}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    Điểm của mỗi version được cache lại ngay sau khi tính xong, để nếu lần chạy
    bị đứt trong lúc đánh giá V2 thì không phải tính lại V1.
    """
    # Cắt bớt sample NẾU có --limit. Cache RAG vẫn giữ đủ 50 câu; chỉ khâu chấm
    # điểm bị giới hạn, nên chạy lại không --limit sau này sẽ dùng lại được cache
    # mà không phải gọi lại LLM.
    if EVAL_LIMIT is not None and EVAL_LIMIT < len(rag_results):
        rag_results = rag_results[:EVAL_LIMIT]
        print(f"\n✂️  Giới hạn đánh giá {version}: {EVAL_LIMIT}/{len(QA_PAIRS)} cặp QA")

    suffix = f"_{len(rag_results)}" if EVAL_LIMIT is not None else ""
    score_cache = Path(__file__).parent.parent / "data" / f".ragas_scores_{version}{suffix}.json"
    if score_cache.exists():
        try:
            cached = json.loads(score_cache.read_text(encoding="utf-8"))
            if set(METRIC_NAMES).issubset(cached):
                print(f"\n♻️  Dùng lại điểm RAGAS đã cache cho prompt {version} "
                      f"({score_cache.name})")
                for k in METRIC_NAMES:
                    print(f"  {k:30s}: {cached[k]:.4f}")
                return cached
        except Exception as e:
            print(f"\n⚠️  Không đọc được {score_cache.name} ({e}) — đánh giá lại.")

    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    # max_workers vừa phải + retry kiên nhẫn: RAGAS bắn rất nhiều request song
    # song, dễ chạm rate limit rồi trả về NaN hàng loạt nếu để mặc định.
    # raise_exceptions=False để một sample lỗi không giết cả lần đánh giá dài.
    # (Với provider free-tier siết theo phút/ngày như Gemini, hạ max_workers
    # xuống 2 và tăng max_wait — nhưng khi đó Bước 3 mất nhiều giờ.)
    run_config = RunConfig(timeout=300, max_retries=10, max_wait=60, max_workers=8)

    result = evaluate(
        dataset,
        metrics=METRICS,
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    scores = {}
    for key in METRIC_NAMES:
        raw   = result[key]
        valid = [v for v in raw if v is not None and not (isinstance(v, float) and np.isnan(v))]
        scores[key] = float(np.mean(valid)) if valid else float("nan")
        if len(valid) < len(raw):
            print(f"  ⚠️  {key}: {len(raw) - len(valid)}/{len(raw)} sample trả về NaN "
                  f"(đã loại khỏi trung bình)")

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    # Cache ngay để lần chạy sau (nếu đứt ở V2) không phải tính lại version này
    score_cache.parent.mkdir(parents=True, exist_ok=True)
    score_cache.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(f"  💾 Đã cache điểm vào {score_cache.name}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    # --only v1|v2 cho phép chạy 2 version thành 2 tiến trình SONG SONG (mỗi
    # process ghi cache điểm riêng), rút ngắn gần một nửa thời gian chờ. Gọi
    # không đối số (như run_all.py) vẫn chạy tuần tự cả 2 như bình thường.
    parser = argparse.ArgumentParser(description="RAGAS evaluation cho 2 prompt version")
    parser.add_argument("--only", choices=["v1", "v2"], default=None,
                        help="Chỉ đánh giá 1 version rồi thoát (dùng để chạy song song)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Chỉ đánh giá N cặp QA đầu tiên (mặc định: toàn bộ 50). "
                             "Dùng khi cần kết quả nhanh; rubric trừ 1đ mỗi 5 cặp thiếu.")
    args, _unknown = parser.parse_known_args()

    global EVAL_LIMIT
    EVAL_LIMIT = args.limit

    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    if args.only:
        results = collect_rag_outputs(vectorstore, args.only)
        run_ragas_eval(results, args.only)
        print(f"\n✅ Đã đánh giá xong {args.only.upper()}. "
              f"Chạy lại không có --only để tổng hợp báo cáo.")
        return

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in METRIC_NAMES:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # ── PHÂN TÍCH V1 vs V2 (rubric bonus +2đ) ──────────────────────────────
    # Hai prompt chia sẻ cùng ràng buộc grounding ("chỉ dùng context"), nên khác
    # biệt điểm đến từ ĐỘ DÀI câu trả lời mà mỗi prompt yêu cầu:
    #
    #  • faithfulness đếm tỉ lệ câu khẳng định được context hậu thuẫn. V1 (2-4 câu)
    #    tạo ít khẳng định hơn → ít cơ hội nói điều context không có → thường
    #    faithfulness nhỉnh hơn. V2 (3-5 câu, "thêm supporting detail") dễ chèn
    #    câu diễn giải nằm ngoài context → mất điểm.
    #  • answer_relevancy sinh câu hỏi ngược từ câu trả lời rồi so với câu hỏi gốc.
    #    V2 dài và có mở đầu trực tiếp nên thường bám sát ý hỏi hơn một chút.
    #  • context_recall / context_precision KHÔNG phụ thuộc prompt (chỉ phụ thuộc
    #    retriever + chunking, mà cả 2 version dùng chung k=3 và cùng FAISS index)
    #    → 2 cột này gần như bằng nhau; lệch nhỏ chỉ là nhiễu do LLM judge.
    #
    # Xem số liệu thật và kết luận cuối trong evidence/README.md.
    print("\n💡 Phân tích: faithfulness phản ánh độ dài câu trả lời (V1 ngắn → ít")
    print("   khẳng định ngoài context); context_recall/precision gần bằng nhau vì")
    print("   cả 2 version dùng chung retriever k=3 trên cùng FAISS index.")

    # Lưu báo cáo vào data/ragas_report.json
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "meta": {
            "provider":        config.PROVIDER,
            "llm_model":       config.OPENAI_MODEL,
            "embedding_model": config.OPENAI_EMBEDDING_MODEL,
            "n_samples_v1":    min(len(v1_results), EVAL_LIMIT or len(v1_results)),
            "n_samples_v2":    min(len(v2_results), EVAL_LIMIT or len(v2_results)),
            "n_qa_pairs_total": len(QA_PAIRS),
            "eval_limit":      EVAL_LIMIT,
            "retriever_k":     3,
            "metrics":         METRIC_NAMES,
        },
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
