"""
Tiện ích để tải và xử lý dữ liệu cho RAG pipeline.

Cách dùng:
    from utils.data_loader import load_knowledge_base, split_text, build_vectorstore

    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks, embeddings)
"""
import time
from pathlib import Path


def load_knowledge_base(path: str = None) -> str:
    """
    Đọc file knowledge base và trả về nội dung dạng chuỗi.

    Args:
        path: đường dẫn tới file text.
              Mặc định: data/knowledge_base.txt (thư mục gốc của project)

    Returns:
        Nội dung file dưới dạng str
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.txt"
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chia văn bản thành các đoạn nhỏ (chunks) để index.

    Dùng RecursiveCharacterTextSplitter — tách ưu tiên theo đoạn văn, câu, rồi ký tự.

    Args:
        text         : văn bản cần chia
        chunk_size   : số ký tự tối đa mỗi chunk (mặc định: 500)
        chunk_overlap: số ký tự chồng lên nhau giữa 2 chunks liên tiếp (mặc định: 50)

    Returns:
        list[str] — danh sách các chuỗi chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list, embeddings, batch_size: int = 20, pause_sec: float = 3.0):
    """
    Tạo FAISS vectorstore từ danh sách chunks và embeddings.

    Args:
        chunks    : list[str] — danh sách text chunks đã chia
        embeddings: Embeddings instance (từ get_embeddings())
        batch_size: số chunk embed mỗi lần gọi API
        pause_sec : thời gian nghỉ giữa các batch (giây)

    Returns:
        FAISS vectorstore đã được index và sẵn sàng dùng để retrieve

    Lưu ý: FAISS.from_texts() gọi embeddings.embed_documents(chunks) trong MỘT
    lần duy nhất. Với các provider free-tier có giới hạn request/phút thấp
    (vd. Gemini free tier: ~100 request/phút), việc dồn toàn bộ chunks vào 1-2
    batch lớn (langchain_google_genai mặc định batch_size=100) sẽ vượt ngưỡng
    ngay ở lần build đầu tiên. Ở đây embed theo batch nhỏ + nghỉ giữa các batch
    + retry khi gặp lỗi tạm thời, để hoạt động ổn định với mọi provider mà
    không cần biết trước giới hạn cụ thể của từng nơi.
    """
    from langchain_community.vectorstores import FAISS
    from utils.retry import call_with_backoff

    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks "
          f"(batch={batch_size}) ...")

    all_vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = call_with_backoff(
            lambda b=batch: embeddings.embed_documents(b),
            label=f"embed batch {i // batch_size + 1}",
        )
        all_vectors.extend(vectors)
        print(f"  … đã embed {len(all_vectors)}/{len(chunks)} chunks")
        if i + batch_size < len(chunks):
            time.sleep(pause_sec)

    text_embeddings = list(zip(chunks, all_vectors))
    vectorstore = FAISS.from_embeddings(text_embeddings, embeddings)
    print("✅ FAISS vectorstore đã sẵn sàng.")
    return vectorstore
