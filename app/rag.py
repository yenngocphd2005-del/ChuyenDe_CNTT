"""
Module RAG - Retrieval-Augmented Generation
Xây dựng FAISS index từ dữ liệu Knowledge/FAQ và tìm kiếm ngữ nghĩa.
"""
import csv
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Lazy-loaded globals
_embedding_model = None
_faiss_index = None
_documents: list[dict] = []


def _load_embedding_model():
    """Load mô hình embedding từ đường dẫn local."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    from sentence_transformers import SentenceTransformer

    model_path = settings.EMBEDDING_MODEL_PATH
    logger.info(f"Đang tải mô hình embedding từ: {model_path}")
    _embedding_model = SentenceTransformer(model_path)
    logger.info("Đã tải mô hình embedding thành công.")
    return _embedding_model


def _load_knowledge_csv(filepath: str) -> list[dict]:
    """Đọc file knowledge CSV."""
    documents = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = {
                "id": row.get("knowledge_id", ""),
                "type": "knowledge",
                "title": row.get("title", ""),
                "content": row.get("content", ""),
                "source_note": row.get("source_note", ""),
                "language": row.get("language", "vi"),
                "data_status": row.get("data_status", ""),
            }
            # Tạo text để embedding (kết hợp title + content)
            doc["embedding_text"] = f"{doc['title']}: {doc['content']}"
            documents.append(doc)
    logger.info(f"Đã đọc {len(documents)} đoạn kiến thức từ {filepath}")
    return documents


def _load_faq_csv(filepath: str) -> list[dict]:
    """Đọc file FAQ CSV."""
    documents = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = {
                "id": row.get("faq_id", ""),
                "type": "faq",
                "title": row.get("question", ""),
                "content": row.get("answer", ""),
                "source_note": "",
                "language": "vi",
                "data_status": row.get("data_status", ""),
            }
            # Embedding text: câu hỏi + câu trả lời
            doc["embedding_text"] = f"{doc['title']} {doc['content']}"
            documents.append(doc)
    logger.info(f"Đã đọc {len(documents)} câu hỏi FAQ từ {filepath}")
    return documents


def build_index(force_rebuild: bool = False) -> bool:
    """
    Xây dựng FAISS index từ dữ liệu Knowledge và FAQ.
    Nếu index đã tồn tại và force_rebuild=False, sẽ load từ file.
    """
    global _faiss_index, _documents

    import faiss

    index_dir = Path(settings.FAISS_INDEX_PATH)
    index_file = index_dir / "index.faiss"
    docs_file = index_dir / "documents.pkl"

    # Kiểm tra nếu index đã tồn tại
    if not force_rebuild and index_file.exists() and docs_file.exists():
        logger.info("Đang tải FAISS index từ file đã lưu...")
        _faiss_index = faiss.read_index(str(index_file))
        with open(docs_file, "rb") as f:
            _documents = pickle.load(f)
        logger.info(f"Đã tải index với {_faiss_index.ntotal} vectors, {len(_documents)} documents.")
        return True

    # Xây dựng index mới
    logger.info("Đang xây dựng FAISS index mới...")

    # Đọc dữ liệu
    base_dir = Path(settings.LLM_MODEL_PATH).parent.parent.parent
    bo_sung_path = base_dir / "DuLieu" / "cu_chi_knowledge_bo_sung_lien_he.csv"
    if bo_sung_path.exists():
        knowledge_path = bo_sung_path
        logger.info(f"Sử dụng bộ dữ liệu kiến thức bổ sung: {knowledge_path.name}")
    else:
        knowledge_path = base_dir / "DuLieu" / "cu_chi_knowledge.csv"
    faq_path = base_dir / "DuLieu" / "cu_chi_faq.csv"

    _documents = []
    if knowledge_path.exists():
        _documents.extend(_load_knowledge_csv(str(knowledge_path)))
    else:
        logger.warning(f"Không tìm thấy file knowledge: {knowledge_path}")

    if faq_path.exists():
        _documents.extend(_load_faq_csv(str(faq_path)))
    else:
        logger.warning(f"Không tìm thấy file FAQ: {faq_path}")

    if not _documents:
        logger.error("Không có dữ liệu để xây dựng index!")
        return False

    # Tạo embeddings
    model = _load_embedding_model()
    texts = [doc["embedding_text"] for doc in _documents]
    logger.info(f"Đang tạo embeddings cho {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Xây dựng FAISS index (Inner Product vì đã normalize → cosine similarity)
    dimension = embeddings.shape[1]
    _faiss_index = faiss.IndexFlatIP(dimension)
    _faiss_index.add(embeddings)

    # Lưu index
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(_faiss_index, str(index_file))
    with open(docs_file, "wb") as f:
        pickle.dump(_documents, f)

    logger.info(f"Đã xây dựng FAISS index: {_faiss_index.ntotal} vectors, dimension={dimension}")
    return True


def search(
    query: str,
    top_k: int = None,
    doc_type: Optional[str] = None,
) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa trong FAISS index.

    Args:
        query: Câu hỏi cần tìm kiếm.
        top_k: Số kết quả trả về tối đa.
        doc_type: Lọc theo loại ('knowledge', 'faq', hoặc None để tìm tất cả).

    Returns:
        Danh sách kết quả [{id, type, title, content, score, source_note}, ...]
    """
    global _faiss_index, _documents

    if _faiss_index is None or not _documents:
        logger.warning("FAISS index chưa được khởi tạo. Đang xây dựng...")
        build_index()

    if _faiss_index is None:
        return []

    if top_k is None:
        top_k = settings.RAG_TOP_K

    model = _load_embedding_model()
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # Tìm kiếm nhiều hơn nếu cần lọc theo type
    search_k = top_k * 3 if doc_type else top_k
    scores, indices = _faiss_index.search(query_embedding, min(search_k, len(_documents)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_documents):
            continue
        if score < settings.RAG_SCORE_THRESHOLD:
            continue

        doc = _documents[idx]
        if doc_type and doc["type"] != doc_type:
            continue

        results.append({
            "id": doc["id"],
            "type": doc["type"],
            "title": doc["title"],
            "content": doc["content"],
            "score": float(score),
            "source_note": doc.get("source_note", ""),
            "data_status": doc.get("data_status", ""),
        })

        if len(results) >= top_k:
            break

    logger.info(f"Tìm kiếm '{query[:50]}...' → {len(results)} kết quả (threshold={settings.RAG_SCORE_THRESHOLD})")
    return results


def search_knowledge(query: str, top_k: int = 3) -> list[dict]:
    """Tìm kiếm trong dữ liệu Knowledge."""
    return search(query, top_k=top_k, doc_type="knowledge")


def search_faq(query: str, top_k: int = 3) -> list[dict]:
    """Tìm kiếm trong dữ liệu FAQ."""
    return search(query, top_k=top_k, doc_type="faq")
