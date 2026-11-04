"""
Script xây dựng FAISS index từ dữ liệu.
Chạy: python build_index.py
"""
import sys
import os
import io
import logging

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app.rag import build_index

if __name__ == "__main__":
    force = "--force" in sys.argv
    print("=" * 50)
    print("XÂY DỰNG FAISS INDEX")
    if force:
        print("(Chế độ: xây dựng lại từ đầu)")
    print("=" * 50)

    success = build_index(force_rebuild=force)
    if success:
        print("\n✓ FAISS index đã sẵn sàng!")
    else:
        print("\n✗ Lỗi xây dựng FAISS index.")
        sys.exit(1)
