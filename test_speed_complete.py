import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import time
import torch
from app import rag, agent
from app.config import settings

def test():
    print("Settings MAX_NEW_TOKENS:", settings.MAX_NEW_TOKENS)
    queries = [
        "Cách đi xe buýt đến Địa đạo Củ Chi như thế nào?",
        "Giá vé tham quan là bao nhiêu?",
        "Where are the Cu Chi Tunnels located and how to visit?"
    ]
    
    for q in queries:
        print(f"\n==========================================")
        print(f"HỎI: {q}")
        t0 = time.time()
        resp, trace, sources = agent.process_query(q)
        dt = time.time() - t0
        print(f"THỜI GIAN: {dt:.2f}s")
        print(f"ĐÁP ÁN: {resp}")
        print(f"SỐ KÝ TỰ: {len(resp)}")
        print(f"KẾT THÚC BẰNG: '{resp[-1] if resp else ''}'")

if __name__ == "__main__":
    test()
