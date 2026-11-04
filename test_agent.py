import io
import sys

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from app import agent

print("="*60)
print("KIỂM THỬ TOÀN BỘ LUỒNG AGENTIC AI + RAG + QWEN LOCAL")
print("="*60)

query = "Địa đạo Bến Dược có gì để tham quan?"
print(f"\n[1] Người dùng hỏi: '{query}'")
resp, trace, sources = agent.process_query(query)

print("\n[2] Phản hồi từ Agent & Qwen:")
print("-" * 50)
print(resp)
print("-" * 50)

print(f"\n[3] Agent Trace (Thời gian xử lý: {trace.total_time:.1f}s):")
print(f"  Ý định: {trace.detected_intent}")
for s in trace.steps:
    print(f"  • [Bước {s['step']}] {s['action']} (Tool: {s['tool']})")
    print(f"    ↳ {s['observation'][:120]}")

if sources:
    print(f"\n[4] Nguồn tham khảo: {sources}")

print("\n" + "="*60)
print("KIỂM THỬ AGENT THÀNH CÔNG RỰC RỠ!")
print("="*60)
