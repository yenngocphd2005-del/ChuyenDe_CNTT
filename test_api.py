"""
Script test API endpoints - tránh vấn đề encoding PowerShell.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import json
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print("=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)
    r = requests.get(f"{BASE_URL}/health")
    data = r.json()
    print(f"  Status: {data['status']}")
    print(f"  LLM loaded: {data['llm_loaded']}")
    print(f"  Database: {data['database']}")
    assert data["status"] == "ok"
    assert data["llm_loaded"] == True
    assert data["database"] == True
    print("  ✓ PASS\n")
    return data

def test_create_session():
    print("=" * 60)
    print("TEST 2: Tạo session mới")
    print("=" * 60)
    r = requests.post(f"{BASE_URL}/api/sessions", json={})
    data = r.json()
    print(f"  Session ID: {data['id']}")
    print(f"  Created at: {data['created_at']}")
    print("  ✓ PASS\n")
    return data["id"]

def test_greeting(session_id):
    print("=" * 60)
    print("TEST 3: Gửi lời chào 'Xin chao'")
    print("=" * 60)
    r = requests.post(f"{BASE_URL}/api/chat", json={
        "message": "Xin chao",
        "session_id": session_id,
    })
    data = r.json()
    print(f"  Response: {data['response']}")
    print(f"  Intent: {data['agent_trace']['detected_intent']}")
    print(f"  Steps: {len(data['agent_trace']['steps'])}")
    print(f"  Time: {data['agent_trace']['total_time']}s")
    for step in data["agent_trace"]["steps"]:
        print(f"    Step {step['step']}: [{step.get('tool','?')}] {step['action']}")
    assert data["agent_trace"]["detected_intent"] == "greeting", \
        f"Expected greeting, got {data['agent_trace']['detected_intent']}"
    assert len(data["response"]) > 5, "Response quá ngắn"
    print("  ✓ PASS\n")

def test_knowledge_query(session_id):
    print("=" * 60)
    print("TEST 4: Hỏi kiến thức 'Bến Dược có gì?'")
    print("=" * 60)
    r = requests.post(f"{BASE_URL}/api/chat", json={
        "message": "Bến Dược có gì?",
        "session_id": session_id,
    })
    data = r.json()
    print(f"  Response: {data['response'][:300]}...")
    print(f"  Intent: {data['agent_trace']['detected_intent']}")
    print(f"  Steps: {len(data['agent_trace']['steps'])}")
    print(f"  Time: {data['agent_trace']['total_time']}s")
    if data.get("sources"):
        print(f"  Sources:")
        for s in data["sources"]:
            print(f"    - {s['id']}: {s['title']} (score={s['score']})")
    for step in data["agent_trace"]["steps"]:
        print(f"    Step {step['step']}: [{step.get('tool','?')}] {step['action']}")
    assert data["response"] and len(data["response"]) > 10
    print("  ✓ PASS\n")

def test_history(session_id):
    print("=" * 60)
    print("TEST 5: Kiểm tra lịch sử hội thoại")
    print("=" * 60)
    r = requests.get(f"{BASE_URL}/api/sessions/{session_id}/messages")
    messages = r.json()
    print(f"  Số tin nhắn: {len(messages)}")
    for m in messages:
        print(f"    #{m['sequence_number']} [{m['role']}]: {m['content'][:80]}...")
    assert len(messages) >= 2, "Phải có ít nhất 2 tin nhắn"
    print("  ✓ PASS\n")

def test_sessions_list():
    print("=" * 60)
    print("TEST 6: Liệt kê sessions")
    print("=" * 60)
    r = requests.get(f"{BASE_URL}/api/sessions")
    sessions = r.json()
    print(f"  Số sessions: {len(sessions)}")
    for s in sessions[:5]:
        print(f"    - {s['id'][:8]}... | {s.get('title','(no title)')}")
    assert len(sessions) >= 1
    print("  ✓ PASS\n")


if __name__ == "__main__":
    print("\n🤖 KIỂM THỬ CHATBOT ĐỊA ĐẠO CỦ CHI\n")

    # Test 1: Health
    test_health()

    # Test 2: Session
    sid = test_create_session()

    # Test 3: Greeting (nhanh, không cần RAG)
    test_greeting(sid)

    # Test 4: Knowledge query (cần RAG + LLM, chậm ~60-120s)
    print("⏳ Đang chờ LLM xử lý (có thể mất 1-3 phút)...\n")
    test_knowledge_query(sid)

    # Test 5: History
    test_history(sid)

    # Test 6: Sessions list
    test_sessions_list()

    print("=" * 60)
    print("✅ TẤT CẢ CÁC TEST ĐỀU PASS!")
    print("=" * 60)
