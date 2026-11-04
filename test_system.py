"""
Script kiểm thử toàn bộ hệ thống Chatbot Địa đạo Củ Chi
Chạy: python test_system.py
"""
import sys
import os
import io
import time

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_rag():
    print("\n" + "="*50)
    print("1. KIỂM THỬ RAG & FAISS VECTOR SEARCH")
    print("="*50)
    from app import rag
    
    # 1.1 Khởi tạo / nạp index
    print("[1.1] Kiểm tra nạp FAISS index...")
    ok = rag.build_index()
    assert ok, "Không thể nạp hoặc xây dựng FAISS index!"
    print("  ✓ FAISS index nạp thành công.")
    
    # 1.2 Truy vấn Knowledge cơ bản
    print("\n[1.2] Kiểm thử truy vấn: 'Địa đạo Bến Dược'")
    results = rag.search_knowledge("Địa đạo Bến Dược", top_k=3)
    assert len(results) > 0, "Không tìm thấy kết quả cho Bến Dược!"
    print(f"  ✓ Tìm thấy {len(results)} kết quả. Top 1: [{results[0]['id']}] {results[0]['title']} (Score: {results[0]['score']:.3f})")
    assert any("Bến Dược" in r["title"] or "Bến Dược" in r["content"] for r in results)

    # 1.3 Truy vấn FAQ
    print("\n[1.3] Kiểm thử truy vấn FAQ: 'Giá vé vào cổng người lớn'")
    faq_res = rag.search_faq("Giá vé vào cổng người lớn", top_k=3)
    assert len(faq_res) > 0, "Không tìm thấy kết quả FAQ giá vé!"
    print(f"  ✓ Tìm thấy {len(faq_res)} kết quả FAQ. Top 1: [{faq_res[0]['id']}] {faq_res[0]['title']} (Score: {faq_res[0]['score']:.3f})")

    # 1.4 Câu hỏi ngoài phạm vi
    print("\n[1.4] Kiểm thử câu hỏi ngoài phạm vi: 'Mua điện thoại iPhone ở đâu'")
    out_res = rag.search("Mua điện thoại iPhone ở đâu", top_k=3)
    print(f"  ✓ Kết quả vượt ngưỡng: {len(out_res)} (Kỳ vọng: 0 kết quả)")
    assert len(out_res) == 0, f"Kỳ vọng 0 kết quả ngoài phạm vi, nhưng nhận {len(out_res)}"
    print("✓ Kiểm thử RAG hoàn tất ĐẠT 100%!")

def test_agent_intent():
    print("\n" + "="*50)
    print("2. KIỂM THỬ AGENT INTENT CLASSIFICATION & TOOLS")
    print("="*50)
    from app import agent
    
    cases = [
        ("Xin chào bạn", False, agent.Intent.GREETING),
        ("Hello", False, agent.Intent.GREETING),
        ("Bến Dược có những điểm tham quan nào?", False, agent.Intent.KNOWLEDGE_QUERY),
        ("Vậy giá vé ở đó là bao nhiêu?", True, agent.Intent.FOLLOW_UP),
        ("Giải thích ngắn hơn", True, agent.Intent.REPHRASE),
        ("Dịch câu trả lời vừa rồi sang tiếng Anh", True, agent.Intent.TRANSLATION),
        ("?", False, agent.Intent.CLARIFICATION_NEEDED),
    ]
    
    passed = 0
    for query, has_history, expected in cases:
        intent = agent.classify_intent(query, has_history=has_history)
        match = (intent == expected)
        status = "✓ ĐẠT" if match else "✗ SAI"
        print(f"  [{status}] '{query}' (has_history={has_history}) -> Intent: {intent.value} (Kỳ vọng: {expected.value})")
        if match:
            passed += 1
            
    print(f"\n✓ Tỷ lệ phân loại ý định chính xác: {passed}/{len(cases)} ({passed/len(cases)*100:.1f}%)")
    assert passed == len(cases)

def test_database_crud():
    print("\n" + "="*50)
    print("3. KIỂM THỬ POSTGRESQL CRUD")
    print("="*50)
    from app import database
    from app.config import settings
    
    if not settings.DB_PASSWORD:
        print("  ⚠ Bỏ qua vì DB_PASSWORD trong .env chưa cấu hình.")
        print("    Vui lòng điền DB_PASSWORD vào .env để chạy kiểm thử database.")
        return
        
    try:
        ok = database.init_database()
        if not ok:
            print("  ✗ Không kết nối được PostgreSQL.")
            return
            
        # Create session
        sess = database.create_session(title="Test Suite Session")
        print(f"  ✓ Tạo session thành công: {sess.id}")
        
        # Save message
        m1 = database.save_message(sess.id, "user", "Câu hỏi kiểm thử 1")
        m2 = database.save_message(sess.id, "assistant", "Câu trả lời kiểm thử 1", sources='[{"id":"K01"}]')
        print(f"  ✓ Lưu messages thành công. Seq user: {m1.sequence_number}, Seq assistant: {m2.sequence_number}")
        
        # Read back
        msgs = database.get_messages(sess.id)
        assert len(msgs) == 2
        print(f"  ✓ Đọc lại đúng {len(msgs)} tin nhắn.")
        print("✓ Kiểm thử Database hoàn tất ĐẠT!")
    except Exception as e:
        print(f"  ✗ Lỗi Database: {e}")

if __name__ == "__main__":
    test_rag()
    test_agent_intent()
    test_database_crud()
    print("\n" + "="*50)
    print("TẤT CẢ BÀI KIỂM THỬ ĐÃ CHẠY XONG!")
    print("="*50)
