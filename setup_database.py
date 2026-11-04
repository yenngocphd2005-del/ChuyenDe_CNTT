"""
Script tạo database và bảng cho dự án.
Chạy: python setup_database.py
"""
import sys
import os
import io

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import settings


def setup_database():
    """Tạo database và bảng."""
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    print("=" * 50)
    print("THIẾT LẬP DATABASE POSTGRESQL")
    print("=" * 50)

    # Kiểm tra cấu hình
    if not settings.DB_PASSWORD:
        print("\n⚠ Chưa cấu hình DB_PASSWORD trong file .env")
        print("Hãy tạo file .env với nội dung:")
        print(f"  DB_HOST={settings.DB_HOST}")
        print(f"  DB_PORT={settings.DB_PORT}")
        print(f"  DB_NAME={settings.DB_NAME}")
        print(f"  DB_USER={settings.DB_USER}")
        print(f"  DB_PASSWORD=<mật khẩu PostgreSQL của bạn>")
        return False

    # 1. Kết nối PostgreSQL và tạo database nếu chưa có
    print(f"\n[1] Kết nối PostgreSQL ({settings.DB_HOST}:{settings.DB_PORT})...")
    try:
        conn = psycopg2.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            dbname="postgres",  # Kết nối database mặc định trước
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        print("  ✓ Kết nối PostgreSQL thành công.")
    except Exception as e:
        print(f"  ✗ Lỗi kết nối: {e}")
        return False

    # 2. Tạo database
    print(f"\n[2] Kiểm tra database '{settings.DB_NAME}'...")
    try:
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (settings.DB_NAME,)
        )
        if cursor.fetchone():
            print(f"  ✓ Database '{settings.DB_NAME}' đã tồn tại.")
        else:
            cursor.execute(f'CREATE DATABASE "{settings.DB_NAME}"')
            print(f"  ✓ Đã tạo database '{settings.DB_NAME}'.")
    except Exception as e:
        print(f"  ✗ Lỗi tạo database: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

    # 3. Tạo bảng bằng SQLAlchemy
    print("\n[3] Tạo bảng...")
    try:
        from app.database import init_database
        result = init_database()
        if result:
            print("  ✓ Đã tạo/kiểm tra các bảng thành công.")
        else:
            print("  ✗ Lỗi tạo bảng.")
            return False
    except Exception as e:
        print(f"  ✗ Lỗi: {e}")
        return False

    # 4. Kiểm tra ghi/đọc
    print("\n[4] Kiểm tra ghi/đọc dữ liệu...")
    try:
        from app.database import create_session, save_message, get_messages

        # Tạo session test
        session = create_session(title="Test Session")
        print(f"  ✓ Tạo session: {session.id}")

        # Lưu tin nhắn
        save_message(session.id, "user", "Xin chào, đây là tin nhắn test.")
        save_message(session.id, "assistant", "Chào bạn! Đây là phản hồi test.")
        print("  ✓ Lưu tin nhắn thành công.")

        # Đọc lại
        messages = get_messages(session.id)
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        print(f"  ✓ Đọc lại {len(messages)} tin nhắn thành công.")

    except Exception as e:
        print(f"  ✗ Lỗi kiểm tra: {e}")
        return False

    print("\n" + "=" * 50)
    print("THIẾT LẬP DATABASE HOÀN TẤT!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = setup_database()
    sys.exit(0 if success else 1)
