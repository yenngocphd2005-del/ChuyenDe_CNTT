@echo off
chcp 65001 >nul
echo ============================================
echo  KHỞI ĐỘNG CHATBOT ĐỊA ĐẠO CỦ CHI
echo ============================================
echo.

:: Kiểm tra .env
if not exist ".env" (
    echo [!] File .env chưa tồn tại.
    echo     Hãy sao chép .env.example thành .env và điền mật khẩu PostgreSQL.
    echo     copy .env.example .env
    pause
    exit /b 1
)

:: Kiểm tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [!] Python chưa được cài đặt hoặc chưa có trong PATH.
    pause
    exit /b 1
)

echo [1/3] Kiểm tra FAISS index...
if not exist "data\faiss_index\index.faiss" (
    echo      Chưa có index, đang xây dựng...
    python build_index.py
    if errorlevel 1 (
        echo [!] Lỗi xây dựng FAISS index.
        pause
        exit /b 1
    )
) else (
    echo      FAISS index đã có sẵn.
)

echo.
echo [2/3] Kiểm tra database...
python setup_database.py
if errorlevel 1 (
    echo [!] Lỗi thiết lập database. Kiểm tra file .env
    pause
    exit /b 1
)

echo.
echo [3/3] Khởi động server...
echo      Truy cập: http://localhost:8000
echo      Nhấn Ctrl+C để dừng server.
echo.
python -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
