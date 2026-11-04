@echo off
title Chatbot Dia Dao Cu Chi - Backend Server
color 0A

echo ========================================================
echo       KHOI DONG BACKEND CHATBOT DIA DAO CU CHI
echo ========================================================
echo.

:: Di chuyen den thu muc du an
cd /d "D:\CD_CNTT\DA_CD"

echo [1/3] Kiem tra Python...
python --version
if %errorlevel% neq 0 (
    echo [LOI] Khong tim thay Python tren he thong!
    pause
    exit /b
)

echo [2/3] Dang khoi dong may chu Backend va tai AI Model...
echo Luu y: Qua trinh tai Model se mat khoang 10-20 giay.
echo Sau khi hien thi 'Application startup complete', ban co the truy cap.
echo.

:: Tu dong mo trinh duyet sau 12 giay
start "" cmd /c "timeout /t 12 /nobreak >nul & start http://127.0.0.1:8000"

:: Chay uvicorn server
python -m uvicorn main:app --host 127.0.0.1 --port 8000

pause
