@echo off
echo ============================================
echo  MARC-Clinical Frontend Setup
echo ============================================
cd /d D:\MARC_Clinical\frontend
echo Installing dependencies...
call npm install
echo.
echo Starting development server...
call npm start
