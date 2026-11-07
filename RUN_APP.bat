@echo off
title FIRST Saathi
REM FIRST Saathi - open the app (sets everything up automatically if needed).
cd /d "%~dp0"
call "%~dp0setup_env.bat"
if errorlevel 1 goto :failed
echo.
echo Opening FIRST Saathi at http://localhost:8501 in your browser.
echo Keep this window open while you use the app; close it to stop the app.
echo Add your API keys on the "7. Settings (API keys)" page. Without keys it runs in offline mode.
echo.
"%VPY%" -m streamlit run app.py --server.port 8501
pause
exit /b 0

:failed
echo.
echo Setup did not finish. Follow the steps above, then double-click this file again.
pause
exit /b 1
