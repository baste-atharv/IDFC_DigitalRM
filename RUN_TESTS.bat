@echo off
title FIRST Saathi - tests
REM FIRST Saathi - run the 20 checks (no API key needed: LLM calls go through a fake transport).
cd /d "%~dp0"
call "%~dp0setup_env.bat"
if errorlevel 1 goto :failed
"%VPY%" tests\test_saathi.py
pause
exit /b 0

:failed
echo.
echo Setup did not finish. Follow the steps above, then run this file again.
pause
exit /b 1
