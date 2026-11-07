@echo off
title FIRST Saathi - RAFT dataset
REM FIRST Saathi - build the RAFT fine-tuning dataset (data\raft_dataset.jsonl).
REM Uses the LLM saved in Settings ("Remember on this computer") to write questions; otherwise template questions.
cd /d "%~dp0"
call "%~dp0setup_env.bat"
if errorlevel 1 goto :failed
if exist ".saathi_keys.env" goto :withllm
"%VPY%" -m saathi.raft
pause
exit /b 0

:withllm
"%VPY%" -m saathi.raft --llm
pause
exit /b 0

:failed
echo.
echo Setup did not finish. Follow the steps above, then run this file again.
pause
exit /b 1
