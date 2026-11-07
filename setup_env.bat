@echo off
REM Helper used by the other .bat files (you do not need to open it yourself).
REM 1. Finds a real Python 3.10-3.14 (ignores the Microsoft Store shortcut).
REM 2. If none is installed, installs Python 3.12 with winget, or opens python.org.
REM 3. Creates a private environment in .venv and installs the packages.
REM Leaves VPY = path of the private python.exe. Returns errorlevel 1 if something failed.
cd /d "%~dp0"
set "VPY=%~dp0.venv\Scripts\python.exe"

if exist "%VPY%" (
  "%VPY%" -c "import streamlit, pandas, sklearn, plotly, requests, anthropic, openai, edge_tts" >nul 2>nul && exit /b 0
  goto :install
)

call :findpy
if not defined PY call :wingetpy
if not defined PY goto :nopython

echo Using Python: %PY%
if exist ".venv" rmdir /s /q ".venv"
echo Creating a private Python environment (.venv) ...
"%PY%" -m venv .venv
if exist "%VPY%" goto :install
echo.
echo Could not create the .venv folder in this project folder.
echo Move the project to a normal folder you can write to, e.g. D:\projects, and try again.
exit /b 1

:install
echo Installing packages (first run only, about 3-5 minutes; needs internet) ...
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Package installation failed. Check your internet connection, then run this file again.
  exit /b 1
)
exit /b 0

:nopython
echo.
echo ===================================================================
echo  Python is not installed on this PC. (Typing "python" only opens
echo  the Microsoft Store, which is why you saw "Python was not found".)
echo.
echo  1. A browser tab is opening: download Python 3.12 for Windows.
echo  2. In the installer, TICK "Add python.exe to PATH", then Install Now.
echo  3. Close this window and double-click SETUP_AND_RUN.bat again.
echo ===================================================================
start "" "https://www.python.org/downloads/windows/"
exit /b 1

:wingetpy
where winget >nul 2>nul || exit /b 0
echo Python not found. Installing Python 3.12 for your user with winget (1-2 minutes) ...
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
call :findpy
exit /b 0

:findpy
set "PY="
call :trycmd py -3
if not defined PY call :trycmd python
if not defined PY call :trycmd python3
if not defined PY for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do call :trypath "%%~D\python.exe"
if not defined PY for /d %%D in ("%ProgramFiles%\Python3*") do call :trypath "%%~D\python.exe"
if not defined PY for /d %%D in ("%LocalAppData%\Python\pythoncore-3*") do call :trypath "%%~D\python.exe"
exit /b 0

:trycmd
if defined PY exit /b 0
set "CAND="
for /f "usebackq delims=" %%P in (`%* -c "import sys; print(sys.executable)" 2^>nul ^<nul`) do set "CAND=%%P"
if defined CAND call :trypath "%CAND%"
exit /b 0

:trypath
if defined PY exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) else 1)" >nul 2>nul && set "PY=%~1"
exit /b 0
