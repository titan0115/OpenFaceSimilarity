@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PY_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20250409/cpython-3.13.3+20250409-x86_64-pc-windows-msvc-install_only.tar.gz"
set "PY_DIR=%SCRIPT_DIR%python"
set "VENV_DIR=%SCRIPT_DIR%venv"

:: Download and extract Python
if not exist "%PY_DIR%" (
    echo Downloading Python...
    set "TEMP_FILE=%SCRIPT_DIR%python.tar.gz"
    
    :: Try different download tools
    where curl >nul 2>&1 && (
        curl -L -o "!TEMP_FILE!" "!PY_URL!" || goto :error
    ) || (
        bitsadmin /transfer job /download /priority normal "!PY_URL!" "!TEMP_FILE!" || (
            echo ❌ Need curl or bitsadmin to download
            goto :error
        )
    )

    echo Extracting Python...
    where tar >nul 2>&1 || (
        echo ❌ Tar not available. Install with:
        echo DISM /Online /Add-Capability /CapabilityName:Microsoft.Windows.Tar~~~~0.0.1.0
        goto :error
    )
    
    mkdir "!PY_DIR!" >nul 2>&1
    tar -xzf "!TEMP_FILE!" -C "!PY_DIR!" --strip-components=1 || (
        echo ❌ Extraction failed
        goto :error
    )
    
    del /q "!TEMP_FILE!" 2>nul
    
    if not exist "%PY_DIR%\python.exe" (
        echo ❌ Python installation failed!
        goto :error
    )
    echo ✅ Python installed
)

:: Create virtual environment
if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    set "PATH=%PY_DIR%;%PATH%"
    
    .\python\python.exe -m venv "%VENV_DIR%" || (
        echo ❌ Failed to create venv
        goto :error
    )
    
    .\venv\Scripts\python.exe -m pip install --upgrade pip || (
        echo ❌ Failed to upgrade pip
        goto :error
    )

    :: Fixed requirements file
    set "REQUIREMENTS=requirements.txt"
    
    if exist "%SCRIPT_DIR%!REQUIREMENTS!" (
        echo Installing dependencies from !REQUIREMENTS!...
        .\venv\Scripts\python.exe -m pip install -r "%SCRIPT_DIR%!REQUIREMENTS!" || (
            echo ❌ Failed to install dependencies
            echo Check the requirements file: !REQUIREMENTS!
            goto :error
        )
    ) else (
        echo ❌ !REQUIREMENTS! not found
        goto :error
    )
)

:: Run application
call "%VENV_DIR%\Scripts\activate.bat"
python start.py

pause

:error
echo Installation failed. Check errors above.
pause
exit /b 1