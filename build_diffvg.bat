@echo off
set PROJECT_ROOT=%~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%PATH%

set PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe

echo === Environment check ===
where cmake
where nvcc
%PYTHON% -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
if %ERRORLEVEL% neq 0 (
    echo FAILED: torch not found in venv
    pause
    exit /b 1
)

echo === Cleaning previous build ===
cd /d %PROJECT_ROOT%third-party\diffvg
if exist build rmdir /s /q build

echo === Building diffvg ===
%PYTHON% setup.py install
echo === Done (exit code: %ERRORLEVEL%) ===
pause
