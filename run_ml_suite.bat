@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if exist ".venv311\Scripts\python.exe" (
    ".venv311\Scripts\python.exe" "ml_dl_suite.py" --task all
) else (
    py -3.11 "ml_dl_suite.py" --task all
)

echo.
echo 运行结束，按任意键关闭窗口...
pause >nul
