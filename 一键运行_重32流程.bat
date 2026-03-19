@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if exist ".venv311\Scripts\python.exe" (
    ".venv311\Scripts\python.exe" "数据处理及油水解释.py"
) else (
    py -3.11 "数据处理及油水解释.py"
)

echo.
echo 运行结束，按任意键关闭窗口...
pause >nul
