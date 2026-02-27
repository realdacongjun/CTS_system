@echo off
echo ========================================
echo CTS 系统实验启动脚本
echo ========================================

echo 1. 检查依赖...
python -c "import yaml, pandas, torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo 缺少必要依赖，请运行: pip install -r requirements.txt
    pause
    exit /b 1
)

echo 2. 准备测试镜像...
python prepare_images.py

echo 3. 启动代理服务器 (请在新终端中运行)...
echo    python proxy_server.py
echo.

echo 4. 运行实验...
echo    python experiment_runner.py
echo.

pause