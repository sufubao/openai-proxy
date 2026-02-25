#!/bin/bash
# 启动脚本 - 多进程模式

set -e

echo "选择启动模式:"
echo "1) 开发模式 (单进程, 热重载)"
echo "2) 生产模式 (多进程 Gunicorn)"
echo "3) Docker 模式"
echo ""
read -p "请选择 [1-3]: " choice

case $choice in
    1)
        echo "启动开发模式..."
        exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    2)
        echo "启动生产模式 (多进程)..."

        # 安装依赖
        if [ ! -d "venv" ]; then
            python3 -m venv venv
        fi
        source venv/bin/activate
        pip install -q -r requirements.txt

        # 创建日志目录
        mkdir -p logs

        # 获取 CPU 核心数
        cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
        workers=${WORKERS:-$((cores * 2 + 1))}

        echo "使用 $workers 个 worker 进程"

        exec gunicorn -c gunicorn_config.py main:app
        ;;
    3)
        echo "启动 Docker 模式..."
        docker-compose up -d
        echo ""
        echo "查看日志: docker-compose logs -f"
        echo "停止服务: docker-compose down"
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
