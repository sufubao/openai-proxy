#!/bin/bash
# 部署脚本 - 使用 supervisor 管理进程

set -e

echo "🚀 OpenAI Proxy 部署脚本"

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "安装依赖..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 安装 supervisor
if ! command -v supervisord &> /dev/null; then
    echo "安装 supervisor..."
    pip install -q supervisor
fi

# 创建日志目录
mkdir -p logs

# 生成 supervisor 配置
cat > supervisord.conf << EOF
[supervisord]
nodaemon=true
logfile=logs/supervisord.log
pidfile=supervisord.pid
childlogdir=logs

[program:openai-proxy]
command=venv/bin/python main.py
directory=$(pwd)
environment=UPSTREAM_URL="${UPSTREAM_URL:-http://10.42.53.44:8000}",PORT="${PORT:-8000}",HOST="${HOST:-0.0.0.0}"
autostart=true
autorestart=true
startretries=3
stderr_logfile=logs/proxy.err.log
stdout_logfile=logs/proxy.out.log
user=$(whoami)
priority=999

[supervisorctl]
serverurl=unix:///tmp/supervisor.sock
EOF

echo ""
echo "✅ 部署完成！"
echo ""
echo "启动方式："
echo "  直接启动:   ./venv/bin/python main.py"
echo "  Supervisor:  supervisord -c supervisord.conf"
echo ""
echo "管理命令 (supervisor):"
echo "  查看状态:   supervisorctl -c supervisord.conf status"
echo "  重启服务:   supervisorctl -c supervisord.conf restart openai-proxy"
echo "  停止服务:   supervisorctl -c supervisord.conf stop openai-proxy"
echo "  查看日志:   tail -f logs/proxy.out.log"
echo ""
