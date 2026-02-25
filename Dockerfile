FROM python:3.11-slim

LABEL maintainer="OpenAI Proxy"
LABEL version="2.0.0"

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY main.py .
COPY gunicorn_config.py .

# 创建日志目录
RUN mkdir -p logs

# 创建非 root 用户
RUN useradd -m -u 1000 proxy && \
    chown -R proxy:proxy /app

USER proxy

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# 启动命令 (多进程)
CMD ["gunicorn", "-c", "gunicorn_config.py", "main:app"]
