"""
Gunicorn 配置文件
多进程部署，提高并发能力
"""
import os
import multiprocessing

# 服务器 socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker 进程数
# 建议值: (CPU核心数 * 2) + 1
workers = int(os.getenv("WORKERS", multiprocessing.cpu_count() * 2 + 1))

# Worker 类型
# sync: 同步 worker (简单，但并发低)
# uvicorn.workers.UvicornWorker: 异步 worker (推荐，高并发)
worker_class = "uvicorn.workers.UvicornWorker"

# 每个 worker 的并发请求数
worker_connections = int(os.getenv("WORKER_CONNECTIONS", "1000"))

# 最大请求数后重启 worker（防止内存泄漏）
max_requests = int(os.getenv("MAX_REQUESTS", "10000"))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", "1000"))

# 超时
timeout = int(os.getenv("GTIMEOUT", "300"))
keepalive = int(os.getenv("KEEPALIVE", "5"))

# 进程管理
preload_app = True  # 预加载应用，节省内存
daemon = False  # Docker 环境设为 False
pidfile = "gunicorn.pid"
umask = 0o007
user = None
group = None
tmp_upload_dir = None

# 日志
loglevel = os.getenv("LOG_LEVEL", "info")
accesslog = os.getenv("ACCESS_LOG", "logs/access.log")
errorlog = os.getenv("ERROR_LOG", "logs/error.log")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 进程命名
proc_name = "openai-proxy"

# 优雅关闭
graceful_timeout = int(os.getenv("GRACEFUL_TIMEOUT", "30"))

# 安全
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# SSL (如需 HTTPS)
# keyfile = "/path/to/key.pem"
# certfile = "/path/to/cert.pem"

# 启动前打印配置
def on_starting(server):
    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║              OpenAI Proxy (多进程版本)                      ║
    ╠════════════════════════════════════════════════════════════╣
    ║  Workers:              {workers:<45} ║
    ║  Worker Class:         {worker_class:<45} ║
    ║  Worker Connections:   {worker_connections:<45} ║
    ║  Bind:                 {bind:<45} ║
    ║  Max Requests:         {max_requests:<45} ║
    ║  Timeout:              {timeout}s (Graceful: {graceful_timeout}s){" " * (20 - len(str(graceful_timeout)))}║
    ╠════════════════════════════════════════════════════════════╣
    ║  理论最大并发: ~{workers * worker_connections:<42} ║
    ╚════════════════════════════════════════════════════════════╝
    """)
