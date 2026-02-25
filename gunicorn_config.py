"""
Gunicorn Configuration File
Multi-process deployment for higher concurrency
"""
import os
import multiprocessing

# Server socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker process count
# Recommended: (CPU cores * 2) + 1
workers = int(os.getenv("WORKERS", multiprocessing.cpu_count() * 2 + 1))

# Worker type
# sync: synchronous worker (simple, low concurrency)
# uvicorn.workers.UvicornWorker: asynchronous worker (recommended, high concurrency)
worker_class = "uvicorn.workers.UvicornWorker"

# Concurrent requests per worker
worker_connections = int(os.getenv("WORKER_CONNECTIONS", "1000"))

# Restart worker after max requests (prevent memory leaks)
max_requests = int(os.getenv("MAX_REQUESTS", "10000"))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", "1000"))

# Timeout
timeout = int(os.getenv("GTIMEOUT", "300"))
keepalive = int(os.getenv("KEEPALIVE", "5"))

# Process management
preload_app = True  # Preload app to save memory
daemon = False  # Set to False in Docker
pidfile = "gunicorn.pid"
umask = 0o007
user = None
group = None
tmp_upload_dir = None

# Logging
loglevel = os.getenv("LOG_LEVEL", "info")
accesslog = os.getenv("ACCESS_LOG", "logs/access.log")
errorlog = os.getenv("ERROR_LOG", "logs/error.log")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "openai-proxy"

# Graceful shutdown
graceful_timeout = int(os.getenv("GRACEFUL_TIMEOUT", "30"))

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# SSL (if HTTPS needed)
# keyfile = "/path/to/key.pem"
# certfile = "/path/to/cert.pem"

# Print config on startup
def on_starting(server):
    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║              OpenAI Proxy (Multi-Process)                   ║
    ╠════════════════════════════════════════════════════════════╣
    ║  Workers:              {workers:<45} ║
    ║  Worker Class:         {worker_class:<45} ║
    ║  Worker Connections:   {worker_connections:<45} ║
    ║  Bind:                 {bind:<45} ║
    ║  Max Requests:         {max_requests:<45} ║
    ║  Timeout:              {timeout}s (Graceful: {graceful_timeout}s){" " * (20 - len(str(graceful_timeout)))}║
    ╠════════════════════════════════════════════════════════════╣
    ║  Max Concurrent: ~{workers * worker_connections:<42} ║
    ╚════════════════════════════════════════════════════════════╝
    """)
