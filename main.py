"""
OpenAI API 代理服务器 - 生产级版本
支持：进程管理、连接池、限流熔断、监控、日志、优雅关闭
"""
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import os
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from collections import deque
import signal
import sys

# ==================== 配置 ====================

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://10.42.53.44:8000")
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

# 超时配置
TIMEOUT = float(os.getenv("TIMEOUT", "300"))
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "30"))

# 重试配置
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1"))

# 限流配置
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))  # 每分钟请求数
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # 时间窗口(秒)

# 熔断配置
CIRCUIT_BREAKER_ENABLED = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))  # 失败阈值
CIRCUIT_BREAKER_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))  # 熔断恢复时间(秒)

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/proxy.log")
REQUEST_LOG_ENABLED = os.getenv("REQUEST_LOG_ENABLED", "true").lower() == "true"

# 监控配置
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"

# 优雅关闭配置
SHUTDOWN_TIMEOUT = int(os.getenv("SHUTDOWN_TIMEOUT", "30"))  # 优雅关闭超时(秒)

# ==================== 日志设置 ====================

# 创建日志目录
os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else ".", exist_ok=True)

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ==================== 监控指标 ====================

class Metrics:
    """Prometheus 风格指标收集"""

    def __init__(self):
        self.requests_total = 0
        self.requests_success = 0
        self.requests_error = 0
        self.requests_timeout = 0
        self.requests_retry = 0
        self.requests_active = 0
        self.requests_streaming = 0
        self.latency_total = 0
        self.latency_count = 0
        self.upstream_errors = 0
        self.circuit_breaker_trips = 0
        self.start_time = time.time()

        # 按状态码统计
        self.status_codes: Dict[int, int] = {}

        # 按路径统计
        self.path_stats: Dict[str, int] = {}

    def record_request(self, path: str):
        self.requests_total += 1
        self.requests_active += 1
        self.path_stats[path] = self.path_stats.get(path, 0) + 1

    def record_response(self, status_code: int, latency: float, is_timeout: bool = False):
        self.requests_active -= 1
        self.status_codes[status_code] = self.status_codes.get(status_code, 0) + 1
        self.latency_total += latency
        self.latency_count += 1

        if is_timeout:
            self.requests_timeout += 1
        elif 200 <= status_code < 400:
            self.requests_success += 1
        else:
            self.requests_error += 1

    def record_retry(self):
        self.requests_retry += 1

    def record_upstream_error(self):
        self.upstream_errors += 1

    def record_streaming_start(self):
        self.requests_streaming += 1

    def record_streaming_end(self):
        self.requests_streaming -= 1

    def record_circuit_breaker_trip(self):
        self.circuit_breaker_trips += 1

    def get_prometheus_metrics(self) -> str:
        """导出 Prometheus 格式指标"""
        uptime = time.time() - self.start_time
        avg_latency = (self.latency_total / self.latency_count) if self.latency_count > 0 else 0

        metrics = [
            "# HELP proxy_requests_total Total number of requests",
            "# TYPE proxy_requests_total counter",
            f"proxy_requests_total {self.requests_total}",
            "",
            "# HELP proxy_requests_success Total number of successful requests",
            "# TYPE proxy_requests_success counter",
            f"proxy_requests_success {self.requests_success}",
            "",
            "# HELP proxy_requests_error Total number of error requests",
            "# TYPE proxy_requests_error counter",
            f"proxy_requests_error {self.requests_error}",
            "",
            "# HELP proxy_requests_timeout Total number of timeout requests",
            "# TYPE proxy_requests_timeout counter",
            f"proxy_requests_timeout {self.requests_timeout}",
            "",
            "# HELP proxy_requests_retry Total number of retried requests",
            "# TYPE proxy_requests_retry counter",
            f"proxy_requests_retry {self.requests_retry}",
            "",
            "# HELP proxy_requests_active Current number of active requests",
            "# TYPE proxy_requests_active gauge",
            f"proxy_requests_active {self.requests_active}",
            "",
            "# HELP proxy_requests_streaming Current number of streaming requests",
            "# TYPE proxy_requests_streaming gauge",
            f"proxy_requests_streaming {self.requests_streaming}",
            "",
            "# HELP proxy_latency_seconds Average request latency in seconds",
            "# TYPE proxy_latency_seconds gauge",
            f"proxy_latency_seconds {avg_latency:.3f}",
            "",
            "# HELP proxy_upstream_errors_total Total upstream errors",
            "# TYPE proxy_upstream_errors_total counter",
            f"proxy_upstream_errors_total {self.upstream_errors}",
            "",
            "# HELP proxy_circuit_breaker_trips_total Total circuit breaker trips",
            "# TYPE proxy_circuit_breaker_trips_total counter",
            f"proxy_circuit_breaker_trips_total {self.circuit_breaker_trips}",
            "",
            "# HELP proxy_uptime_seconds Proxy uptime in seconds",
            "# TYPE proxy_uptime_seconds gauge",
            f"proxy_uptime_seconds {uptime:.0f}",
        ]

        # 状态码分布
        metrics.append("\n# HELP proxy_response_status Response status codes")
        metrics.append("# TYPE proxy_response_status counter")
        for code, count in self.status_codes.items():
            metrics.append(f'proxy_response_status{{status_code="{code}"}} {count}')

        # 路径统计
        metrics.append("\n# HELP proxy_path_requests Requests per path")
        metrics.append("# TYPE proxy_path_requests counter")
        for path, count in self.path_stats.items():
            safe_path = path.replace('"', '\\"')
            metrics.append(f'proxy_path_requests{{path="{safe_path}"}} {count}')

        return "\n".join(metrics)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        uptime = time.time() - self.start_time
        avg_latency = (self.latency_total / self.latency_count) if self.latency_count > 0 else 0

        return {
            "uptime_seconds": round(uptime, 1),
            "requests": {
                "total": self.requests_total,
                "success": self.requests_success,
                "error": self.requests_error,
                "timeout": self.requests_timeout,
                "retry": self.requests_retry,
                "active": self.requests_active,
                "streaming": self.requests_streaming,
            },
            "latency": {
                "avg_seconds": round(avg_latency, 3),
            },
            "errors": {
                "upstream": self.upstream_errors,
                "circuit_breaker_trips": self.circuit_breaker_trips,
            },
            "status_codes": self.status_codes,
        }


# 全局指标实例
metrics = Metrics()


# ==================== 熔断器 ====================

class CircuitBreaker:
    """熔断器 - 防止雪崩效应"""

    def __init__(self, threshold: int, timeout: int):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
        self._lock = asyncio.Lock()

    async def record_success(self):
        async with self._lock:
            if self.state == "half_open":
                self.state = "closed"
                logger.info("熔断器恢复到关闭状态")
            self.failure_count = 0

    async def record_failure(self):
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.threshold and self.state != "open":
                self.state = "open"
                metrics.record_circuit_breaker_trip()
                logger.error(f"熔断器打开！失败次数: {self.failure_count}")

    async def can_request(self) -> bool:
        """检查是否可以发送请求"""
        async with self._lock:
            if self.state == "closed":
                return True

            if self.state == "open":
                # 检查是否可以尝试恢复
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = "half_open"
                    logger.info("熔断器进入半开状态，尝试恢复")
                    return True
                return False

            if self.state == "half_open":
                return True

        return False

    def get_state(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "threshold": self.threshold,
            "timeout_seconds": self.timeout,
            "last_failure_time": self.last_failure_time,
        }


# 全局熔断器
circuit_breaker = CircuitBreaker(CIRCUIT_BREAKER_THRESHOLD, CIRCUIT_BREAKER_TIMEOUT) if CIRCUIT_BREAKER_ENABLED else None


# ==================== 限流器 ====================

class RateLimiter:
    """滑动窗口限流器"""

    def __init__(self, requests: int, window: int):
        self.requests = requests
        self.window = window
        self.clients: Dict[str, deque] = {}

    async def is_allowed(self, client_id: str) -> bool:
        now = time.time()

        # 获取或创建客户端记录
        if client_id not in self.clients:
            self.clients[client_id] = deque()

        # 移除过期记录
        client_requests = self.clients[client_id]
        while client_requests and client_requests[0] < now - self.window:
            client_requests.popleft()

        # 检查是否超过限制
        if len(client_requests) >= self.requests:
            return False

        # 记录本次请求
        client_requests.append(now)
        return True

    def cleanup(self):
        """清理过期数据"""
        now = time.time()
        for client_id in list(self.clients.keys()):
            client_requests = self.clients[client_id]
            while client_requests and client_requests[0] < now - self.window:
                client_requests.popleft()
            if not client_requests:
                del self.clients[client_id]


# 全局限流器
rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW) if RATE_LIMIT_ENABLED else None


# ==================== 请求日志 ====================

class RequestLogger:
    """请求日志记录"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.log_file = "logs/requests.log"
        if self.enabled:
            os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else ".", exist_ok=True)

    def log(self, request_id: str, method: str, path: str, status_code: int,
            latency: float, client_ip: str, error: Optional[str] = None):
        if not self.enabled:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "latency_ms": round(latency * 1000, 2),
            "client_ip": client_ip,
            "error": error,
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"写入请求日志失败: {e}")


# 全局请求日志记录器
request_logger = RequestLogger(REQUEST_LOG_ENABLED)


# ==================== HTTP 客户端 ====================

class ProxyClient:
    """带连接池的 HTTP 客户端"""

    def __init__(self):
        self.timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=TIMEOUT,
            write=TIMEOUT,
            pool=TIMEOUT,
        )
        self.limits = httpx.Limits(
            max_keepalive_connections=50,
            max_connections=100,
            keepalive_expiry=30,
        )
        self.client: Optional[httpx.AsyncClient] = None

    async def init(self):
        """初始化客户端"""
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            verify=False,
            limits=self.limits,
            http2=True,
        )

    async def close(self):
        """关闭客户端"""
        if self.client:
            await self.client.aclose()

    async def request(self, method: str, url: str, **kwargs):
        """发送请求（带重试）"""
        if not self.client:
            raise RuntimeError("客户端未初始化")

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    metrics.record_retry()
                    await asyncio.sleep(RETRY_DELAY * attempt)

                response = await self.client.request(method, url, **kwargs)
                return response

            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                last_error = e
                if circuit_breaker:
                    await circuit_breaker.record_failure()

            except httpx.TimeoutException as e:
                last_error = e

            except httpx.RemoteProtocolError as e:
                last_error = e
                if circuit_breaker:
                    await circuit_breaker.record_failure()

            except Exception as e:
                last_error = e
                raise

        # 所有重试都失败
        raise last_error

    async def stream(self, method: str, url: str, **kwargs):
        """流式请求"""
        if not self.client:
            raise RuntimeError("客户端未初始化")
        return self.client.stream(method, url, **kwargs)


# 全局客户端实例
proxy_client = ProxyClient()


# ==================== 生命周期管理 ====================

shutdown_event = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动
    logger.info("=" * 60)
    logger.info("OpenAI Proxy 服务器启动中...")
    logger.info(f"上游地址: {UPSTREAM_URL}")
    logger.info(f"监听地址: {HOST}:{PORT}")
    logger.info(f"日志级别: {LOG_LEVEL}")
    logger.info(f"限流: {'启用' if RATE_LIMIT_ENABLED else '禁用'}")
    logger.info(f"熔断: {'启用' if CIRCUIT_BREAKER_ENABLED else '禁用'}")
    logger.info("=" * 60)

    # 初始化 HTTP 客户端
    await proxy_client.init()

    # 启动清理任务
    cleanup_task = asyncio.create_task(cleanup_loop())

    yield

    # 关闭
    logger.info("正在关闭代理服务器...")
    shutdown_event.set()

    # 等待现有请求完成
    logger.info(f"等待现有请求完成 (最多 {SHUTDOWN_TIMEOUT} 秒)...")
    for _ in range(SHUTDOWN_TIMEOUT * 10):
        if metrics.requests_active == 0:
            break
        await asyncio.sleep(0.1)

    if metrics.requests_active > 0:
        logger.warning(f"关闭超时，仍有 {metrics.requests_active} 个活跃请求")

    # 取消清理任务
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # 关闭 HTTP 客户端
    await proxy_client.close()

    logger.info("代理服务器已关闭")


app = FastAPI(
    title="OpenAI Proxy",
    version="2.0.0",
    lifespan=lifespan,
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ==================== 清理任务 ====================

async def cleanup_loop():
    """定期清理任务"""
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(60)  # 每分钟清理一次

            # 清理限流器过期数据
            if rate_limiter:
                rate_limiter.cleanup()

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"清理任务错误: {e}")


# ==================== 路由 ====================

def get_client_ip(request: Request) -> str:
    """获取客户端 IP"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.get("/")
async def root():
    """健康检查"""
    upstream_status = "unknown"
    upstream_latency = 0

    try:
        start = time.time()
        response = await proxy_client.client.get(f"{UPSTREAM_URL}/", timeout=5)
        upstream_latency = (time.time() - start) * 1000
        upstream_status = "healthy" if response.status_code < 500 else "degraded"
    except Exception as e:
        upstream_status = "unreachable"
        logger.warning(f"上游健康检查失败: {e}")

    return {
        "service": "OpenAI Proxy",
        "version": "2.0.0",
        "status": "running",
        "upstream": UPSTREAM_URL,
        "upstream_status": upstream_status,
        "upstream_latency_ms": round(upstream_latency, 2),
    }


@app.get("/health")
async def health():
    """详细健康检查"""
    health_info = {
        "service": "OpenAI Proxy",
        "status": "healthy",
        "upstream": UPSTREAM_URL,
    }

    # 检查上游连接
    try:
        start = time.time()
        response = await proxy_client.client.get(f"{UPSTREAM_URL}/", timeout=5)
        latency = (time.time() - start) * 1000
        health_info["upstream_status"] = "healthy"
        health_info["upstream_latency_ms"] = round(latency, 2)
        health_info["upstream_code"] = response.status_code
    except Exception as e:
        health_info["upstream_status"] = "unreachable"
        health_info["upstream_error"] = str(e)

    # 熔断器状态
    if circuit_breaker:
        health_info["circuit_breaker"] = circuit_breaker.get_state()

    # 活跃请求
    health_info["active_requests"] = metrics.requests_active

    status_code = 200 if health_info["upstream_status"] == "healthy" else 503
    return JSONResponse(content=health_info, status_code=status_code)


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标"""
    if not METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="指标未启用")
    return Response(content=metrics.get_prometheus_metrics(), media_type="text/plain")


@app.get("/stats")
async def stats():
    """统计信息"""
    return {
        **metrics.get_stats(),
        "circuit_breaker": circuit_breaker.get_state() if circuit_breaker else None,
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(request: Request, path: str):
    """通用代理"""

    # 生成请求 ID
    request_id = str(uuid.uuid4())[:8]
    client_ip = get_client_ip(request)
    start_time = time.time()

    # 构建 URL
    url = f"{UPSTREAM_URL.rstrip('/')}/{path.lstrip('/')}"
    if request.url.query:
        url += f"?{request.url.query}"

    # 准备请求头
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    # OPTIONS 预检
    if request.method == "OPTIONS":
        return Response(status_code=200)

    # 获取请求体
    body = await request.body()

    # 限流检查
    if rate_limiter:
        if not await rate_limiter.is_allowed(client_ip):
            logger.warning(f"限流触发: {client_ip}")
            request_logger.log(
                request_id, request.method, path, 429,
                time.time() - start_time, client_ip, "Rate limited"
            )
            return JSONResponse(
                status_code=429,
                content={"error": {"message": "请求过于频繁，请稍后再试", "type": "rate_limit_error"}}
            )

    # 熔断器检查
    if circuit_breaker:
        if not await circuit_breaker.can_request():
            logger.warning(f"熔断器打开，拒绝请求: {request_id}")
            request_logger.log(
                request_id, request.method, path, 503,
                time.time() - start_time, client_ip, "Circuit breaker open"
            )
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "服务暂时不可用，请稍后再试", "type": "service_unavailable"}}
            )

    # 记录请求开始
    metrics.record_request(f"/{path}")

    try:
        # 检查是否为流式请求
        is_stream = False
        if body:
            try:
                body_json = json.loads(body)
                is_stream = body_json.get("stream", False)
            except:
                pass

        if is_stream:
            # 流式响应
            logger.info(f"[{request_id}] 流式请求: {request.method} /{path}")
            metrics.record_streaming_start()

            async def stream_generator():
                try:
                    async with await proxy_client.stream(
                        request.method,
                        url,
                        content=body if body else None,
                        headers=headers,
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk

                        # 成功完成
                        if circuit_breaker:
                            await circuit_breaker.record_success()

                except Exception as e:
                    logger.error(f"[{request_id}] 流式传输错误: {e}")
                    metrics.record_upstream_error()
                    if circuit_breaker:
                        await circuit_breaker.record_failure()
                    # 发送错误到客户端
                    yield f"data: {json.dumps({'error': {'message': f'流式传输错误: {str(e)}', 'type': 'stream_error'}})}\n\n".encode()

                finally:
                    metrics.record_streaming_end()
                    latency = time.time() - start_time
                    request_logger.log(request_id, request.method, path, 200, latency, client_ip)

            return StreamingResponse(
                stream_generator(),
                status_code=200,
                media_type="text/event-stream",
            )

        else:
            # 普通响应
            logger.info(f"[{request_id}] 请求: {request.method} /{path}")

            response = await proxy_client.request(
                method=request.method,
                url=url,
                content=body if body else None,
                headers=headers,
            )

            # 记录成功
            if circuit_breaker:
                await circuit_breaker.record_success()

            latency = time.time() - start_time
            metrics.record_response(response.status_code, latency)
            request_logger.log(request_id, request.method, path, response.status_code, latency, client_ip)

            logger.debug(f"[{request_id}] 响应: {response.status_code} ({latency*1000:.0f}ms)")

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
            )

    except HTTPException:
        raise
    except Exception as e:
        latency = time.time() - start_time
        metrics.record_response(500, latency)
        metrics.record_upstream_error()

        if circuit_breaker:
            await circuit_breaker.record_failure()

        logger.error(f"[{request_id}] 代理错误: {e}")
        request_logger.log(request_id, request.method, path, 500, latency, client_ip, str(e))

        raise HTTPException(status_code=500, detail=f"代理内部错误: {str(e)}")


# ==================== 信号处理 ====================

def handle_signal(signum, frame):
    """处理关闭信号"""
    logger.info(f"收到信号 {signum}，开始优雅关闭...")
    shutdown_event.set()


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL.lower(),
        access_log=True,
        use_colors=True,
        loop="asyncio",
    )
