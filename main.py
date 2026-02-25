"""
OpenAI API Proxy - Production Grade
Features: process management, connection pool, rate limiting, circuit breaker, monitoring, logging, graceful shutdown
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

# ==================== Configuration ====================

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://10.42.53.44:8000")
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

# Timeout configuration
TIMEOUT = float(os.getenv("TIMEOUT", "300"))
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "30"))

# Retry configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1"))

# Rate limiting configuration
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))  # Requests per minute
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # Time window (seconds)

# Circuit breaker configuration
CIRCUIT_BREAKER_ENABLED = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))  # Failure threshold
CIRCUIT_BREAKER_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))  # Recovery timeout (seconds)

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/proxy.log")
REQUEST_LOG_ENABLED = os.getenv("REQUEST_LOG_ENABLED", "true").lower() == "true"

# Monitoring configuration
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"

# Graceful shutdown configuration
SHUTDOWN_TIMEOUT = int(os.getenv("SHUTDOWN_TIMEOUT", "30"))  # Graceful shutdown timeout (seconds)

# ==================== Logging setup ====================

# Create log directory
os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else ".", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ==================== Monitoring metrics ====================

class Metrics:
    """Prometheus-style metrics collection"""

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

        # Statistics by status code
        self.status_codes: Dict[int, int] = {}

        # Statistics by path
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
        """Export Prometheus format metrics"""
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

        # Status code distribution
        metrics.append("\n# HELP proxy_response_status Response status codes")
        metrics.append("# TYPE proxy_response_status counter")
        for code, count in self.status_codes.items():
            metrics.append(f'proxy_response_status{{status_code="{code}"}} {count}')

        # Path statistics
        metrics.append("\n# HELP proxy_path_requests Requests per path")
        metrics.append("# TYPE proxy_path_requests counter")
        for path, count in self.path_stats.items():
            safe_path = path.replace('"', '\\"')
            metrics.append(f'proxy_path_requests{{path="{safe_path}"}} {count}')

        return "\n".join(metrics)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
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


# Global metrics instance
metrics = Metrics()


# ==================== Circuit Breaker ====================

class CircuitBreaker:
    """Circuit breaker - prevent cascading failures"""

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
                logger.info("Circuit breaker recovered to closed state")
            self.failure_count = 0

    async def record_failure(self):
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.threshold and self.state != "open":
                self.state = "open"
                metrics.record_circuit_breaker_trip()
                logger.error(f"Circuit breaker opened! Failures: {self.failure_count}")

    async def can_request(self) -> bool:
        """Check if request can be sent"""
        async with self._lock:
            if self.state == "closed":
                return True

            if self.state == "open":
                # Check if recovery can be attempted
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = "half_open"
                    logger.info("Circuit breaker entering half-open state, attempting recovery")
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


# Global circuit breaker
circuit_breaker = CircuitBreaker(CIRCUIT_BREAKER_THRESHOLD, CIRCUIT_BREAKER_TIMEOUT) if CIRCUIT_BREAKER_ENABLED else None


# ==================== Rate Limiter ====================

class RateLimiter:
    """Sliding window rate limiter"""

    def __init__(self, requests: int, window: int):
        self.requests = requests
        self.window = window
        self.clients: Dict[str, deque] = {}

    async def is_allowed(self, client_id: str) -> bool:
        now = time.time()

        # Get or create client record
        if client_id not in self.clients:
            self.clients[client_id] = deque()

        # Remove expired records
        client_requests = self.clients[client_id]
        while client_requests and client_requests[0] < now - self.window:
            client_requests.popleft()

        # Check if limit exceeded
        if len(client_requests) >= self.requests:
            return False

        # Record this request
        client_requests.append(now)
        return True

    def cleanup(self):
        """Clean expired data"""
        now = time.time()
        for client_id in list(self.clients.keys()):
            client_requests = self.clients[client_id]
            while client_requests and client_requests[0] < now - self.window:
                client_requests.popleft()
            if not client_requests:
                del self.clients[client_id]


# Global rate limiter
rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW) if RATE_LIMIT_ENABLED else None


# ==================== Request Logging ====================

class RequestLogger:
    """Request logger"""

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
            logger.error(f"Failed to write request log: {e}")


# 全局Request logger器
request_logger = RequestLogger(REQUEST_LOG_ENABLED)


# ==================== HTTP Client ====================

class ProxyClient:
    """HTTP client with connection pool"""

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
        """Initialize client"""
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            verify=False,
            limits=self.limits,
            http2=True,
        )

    async def close(self):
        """Close client"""
        if self.client:
            await self.client.aclose()

    async def request(self, method: str, url: str, **kwargs):
        """Send request (with retry)"""
        if not self.client:
            raise RuntimeError("Client not initialized")

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

        # All retries failed
        raise last_error

    async def stream(self, method: str, url: str, **kwargs):
        """Streaming request"""
        if not self.client:
            raise RuntimeError("Client not initialized")
        return self.client.stream(method, url, **kwargs)


# Global client instance
proxy_client = ProxyClient()


# ==================== Lifecycle Management ====================

shutdown_event = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("=" * 60)
    logger.info("OpenAI Proxy Server Starting...")
    logger.info(f"Upstream: {UPSTREAM_URL}")
    logger.info(f"Listen: {HOST}:{PORT}")
    logger.info(f"Log Level: {LOG_LEVEL}")
    logger.info(f"Rate Limit: {'Enabled' if RATE_LIMIT_ENABLED else 'Disabled'}")
    logger.info(f"Circuit Breaker: {'Enabled' if CIRCUIT_BREAKER_ENABLED else 'Disabled'}")
    logger.info("=" * 60)

    # Initialize HTTP client
    await proxy_client.init()

    # StartupCleanup Task
    cleanup_task = asyncio.create_task(cleanup_loop())

    yield

    # Shutdown
    logger.info("Shutting down proxy server...")
    shutdown_event.set()

    # Wait for active requests
    logger.info(f"Waiting for active requests to complete (max {SHUTDOWN_TIMEOUT}s)...")
    for _ in range(SHUTDOWN_TIMEOUT * 10):
        if metrics.requests_active == 0:
            break
        await asyncio.sleep(0.1)

    if metrics.requests_active > 0:
        logger.warning(f"Shutdown timeout, {metrics.requests_active} active requests remaining")

    # Cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # Close HTTP client
    await proxy_client.close()

    logger.info("Proxy server shutdown complete")


app = FastAPI(
    title="OpenAI Proxy",
    version="2.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ==================== Cleanup Task ====================

async def cleanup_loop():
    """Periodic cleanup task"""
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(60)  # Clean every minute

            # Clean rate limiter expired data
            if rate_limiter:
                rate_limiter.cleanup()

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")


# ==================== Routes ====================

def get_client_ip(request: Request) -> str:
    """Get client IP"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.get("/")
async def root():
    """Health check"""
    upstream_status = "unknown"
    upstream_latency = 0

    try:
        start = time.time()
        response = await proxy_client.client.get(f"{UPSTREAM_URL}/", timeout=5)
        upstream_latency = (time.time() - start) * 1000
        upstream_status = "healthy" if response.status_code < 500 else "degraded"
    except Exception as e:
        upstream_status = "unreachable"
        logger.warning(f"Upstream health check failed: {e}")

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
    """Detailed health check"""
    health_info = {
        "service": "OpenAI Proxy",
        "status": "healthy",
        "upstream": UPSTREAM_URL,
    }

    # Check upstream connection
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

    # Circuit Breaker状态
    if circuit_breaker:
        health_info["circuit_breaker"] = circuit_breaker.get_state()

    # Active requests
    health_info["active_requests"] = metrics.requests_active

    status_code = 200 if health_info["upstream_status"] == "healthy" else 503
    return JSONResponse(content=health_info, status_code=status_code)


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics"""
    if not METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    return Response(content=metrics.get_prometheus_metrics(), media_type="text/plain")


@app.get("/stats")
async def stats():
    """Statistics"""
    return {
        **metrics.get_stats(),
        "circuit_breaker": circuit_breaker.get_state() if circuit_breaker else None,
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(request: Request, path: str):
    """Generic proxy"""

    # Generate request ID
    request_id = str(uuid.uuid4())[:8]
    client_ip = get_client_ip(request)
    start_time = time.time()

    # Build URL
    url = f"{UPSTREAM_URL.rstrip('/')}/{path.lstrip('/')}"
    if request.url.query:
        url += f"?{request.url.query}"

    # Prepare headers
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    # OPTIONS preflight
    if request.method == "OPTIONS":
        return Response(status_code=200)

    # Get request body
    body = await request.body()

    # Rate limit check
    if rate_limiter:
        if not await rate_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit triggered: {client_ip}")
            request_logger.log(
                request_id, request.method, path, 429,
                time.time() - start_time, client_ip, "Rate limited"
            )
            return JSONResponse(
                status_code=429,
                content={"error": {"message": "Too many requests, please try again later", "type": "rate_limit_error"}}
            )

    # Circuit Breaker检查
    if circuit_breaker:
        if not await circuit_breaker.can_request():
            logger.warning(f"Circuit breaker open, request rejected: {request_id}")
            request_logger.log(
                request_id, request.method, path, 503,
                time.time() - start_time, client_ip, "Circuit breaker open"
            )
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "Service temporarily unavailable, please try again later", "type": "service_unavailable"}}
            )

    # Record request start
    metrics.record_request(f"/{path}")

    try:
        # Check if streaming request
        is_stream = False
        if body:
            try:
                body_json = json.loads(body)
                is_stream = body_json.get("stream", False)
            except:
                pass

        if is_stream:
            # Streaming response
            logger.info(f"[{request_id}] Streaming request: {request.method} /{path}")
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

                        # Completed successfully
                        if circuit_breaker:
                            await circuit_breaker.record_success()

                except Exception as e:
                    logger.error(f"[{request_id}] Streaming error: {e}")
                    metrics.record_upstream_error()
                    if circuit_breaker:
                        await circuit_breaker.record_failure()
                    # Send error to client
                    yield f"data: {json.dumps({'error': {'message': f'Streaming transfer error: {str(e)}', 'type': 'stream_error'}})}\n\n".encode()

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
            # Regular response
            logger.info(f"[{request_id}] Request: {request.method} /{path}")

            response = await proxy_client.request(
                method=request.method,
                url=url,
                content=body if body else None,
                headers=headers,
            )

            # Record success
            if circuit_breaker:
                await circuit_breaker.record_success()

            latency = time.time() - start_time
            metrics.record_response(response.status_code, latency)
            request_logger.log(request_id, request.method, path, response.status_code, latency, client_ip)

            logger.debug(f"[{request_id}] Response: {response.status_code} ({latency*1000:.0f}ms)")

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

        logger.error(f"[{request_id}] Proxy error: {e}")
        request_logger.log(request_id, request.method, path, 500, latency, client_ip, str(e))

        raise HTTPException(status_code=500, detail=f"Internal proxy error: {str(e)}")


# ==================== Signal handling ====================

def handle_signal(signum, frame):
    """处理Shutdown信号"""
    logger.info(f"Received signal {signum}, starting graceful shutdown...")
    shutdown_event.set()


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def when_ready(server):
    """Gunicorn Startup完成回调"""
    print("OpenAI Proxy ready, accepting requests...")


def worker_int(worker):
    """Worker received SIGINT signal"""
    print(f"Worker {worker.pid} 收到中断信号")


def pre_fork(server, worker):
    """Pre-fork hook"""
    pass


def post_fork(server, worker):
    """Post-fork hook"""
    print(f"Worker {worker.pid} started")


def pre_exec(server):
    """Pre-exec hook in new master process"""
    print("New master process created")


def pre_request(worker, req):
    """Pre-request hook"""
    worker.log.debug(f"{req.method} {req.path}")


def post_request(worker, req, environ, resp):
    """Post-request hook"""
    pass


def child_exit(server, worker):
    """Child exit hook"""
    print(f"Worker {worker.pid} exited")


def worker_abort(worker):
    """Worker abnormal exit hook"""
    print(f"Worker {worker.pid} exited abnormally")


def nworkers_changed(server, new_value, old_value):
    """Worker count change hook"""
    print(f"Worker count: {old_value} -> {new_value}")


def on_exit(server):
    """Server exit hook"""
    print("Server shutting down...")


if __name__ == "__main__":
    # Single process dev mode
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
