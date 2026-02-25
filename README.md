# OpenAI API 代理服务器 (生产级)

一个高可用的 OpenAI API 格式兼容代理服务器，用于转发请求到内部 LLM 服务。

## 功能特性

- ✅ **完全兼容 OpenAI API 格式** - 支持所有端点
- ✅ **自动重试** - 连接失败、超时自动重试
- ✅ **限流保护** - 防止单个客户端过度请求
- ✅ **熔断器** - 上游故障时快速失败，防止雪崩
- ✅ **连接池** - HTTP/2 支持，复用连接
- ✅ **请求日志** - 记录所有请求用于审计
- ✅ **Prometheus 指标** - 可导出监控数据
- ✅ **优雅关闭** - 等待现有请求完成后再退出
- ✅ **Docker 部署** - 包含健康检查和自动重启

## 安装

### 方式一：直接运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行
python main.py
```

### 方式二：使用部署脚本 (推荐)

```bash
./deploy.sh

# 使用 supervisor 启动
supervisord -c supervisord.conf

# 查看状态
supervisorctl -c supervisord.conf status
```

### 方式三：Docker 部署

```bash
# 构建镜像
docker build -t openai-proxy .

# 运行
docker run -d \
  --name openai-proxy \
  -p 8000:8000 \
  -e UPSTREAM_URL=http://10.42.53.44:8000 \
  -v $(pwd)/logs:/app/logs \
  --restart unless-stopped \
  openai-proxy
```

### 方式四：Docker Compose (推荐)

```bash
# 启动代理
docker-compose up -d

# 启动代理 + 监控
docker-compose --profile monitoring up -d

# 查看日志
docker-compose logs -f
```

## 配置

通过环境变量配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `UPSTREAM_URL` | `http://10.42.53.44:8000` | 上游 LLM 服务地址 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8000` | 监听端口 |
| `TIMEOUT` | `300` | 总超时时间(秒) |
| `CONNECT_TIMEOUT` | `30` | 连接超时(秒) |
| `MAX_RETRIES` | `3` | 最大重试次数 |
| `RETRY_DELAY` | `1` | 重试延迟(秒) |
| `RATE_LIMIT_ENABLED` | `true` | 是否启用限流 |
| `RATE_LIMIT_REQUESTS` | `100` | 每分钟请求数限制 |
| `RATE_LIMIT_WINDOW` | `60` | 限流时间窗口(秒) |
| `CIRCUIT_BREAKER_ENABLED` | `true` | 是否启用熔断器 |
| `CIRCUIT_BREAKER_THRESHOLD` | `5` | 熔断失败阈值 |
| `CIRCUIT_BREAKER_TIMEOUT` | `60` | 熔断恢复时间(秒) |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `REQUEST_LOG_ENABLED` | `true` | 是否记录请求日志 |
| `METRICS_ENABLED` | `true` | 是否启用 Prometheus 指标 |
| `SHUTDOWN_TIMEOUT` | `30` | 优雅关闭超时(秒) |

## 监控端点

| 端点 | 说明 |
|------|------|
| `GET /` | 基础健康检查 |
| `GET /health` | 详细健康检查 |
| `GET /metrics` | Prometheus 格式指标 |
| `GET /stats` | JSON 格式统计信息 |

```bash
# 检查服务状态
curl http://localhost:8000/health

# 查看 Prometheus 指标
curl http://localhost:8000/metrics

# 查看统计信息
curl http://localhost:8000/stats
```

## 日志

日志文件位置：
- `logs/proxy.log` - 应用日志
- `logs/requests.log` - 请求日志 (JSON Lines 格式)
- `logs/supervisord.log` - Supervisor 日志
- `logs/proxy.out.log` - 标准输出
- `logs/proxy.err.log` - 错误输出

## 使用示例

### 使用 curl

```bash
# 健康检查
curl http://localhost:8000/

# 获取模型列表
curl http://localhost:8000/v1/models

# 聊天补全
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 使用 Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://your-proxy-host:8000/v1",
    api_key="any-key"  # 根据上游服务配置
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### 使用其他语言

任何支持 HTTP 的语言都可以直接调用，只需设置 `base_url` 指向代理服务器即可。

## 故障处理

| 状态码 | 说明 | 处理 |
|--------|------|------|
| 429 | 限流触发 | 降低请求频率 |
| 502 | 上游协议错误 | 检查上游服务 |
| 503 | 熔断器打开或上游不可达 | 等待熔断器恢复 |
| 504 | 上游超时 | 增加 TIMEOUT 配置 |

## 生产建议

1. **使用 Docker Compose** - 包含自动重启和健康检查
2. **启用监控** - 使用 Prometheus + Grafana 监控指标
3. **配置日志轮转** - 防止日志文件过大
4. **设置限流** - 防止单个客户端消耗过多资源
5. **调整超时** - 根据实际 LLM 响应时间调整 TIMEOUT
6. **多实例部署** - 使用负载均衡器分发请求
