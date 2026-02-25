"""
压测脚本 - 测试并发性能
使用: python benchmark.py
"""
import asyncio
import aiohttp
import time
import statistics
from datetime import datetime, timedelta


class Benchmark:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []

    async def single_request(self, session: aiohttp.ClientSession, request_id: int):
        """发送单个请求"""
        start = time.time()
        try:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "say hello"}],
                    "max_tokens": 10
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                await response.text()
                latency = time.time() - start
                self.results.append({
                    "id": request_id,
                    "status": response.status,
                    "latency": latency,
                    "success": 200 <= response.status < 400
                })
        except Exception as e:
            latency = time.time() - start
            self.results.append({
                "id": request_id,
                "status": 0,
                "latency": latency,
                "success": False,
                "error": str(e)
            })

    async def run_concurrent(self, concurrent: int, total: int):
        """运行并发测试"""
        print(f"\n{'='*60}")
        print(f"并发测试: {concurrent} 并发, 总共 {total} 请求")
        print(f"{'='*60}")

        self.results = []
        start_time = time.time()

        connector = aiohttp.TCPConnector(limit=concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for i in range(total):
                tasks.append(self.single_request(session, i))
                # 控制并发
                if len(tasks) >= concurrent:
                    await asyncio.gather(*tasks)
                    tasks = []

            # 处理剩余任务
            if tasks:
                await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        self.print_results(total_time)

    def print_results(self, total_time: float):
        """打印测试结果"""
        successes = [r for r in self.results if r["success"]]
        failures = [r for r in self.results if not r["success"]]

        latencies = [r["latency"] for r in successes]
        qps = len(self.results) / total_time

        print(f"\n结果:")
        print(f"  总请求数:    {len(self.results)}")
        print(f"  成功:        {len(successes)} ({len(successes)/len(self.results)*100:.1f}%)")
        print(f"  失败:        {len(failures)} ({len(failures)/len(self.results)*100:.1f}%)")
        print(f"  总耗时:      {total_time:.2f}s")
        print(f"  QPS:         {qps:.2f}")

        if latencies:
            print(f"\n延迟统计 (成功请求):")
            print(f"  平均:        {statistics.mean(latencies)*1000:.0f}ms")
            print(f"  中位数:      {statistics.median(latencies)*1000:.0f}ms")
            print(f"  P95:         {sorted(latencies)[int(len(latencies)*0.95)]*1000:.0f}ms")
            print(f"  P99:         {sorted(latencies)[int(len(latencies)*0.99)]*1000:.0f}ms")
            print(f"  最小:        {min(latencies)*1000:.0f}ms")
            print(f"  最大:        {max(latencies)*1000:.0f}ms")


async def main():
    benchmark = Benchmark()

    print("=" * 60)
    print("OpenAI Proxy 压力测试")
    print("=" * 60)
    print(f"目标: {benchmark.base_url}")
    print("提示: 先启动代理服务")
    print("")

    # 测试场景
    scenarios = [
        (10, 50),    # 10 并发, 50 请求
        (50, 200),   # 50 并发, 200 请求
        (100, 500),  # 100 并发, 500 请求
        (200, 1000), # 200 并发, 1000 请求
    ]

    for concurrent, total in scenarios:
        try:
            await benchmark.run_concurrent(concurrent, total)
            await asyncio.sleep(2)  # 冷却时间
        except KeyboardInterrupt:
            print("\n测试中断")
            break

    print(f"\n{'='*60}")
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
