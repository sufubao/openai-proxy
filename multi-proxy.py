#!/usr/bin/env python3
"""
多实例管理脚本
支持同时运行多个代理实例，每个实例对应不同的上游服务器
"""
import os
import sys
import signal
import time
import subprocess
from pathlib import Path

# 配置文件
CONFIG_FILE = "instances.conf"
# PID 目录
PID_DIR = "pids"
# 日志目录
LOG_DIR = "logs"


def read_instances():
    """读取实例配置"""
    instances = []
    config_path = Path(CONFIG_FILE)

    if not config_path.exists():
        print(f"错误: 配置文件 {CONFIG_FILE} 不存在")
        print("请先创建配置文件，格式: NAME|PORT|UPSTREAM_URL")
        sys.exit(1)

    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) != 3:
                print(f"警告: 跳过无效配置行: {line}")
                continue
            name, port, upstream = parts
            instances.append({
                "name": name.strip(),
                "port": int(port.strip()),
                "upstream": upstream.strip(),
            })

    return instances


def get_pid_file(name):
    """获取 PID 文件路径"""
    return Path(PID_DIR) / f"{name}.pid"


def get_log_file(name):
    """获取日志文件路径"""
    return Path(LOG_DIR) / f"{name}.log"


def start_instance(instance):
    """启动单个实例"""
    name = instance["name"]
    port = instance["port"]
    upstream = instance["upstream"]

    pid_file = get_pid_file(name)
    log_file = get_log_file(name)

    # 检查是否已运行
    if pid_file.exists():
        with open(pid_file) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)  # 检查进程是否存在
            print(f"  [{name}] 已运行 (PID: {pid}, 端口: {port})")
            return False
        except OSError:
            pid_file.unlink()  # 进程已不存在，删除 PID 文件

    # 启动进程
    env = os.environ.copy()
    env["UPSTREAM_URL"] = upstream
    env["PORT"] = str(port)
    env["HOST"] = "0.0.0.0"
    env["LOG_FILE"] = str(log_file)

    # 重定向日志
    log_handle = open(log_file, "a")

    proc = subprocess.Popen(
        [
            "gunicorn",
            "-c", "gunicorn_config.py",
            "main:app",
        ],
        env=env,
        stdout=log_handle,
        stderr=log_handle,
        start_new_session=True,
    )

    # 写入 PID 文件
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_file, "w") as f:
        f.write(str(proc.pid))

    print(f"  [{name}] 启动成功 (PID: {proc.pid}, 端口: {port}, 上游: {upstream})")
    return True


def stop_instance(instance):
    """停止单个实例"""
    name = instance["name"]
    port = instance["port"]

    pid_file = get_pid_file(name)

    if not pid_file.exists():
        print(f"  [{name}] 未运行")
        return False

    with open(pid_file) as f:
        pid = int(f.read().strip())

    try:
        # 先尝试优雅停止
        os.kill(pid, signal.SIGTERM)
        print(f"  [{name}] 发送停止信号 (PID: {pid})")

        # 等待进程结束
        for _ in range(30):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except OSError:
                break
        else:
            # 超时强制停止
            os.kill(pid, signal.SIGKILL)
            print(f"  [{name}] 强制停止")

        pid_file.unlink()
        print(f"  [{name}] 已停止")
        return True

    except OSError as e:
        print(f"  [{name}] 停止失败: {e}")
        pid_file.unlink()
        return False


def restart_instance(instance):
    """重启单个实例"""
    name = instance["name"]
    print(f"  [{name}] 重启中...")
    stop_instance(instance)
    time.sleep(2)
    return start_instance(instance)


def status_instance(instance):
    """查询单个实例状态"""
    name = instance["name"]
    port = instance["port"]
    upstream = instance["upstream"]

    pid_file = get_pid_file(name)

    status = "停止"
    pid = "-"

    if pid_file.exists():
        with open(pid_file) as f:
            pid = f.read().strip()
        try:
            os.kill(int(pid), 0)
            status = "运行中"
        except OSError:
            status = "停止 (PID 文件过期)"
            pid = f"{pid} (过期)"

    print(f"  [{name}] {status} | PID: {pid} | 端口: {port} | 上游: {upstream}")


def cmd_start():
    """启动所有实例"""
    print("启动代理实例...")
    instances = read_instances()

    Path(PID_DIR).mkdir(exist_ok=True)
    Path(LOG_DIR).mkdir(exist_ok=True)

    started = 0
    for inst in instances:
        if start_instance(inst):
            started += 1
        time.sleep(1)  # 错开启动时间

    print(f"\n已启动 {started}/{len(instances)} 个实例")


def cmd_stop():
    """停止所有实例"""
    print("停止代理实例...")
    instances = read_instances()

    stopped = 0
    for inst in instances:
        if stop_instance(inst):
            stopped += 1

    print(f"\n已停止 {stopped}/{len(instances)} 个实例")


def cmd_restart():
    """重启所有实例"""
    print("重启代理实例...")
    instances = read_instances()

    for inst in instances:
        restart_instance(inst)

    print(f"\n已重启 {len(instances)} 个实例")


def cmd_status():
    """查看所有实例状态"""
    print("代理实例状态:")
    print("-" * 80)
    instances = read_instances()

    for inst in instances:
        status_instance(inst)

    print("-" * 80)


def cmd_logs(instance_name=None):
    """查看日志"""
    instances = read_instances()

    if instance_name:
        instances = [i for i in instances if i["name"] == instance_name]
        if not instances:
            print(f"错误: 实例 '{instance_name}' 不存在")
            return

    for inst in instances:
        log_file = get_log_file(inst["name"])
        print(f"\n=== {inst['name']} 日志 (最后 20 行) ===")
        if log_file.exists():
            subprocess.run(["tail", "-20", str(log_file)])
        else:
            print("日志文件不存在")


def main():
    if len(sys.argv) < 2:
        print("OpenAI 代理多实例管理")
        print()
        print("用法: python multi-proxy.py COMMAND [INSTANCE]")
        print()
        print("命令:")
        print("  start [NAME]   启动实例 (未指定 NAME 则启动全部)")
        print("  stop [NAME]    停止实例")
        print("  restart [NAME] 重启实例")
        print("  status         查看状态")
        print("  logs [NAME]    查看日志")
        print()
        print("配置文件: instances.conf")
        sys.exit(1)

    command = sys.argv[1].lower()
    instance_name = sys.argv[2] if len(sys.argv) > 2 else None

    if command == "start":
        if instance_name:
            instances = [i for i in read_instances() if i["name"] == instance_name]
            if not instances:
                print(f"错误: 实例 '{instance_name}' 不存在")
                sys.exit(1)
            for inst in instances:
                start_instance(inst)
        else:
            cmd_start()

    elif command == "stop":
        if instance_name:
            instances = [i for i in read_instances() if i["name"] == instance_name]
            if not instances:
                print(f"错误: 实例 '{instance_name}' 不存在")
                sys.exit(1)
            for inst in instances:
                stop_instance(inst)
        else:
            cmd_stop()

    elif command == "restart":
        if instance_name:
            instances = [i for i in read_instances() if i["name"] == instance_name]
            if not instances:
                print(f"错误: 实例 '{instance_name}' 不存在")
                sys.exit(1)
            for inst in instances:
                restart_instance(inst)
        else:
            cmd_restart()

    elif command == "status":
        cmd_status()

    elif command == "logs":
        cmd_logs(instance_name)

    else:
        print(f"未知命令: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
