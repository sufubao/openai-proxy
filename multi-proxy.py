#!/usr/bin/env python3
"""
Multi-instance Management Script
Manage multiple proxy instances, each connecting to a different upstream server
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
    """Read instance configuration"""
    instances = []
    config_path = Path(CONFIG_FILE)

    if not config_path.exists():
        print(f"Error: Config file {CONFIG_FILE} not found")
        print("Please create config file, format: NAME|PORT|UPSTREAM_URL")
        sys.exit(1)

    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) != 3:
                print(f"Warning: Skipping invalid config line: {line}")
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

    # Check if already running
    if pid_file.exists():
        with open(pid_file) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)  # Check if process exists
            print(f"  [{name}] Already running (PID: {pid}, Port: {port})")
            return False
        except OSError:
            pid_file.unlink()  # Process doesn't exist, remove PID file

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

    # Write PID file
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_file, "w") as f:
        f.write(str(proc.pid))

    print(f"  [{name}] Started (PID: {proc.pid}, Port: {port}, Upstream: {upstream})")
    return True


def stop_instance(instance):
    """停止单个实例"""
    name = instance["name"]
    port = instance["port"]

    pid_file = get_pid_file(name)

    if not pid_file.exists():
        print(f"  [{name}] Not running")
        return False

    with open(pid_file) as f:
        pid = int(f.read().strip())

    try:
        # Try graceful shutdown first
        os.kill(pid, signal.SIGTERM)
        print(f"  [{name}] Sent stop signal (PID: {pid})")

        # Wait for process to end
        for _ in range(30):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except OSError:
                break
        else:
            # Force kill on timeout
            os.kill(pid, signal.SIGKILL)
            print(f"  [{name}] Force killed")

        pid_file.unlink()
        print(f"  [{name}] Stopped")
        return True

    except OSError as e:
        print(f"  [{name}] Stop failed: {e}")
        pid_file.unlink()
        return False


def restart_instance(instance):
    """Restart single instance"""
    name = instance["name"]
    print(f"  [{name}] Restarting...")
    stop_instance(instance)
    time.sleep(2)
    return start_instance(instance)


def status_instance(instance):
    """Query single instance status"""
    name = instance["name"]
    port = instance["port"]
    upstream = instance["upstream"]

    pid_file = get_pid_file(name)

    status = "Stopped"
    pid = "-"

    if pid_file.exists():
        with open(pid_file) as f:
            pid = f.read().strip()
        try:
            os.kill(int(pid), 0)
            status = "Running"
        except OSError:
            status = "Stopped (stale PID)"
            pid = f"{pid} (stale)"

    print(f"  [{name}] {status} | PID: {pid} | Port: {port} | Upstream: {upstream}")


def cmd_start():
    """Start all instances"""
    print("Starting proxy instances...")
    instances = read_instances()

    Path(PID_DIR).mkdir(exist_ok=True)
    Path(LOG_DIR).mkdir(exist_ok=True)

    started = 0
    for inst in instances:
        if start_instance(inst):
            started += 1
        time.sleep(1)  # Stagger startup

    print(f"\nStarted {started}/{len(instances)} instances")


def cmd_stop():
    """Stop all instances"""
    print("Stopping proxy instances...")
    instances = read_instances()

    stopped = 0
    for inst in instances:
        if stop_instance(inst):
            stopped += 1

    print(f"\nStopped {stopped}/{len(instances)} instances")


def cmd_restart():
    """Restart all instances"""
    print("Restarting proxy instances...")
    instances = read_instances()

    for inst in instances:
        restart_instance(inst)

    print(f"\nRestarted {len(instances)} instances")


def cmd_status():
    """Show all instances status"""
    print("Proxy Instances Status:")
    print("-" * 80)
    instances = read_instances()

    for inst in instances:
        status_instance(inst)

    print("-" * 80)


def cmd_logs(instance_name=None):
    """View logs"""
    instances = read_instances()

    if instance_name:
        instances = [i for i in instances if i["name"] == instance_name]
        if not instances:
            print(f"Error: Instance '{instance_name}' not found")
            return

    for inst in instances:
        log_file = get_log_file(inst["name"])
        print(f"\n=== {inst['name']} Logs (last 20 lines) ===")
        if log_file.exists():
            subprocess.run(["tail", "-20", str(log_file)])
        else:
            print("Log file not found")


def main():
    if len(sys.argv) < 2:
        print("OpenAI Proxy Multi-Instance Manager")
        print()
        print("Usage: python multi-proxy.py COMMAND [INSTANCE]")
        print()
        print("Commands:")
        print("  start [NAME]   Start instance(s)")
        print("  stop [NAME]    Stop instance(s)")
        print("  restart [NAME] Restart instance(s)")
        print("  status         Show status")
        print("  logs [NAME]    View logs")
        print()
        print("Config file: instances.conf")
        sys.exit(1)

    command = sys.argv[1].lower()
    instance_name = sys.argv[2] if len(sys.argv) > 2 else None

    if command == "start":
        if instance_name:
            instances = [i for i in read_instances() if i["name"] == instance_name]
            if not instances:
                print(f"Error: Instance '{instance_name}' not found")
                sys.exit(1)
            for inst in instances:
                start_instance(inst)
        else:
            cmd_start()

    elif command == "stop":
        if instance_name:
            instances = [i for i in read_instances() if i["name"] == instance_name]
            if not instances:
                print(f"Error: Instance '{instance_name}' not found")
                sys.exit(1)
            for inst in instances:
                stop_instance(inst)
        else:
            cmd_stop()

    elif command == "restart":
        if instance_name:
            instances = [i for i in read_instances() if i["name"] == instance_name]
            if not instances:
                print(f"Error: Instance '{instance_name}' not found")
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
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
