from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键启动 Streamlit 和 FastAPI 服务。")
    parser.add_argument("--host", default="127.0.0.1", help="FastAPI 监听地址")
    parser.add_argument("--api-port", type=int, default=8000, help="FastAPI 端口")
    parser.add_argument("--streamlit-port", type=int, default=8501, help="Streamlit 端口")
    return parser.parse_args()


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def spawn_process(name: str, command: list[str], env: dict[str, str]) -> subprocess.Popen:
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    return subprocess.Popen(
        command,
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )


def pipe_output(name: str, process: subprocess.Popen) -> threading.Thread:
    def _reader() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            print(f"[{name}] {line.rstrip()}")

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return thread


def stop_process(name: str, process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    print(f"[launcher] stopping {name}...")
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    args = parse_args()
    env = build_env()
    python_executable = sys.executable

    api_command = [
        python_executable,
        "-m",
        "uvicorn",
        "api.api_service:app",
        "--host",
        args.host,
        "--port",
        str(args.api_port),
    ]
    streamlit_command = [
        python_executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "true",
        "--server.port",
        str(args.streamlit_port),
    ]

    print("[launcher] starting services...")
    print(f"[launcher] api        -> http://{args.host}:{args.api_port}")
    print(f"[launcher] streamlit  -> http://127.0.0.1:{args.streamlit_port}")
    print("[launcher] press Ctrl+C to stop both services")

    processes: list[tuple[str, subprocess.Popen]] = []

    try:
        api_process = spawn_process("api", api_command, env)
        pipe_output("api", api_process)
        processes.append(("api", api_process))

        time.sleep(1)

        streamlit_process = spawn_process("streamlit", streamlit_command, env)
        pipe_output("streamlit", streamlit_process)
        processes.append(("streamlit", streamlit_process))

        while True:
            for name, process in processes:
                code = process.poll()
                if code is not None:
                    print(f"[launcher] {name} exited with code {code}")
                    for other_name, other_process in reversed(processes):
                        if other_process is not process:
                            stop_process(other_name, other_process)
                    return code
            time.sleep(0.8)
    except KeyboardInterrupt:
        print("[launcher] received Ctrl+C")
        return 0
    finally:
        for name, process in reversed(processes):
            stop_process(name, process)


if __name__ == "__main__":
    raise SystemExit(main())
