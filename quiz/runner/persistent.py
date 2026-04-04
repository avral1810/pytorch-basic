from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
QUIZ_DIR = ROOT_DIR / "quiz"
WORKER_MODULE = "quiz.runner.worker_process"


def runner_python() -> str:
    local_python = QUIZ_DIR / ".venv" / "bin" / "python"
    if local_python.exists():
        return str(local_python)
    return sys.executable


class PersistentRunner:
    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, object]] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._stderr_lines: list[str] = []
        self._lock = threading.Lock()

    def _start_worker(self) -> None:
        process = subprocess.Popen(
            [runner_python(), "-m", WORKER_MODULE],
            cwd=str(ROOT_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._process = process
        self._reader = threading.Thread(target=self._read_stdout, args=(process,), daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, args=(process,), daemon=True)
        self._stderr_reader.start()

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._responses.put(json.loads(line))
            except json.JSONDecodeError:
                self._responses.put(
                    {
                        "ok": False,
                        "error": "Worker output could not be parsed as JSON.",
                        "stdout": line,
                    }
                )

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            line = line.rstrip()
            if not line:
                continue
            self._stderr_lines.append(line)
            if len(self._stderr_lines) > 20:
                self._stderr_lines = self._stderr_lines[-20:]

    def _ensure_running(self) -> None:
        if self._process is None or self._process.poll() is not None:
            self.close()
            self._start_worker()

    def request(self, payload: dict[str, object], timeout_seconds: int) -> dict[str, object]:
        with self._lock:
            self._ensure_running()
            assert self._process is not None
            assert self._process.stdin is not None
            self._stderr_lines = []

            while not self._responses.empty():
                try:
                    self._responses.get_nowait()
                except queue.Empty:
                    break

            try:
                self._process.stdin.write(json.dumps(payload) + "\n")
                self._process.stdin.flush()
            except BrokenPipeError:
                self.close()
                self._ensure_running()
                assert self._process is not None
                assert self._process.stdin is not None
                self._process.stdin.write(json.dumps(payload) + "\n")
                self._process.stdin.flush()

            try:
                return self._responses.get(timeout=timeout_seconds)
            except queue.Empty:
                if self._process.poll() is not None:
                    stderr = "\n".join(self._stderr_lines)
                    self.close()
                    return {
                        "ok": False,
                        "error": "Worker exited before returning a response.",
                        "stderr": stderr,
                        "stdout": "",
                    }
                self.close()
                raise TimeoutError

    def close(self) -> None:
        if self._process is None:
            return

        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()

        self._process = None
        self._reader = None
        self._stderr_reader = None
