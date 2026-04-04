from __future__ import annotations

import io
import json
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from quiz.runner.run_submission import run_tests
from quiz.questions.data import QUESTIONS_BY_ID


def execute_playground(code: str) -> dict[str, object]:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    namespace = {"__name__": "__main__"}

    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        exec(compile(code, "<playground>", "exec"), namespace, namespace)

    return {
        "ok": True,
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
        "error": "",
    }


def execute_submission(question_id: str, mode: str, code: str) -> dict[str, object]:
    question = QUESTIONS_BY_ID.get(question_id)
    if question is None:
        return {"ok": False, "error": f"Unknown question id: {question_id}"}

    solution_path = Path("/tmp") / "quiz_worker_solution.py"
    solution_path.write_text(code, encoding="utf-8")
    return run_tests(question, solution_path, mode)


def handle_request(payload: dict[str, object]) -> dict[str, object]:
    request_type = payload.get("type")

    if request_type == "warmup":
        import torch  # noqa: F401

        return {"ok": True, "warmed": True}

    if request_type == "playground":
        return execute_playground(str(payload["code"]))

    if request_type == "submission":
        return execute_submission(str(payload["question_id"]), str(payload["mode"]), str(payload["code"]))

    return {"ok": False, "error": f"Unknown worker request type: {request_type}"}


def main() -> int:
    while True:
        try:
            line = input()
        except EOFError:
            return 0

        payload = json.loads(line)

        try:
            response = handle_request(payload)
        except Exception as exc:  # pragma: no cover - defensive path
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "stderr": traceback.format_exc(),
            }

        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
