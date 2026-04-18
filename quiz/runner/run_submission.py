from __future__ import annotations

import importlib
import importlib.util
import io
import json
import sys
import types
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from quiz.questions.hidden_cases import build_hidden_tests
from quiz.questions.data import QUESTIONS_BY_ID


def load_solution_module(solution_path: Path):
    spec = importlib.util.spec_from_file_location("learner_solution", solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load solution module from {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def serialize_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    if isinstance(value, types.ModuleType):
        return f"<module {value.__name__}>"

    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "tolist"):
        try:
            return {
                "kind": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "value": value.tolist(),
            }
        except Exception:
            return repr(value)

    return repr(value)


def extract_test_snapshot(frame_locals: dict[str, object]) -> dict[str, object]:
    filtered_inputs = {}
    for key, value in frame_locals.items():
        if key in {"solution_module", "actual", "expected", "target", "assertion", "case_args", "case_assertions", "case_kwargs", "symbol"}:
            continue
        if key.startswith("__"):
            continue
        filtered_inputs[key] = serialize_value(value)

    snapshot = {
        "input_snapshot": filtered_inputs,
        "expected_output": serialize_value(frame_locals["expected"]) if "expected" in frame_locals else None,
        "actual_output": serialize_value(frame_locals["actual"]) if "actual" in frame_locals else None,
    }
    return snapshot


def run_test_with_snapshot(test_fn, solution_module):
    captured: dict[str, object] = {"input_snapshot": {}, "expected_output": None, "actual_output": None}
    target_code = test_fn.__code__

    def tracer(frame, event, arg):
        if frame.f_code is not target_code:
            return tracer
        if event in {"return", "exception"}:
            captured.update(extract_test_snapshot(frame.f_locals))
        return tracer

    previous_trace = sys.gettrace()
    sys.settrace(tracer)
    try:
        test_fn(solution_module)
    finally:
        sys.settrace(previous_trace)

    return captured


def extract_snapshot_from_traceback(test_fn, exc_tb):
    target_code = test_fn.__code__
    current = exc_tb
    while current is not None:
        if current.tb_frame.f_code is target_code:
            return extract_test_snapshot(current.tb_frame.f_locals)
        current = current.tb_next
    return {"input_snapshot": {}, "expected_output": None, "actual_output": None}


def normalize_test_case(test_case):
    if len(test_case) == 2:
        name, test_fn = test_case
        metadata = {}
    elif len(test_case) == 3:
        name, test_fn, metadata = test_case
    else:
        raise ValueError("Test cases must have 2 or 3 items.")
    return name, test_fn, metadata


def select_tests(question, mode: str):
    if mode == "submit" and question.tests.hidden_binary_key and question.answer.symbol_name:
        generated = build_hidden_tests(question.tests.hidden_binary_key, question.answer.symbol_name)
        if generated:
            return generated

    try:
        test_module = importlib.import_module(question.tests.module_name)
    except ModuleNotFoundError:
        if question.tests.hidden_binary_key:
            return []
        raise
    if mode == "run":
        return test_module.get_visible_tests()
    return test_module.get_hidden_tests()


def run_tests(question, solution_path: Path, mode: str) -> dict[str, object]:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        solution_module = load_solution_module(solution_path)
        tests = select_tests(question, mode)

        results: list[dict[str, object]] = []
        passed = 0
        for test_case in tests:
            name, test_fn, metadata = normalize_test_case(test_case)
            try:
                snapshot = run_test_with_snapshot(test_fn, solution_module)
                results.append(
                    {
                        "name": name,
                        "passed": True,
                        "hint": metadata.get("hint", "Compare your result carefully to the expected behavior for this test."),
                        **snapshot,
                    }
                )
                passed += 1
            except AssertionError as exc:
                snapshot = extract_snapshot_from_traceback(test_fn, sys.exc_info()[2])
                results.append(
                    {
                        "name": name,
                        "passed": False,
                        "message": str(exc),
                        "hint": metadata.get("hint", "Check the function output, tensor shape, and dtype against the prompt."),
                        **snapshot,
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive path
                results.append(
                    {
                        "name": name,
                        "passed": False,
                        "message": f"{type(exc).__name__}: {exc}",
                        "hint": metadata.get("hint", "The test crashed before finishing. Re-check the function name, return type, and core logic."),
                        "input_snapshot": {},
                        "expected_output": None,
                        "actual_output": None,
                    }
                )

    return {
        "ok": passed == len(tests),
        "mode": mode,
        "passed": passed,
        "total": len(tests),
        "results": results,
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
    }


def main() -> int:
    if len(sys.argv) != 4:
        print(json.dumps({"ok": False, "error": "Usage: question_id mode solution_path"}))
        return 1

    question_id, mode, solution_path_text = sys.argv[1:]
    question = QUESTIONS_BY_ID.get(question_id)
    if question is None:
        print(json.dumps({"ok": False, "error": f"Unknown question id: {question_id}"}))
        return 1

    solution_path = Path(solution_path_text)

    try:
        payload = run_tests(question, solution_path, mode)
        print(json.dumps(payload))
        return 0 if payload["ok"] else 0
    except Exception as exc:
        payload = {
            "ok": False,
            "mode": mode,
            "passed": 0,
            "total": 0,
            "results": [],
            "stdout": "",
            "stderr": traceback.format_exc(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(payload))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
