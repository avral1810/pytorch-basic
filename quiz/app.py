from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory

from quiz.content.data import get_chapter, public_chapter_summaries
from quiz.questions.data import QUESTIONS_BY_ID, get_public_chapter_detail
from quiz.runner.persistent import PersistentRunner, runner_python


ROOT_DIR = Path(__file__).resolve().parent.parent
QUIZ_DIR = ROOT_DIR / "quiz"
CONTENT_ASSET_DIR = QUIZ_DIR / "content" / "assets" / "chapters"
EXECUTION_TIMEOUT_SECONDS = int(os.environ.get("QUIZ_EXECUTION_TIMEOUT_SECONDS", "30"))
RUNNER = PersistentRunner()

app = Flask(__name__, template_folder=str(QUIZ_DIR / "templates"), static_folder=str(QUIZ_DIR / "static"))


def timeout_error(message_prefix: str) -> str:
    return f"{message_prefix} timed out after {EXECUTION_TIMEOUT_SECONDS} seconds."


def execute_submission(chapter_id: str, question_id: str, code: str, mode: str) -> tuple[dict[str, object], int]:
    chapter = get_chapter(chapter_id)
    question = QUESTIONS_BY_ID.get(question_id)

    if chapter is None:
        return {"ok": False, "error": f"Unknown chapter id: {chapter_id}"}, 404
    if question is None:
        return {"ok": False, "error": f"Unknown question id: {question_id}"}, 404
    if question.chapter_id != chapter_id or question_id not in chapter["question_ids"]:
        return {"ok": False, "error": "Question does not belong to the requested chapter."}, 400

    try:
        payload = RUNNER.request(
            {
                "type": "submission",
                "question_id": question_id,
                "mode": mode,
                "code": code,
            },
            timeout_seconds=EXECUTION_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        return {
            "ok": False,
            "mode": mode,
            "passed": 0,
            "total": 0,
            "results": [],
            "stdout": "",
            "stderr": "",
            "error": timeout_error("Execution"),
        }, 200

    return payload, 200


def execute_playground(code: str) -> tuple[dict[str, object], int]:
    try:
        payload = RUNNER.request(
            {
                "type": "playground",
                "code": code,
            },
            timeout_seconds=EXECUTION_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        return {"ok": False, "stdout": "", "stderr": "", "error": timeout_error("Playground execution")}, 200

    return payload, 200


@app.get("/")
def index():
    return render_template("home.html", chapters=public_chapter_summaries())


@app.get("/chapters/<chapter_id>")
def chapter_page(chapter_id: str):
    chapter = get_public_chapter_detail(chapter_id)
    if chapter is None:
        abort(404)
    return render_template("chapter.html", chapter=chapter, chapters=public_chapter_summaries())


@app.get("/api/chapters")
def chapters():
    return jsonify({"chapters": public_chapter_summaries()})


@app.get("/api/chapters/<chapter_id>")
def chapter_detail(chapter_id: str):
    chapter = get_public_chapter_detail(chapter_id)
    if chapter is None:
        return jsonify({"ok": False, "error": f"Unknown chapter id: {chapter_id}"}), 404
    return jsonify(chapter)


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "python": runner_python(), "execution_timeout_seconds": EXECUTION_TIMEOUT_SECONDS})


@app.get("/chapter-assets/<path:filename>")
def chapter_assets(filename: str):
    return send_from_directory(CONTENT_ASSET_DIR, filename)


@app.get("/resources/<path:resource_path>")
def resource_file(resource_path: str):
    target = (ROOT_DIR / resource_path).resolve()
    root = ROOT_DIR.resolve()

    if not str(target).startswith(str(root)) or not target.exists() or not target.is_file():
        abort(404)

    return send_file(target)


@app.post("/api/playground")
def run_playground():
    payload = request.get_json(force=True)
    result, status = execute_playground(payload["code"])
    return jsonify(result), status


@app.post("/api/run")
def run_code():
    payload = request.get_json(force=True)
    result, status = execute_submission(payload["chapter_id"], payload["question_id"], payload["code"], "run")
    return jsonify(result), status


@app.post("/api/submit")
def submit_code():
    payload = request.get_json(force=True)
    result, status = execute_submission(payload["chapter_id"], payload["question_id"], payload["code"], "submit")
    return jsonify(result), status


if __name__ == "__main__":
    try:
        RUNNER.request({"type": "warmup"}, timeout_seconds=EXECUTION_TIMEOUT_SECONDS)
    except TimeoutError:
        RUNNER.close()
    app.run(host="127.0.0.1", port=8000, debug=False)
