import json

import pytest
from fastapi import HTTPException

from citetrace import server
from citetrace.schemas import FinalResult


def test_save_run_writes_auditable_result(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RUNS_PATH", tmp_path)

    run_id = server._save_run(
        "paper",
        "Was OpenVLA cited?",
        {"qwen": "qwen3.7-max"},
        {"answer": "Yes [E1].", "abstained": False},
    )

    saved = json.loads((tmp_path / f"{run_id}.json").read_text(encoding="utf-8"))
    assert saved["question"] == "Was OpenVLA cited?"
    assert saved["models"]["qwen"] == "qwen3.7-max"
    assert saved["result"]["answer"] == "Yes [E1]."
    assert saved["trace"] == []


def test_background_job_persists_progress_and_result(tmp_path, monkeypatch):
    jobs_path = tmp_path / "jobs"
    runs_path = tmp_path / "runs"
    monkeypatch.setattr(server, "JOBS_PATH", jobs_path)
    monkeypatch.setattr(server, "RUNS_PATH", runs_path)

    class FakePipeline:
        def run(self, question, paper_id, progress=None):
            progress("direct_retrieval", "Direct retrieval")
            progress("completed", "Done")
            return FinalResult(answer="Yes [E1].", evidences=[], verifications=[])

    class FakeStore:
        def load(self, paper_id):
            return []

    monkeypatch.setattr(server, "build_pipeline", lambda store, config: FakePipeline())
    monkeypatch.setattr(server, "store", FakeStore())
    job = server._create_job("paper", "Was OpenVLA cited?")
    config = {
        "QWEN_MODEL": "qwen",
        "DEEPSEEK_MODEL": "deepseek",
        "GLM_MODEL": "glm",
    }

    server._execute_job(job["job_id"], "paper", "Was OpenVLA cited?", config)

    saved = server._read_job(job["job_id"])
    assert saved["status"] == "completed"
    assert saved["stage"] == "completed"
    assert saved["result"]["answer"] == "Yes [E1]."
    assert saved["run_id"] == saved["result"]["run_id"]
    assert [item["stage"] for item in saved["stage_history"]] == [
        "direct_retrieval",
        "completed",
    ]
    assert all(item["duration_seconds"] >= 0 for item in saved["stage_history"])


def test_user_token_has_no_administrator_access(monkeypatch):
    monkeypatch.setattr(server, "_admin_token", lambda: "admin-secret")
    monkeypatch.setattr(server, "_user_token", lambda: "user-secret")

    auth = server._require_access("user-secret")

    assert auth.role == "user"
    with pytest.raises(HTTPException) as exc:
        server._require_admin(auth)
    assert exc.value.status_code == 403


def test_user_quota_enforces_concurrency_and_daily_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "JOBS_PATH", tmp_path / "jobs")
    monkeypatch.setattr(server, "USAGE_PATH", tmp_path / "usage.json")
    monkeypatch.setenv("CITETRACE_USER_DAILY_JOB_LIMIT", "1")
    monkeypatch.setenv("CITETRACE_USER_MAX_ACTIVE_JOBS", "1")

    server._reserve_user_job()
    job = server._create_job("paper", "question", owner_role="user")

    with pytest.raises(HTTPException) as active_error:
        server._reserve_user_job()
    assert active_error.value.status_code == 429
    assert "同时最多" in active_error.value.detail

    server._update_job(job["job_id"], status="completed")
    with pytest.raises(HTTPException) as daily_error:
        server._reserve_user_job()
    assert daily_error.value.status_code == 429
    assert "额度已用完" in daily_error.value.detail
