import io
import json
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest
from fastapi import BackgroundTasks, HTTPException
from fastapi.testclient import TestClient

from citetrace import server
from citetrace.llm.openai_compatible import OpenAICompatibleLLM
from citetrace.llm.validation import InvalidModelOutput
from citetrace.paper.parser import PaperChunk
from citetrace.paper.store import JsonPaperStore


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(server, "JOBS_PATH", tmp_path / "jobs")
    monkeypatch.setattr(server, "RUNS_PATH", tmp_path / "runs")
    monkeypatch.setattr(server, "USAGE_PATH", tmp_path / "usage.json")
    store = JsonPaperStore(tmp_path / "data/index")
    monkeypatch.setattr(server, "store", store)
    monkeypatch.setenv("CITETRACE_USER_DAILY_JOB_LIMIT", "20")
    monkeypatch.setenv("CITETRACE_USER_MAX_ACTIVE_JOBS", "1")
    store.save("paper", [PaperChunk("old", "paper", 1, "original content")], {"total_pages": 1})
    return tmp_path


@pytest.fixture
def client(isolated):
    server.app.dependency_overrides[server._require_access] = lambda: server.AuthContext("admin")
    try:
        with TestClient(server.app) as client:
            yield client
    finally:
        server.app.dependency_overrides.clear()


def pdf_bytes():
    with fitz.open() as doc:
        doc.new_page().insert_text((72, 72), "replacement content")
        return doc.tobytes()


@pytest.mark.parametrize("failure", ["invalid", "empty", "oversize", "metadata", "commit"])
def test_failed_upload_preserves_old_pdf_index_and_metadata(client, isolated, monkeypatch, failure):
    uploads = isolated / "data/uploads"
    uploads.mkdir(parents=True)
    old_pdf = uploads / "paper.pdf"
    # This is a fixture, not a production document.
    old_pdf.write_bytes(pdf_bytes())
    before_pdf = old_pdf.read_bytes()
    index = isolated / "data/index/paper.json"
    before_index = index.read_bytes()
    body = pdf_bytes()
    if failure == "invalid":
        body = b"not a PDF"
    elif failure == "empty":
        body = b""
    elif failure == "oversize":
        monkeypatch.setattr(server, "_MAX_PDF_BYTES", 8)
    elif failure == "metadata":
        def fail_metadata(*args):
            raise RuntimeError("metadata failure")
        monkeypatch.setattr(server, "build_pdf_metadata", fail_metadata)
    else:
        original_replace = Path.replace
        def fail_commit(self, target):
            if Path(target) == index:
                raise OSError("simulated disk failure")
            return original_replace(self, target)
        monkeypatch.setattr(Path, "replace", fail_commit)
    response = client.post("/api/papers", data={"paper_id": "paper"}, files={"file": ("new.pdf", body)})
    assert response.status_code == {"oversize": 413, "commit": 503}.get(failure, 422)
    assert old_pdf.read_bytes() == before_pdf
    assert index.read_bytes() == before_index
    assert list(uploads.iterdir()) == [old_pdf]
    assert not server._SUBMISSION_LOCK.locked()
    assert client.get("/api/papers/paper").json()["status"] == "ready"


def test_successful_upload_commits_matching_pdf_and_index(client, isolated):
    body = pdf_bytes()
    response = client.post("/api/papers", data={"paper_id": "paper"}, files={"file": ("new.pdf", body)})
    assert response.status_code == 200
    index = json.loads((isolated / "data/index/paper.json").read_text())
    assert (isolated / "data/uploads" / index["pdf_filename"]).read_bytes() == body
    assert index["metadata"]["total_pages"] == 1
    assert "replacement content" in server.store.load("paper")[0].text
    assert server.store.search("paper", "replacement")


def test_upload_during_question_is_rejected_without_changing_index(client, isolated):
    server._create_job("paper", "running question")
    response = client.post("/api/papers", data={"paper_id": "paper"}, files={"file": ("new.pdf", pdf_bytes())})
    assert response.status_code == 409
    assert server.store.load("paper")[0].text == "original content"


@pytest.mark.parametrize("error", [json.JSONDecodeError("bad", "", 0), OSError("unreadable"), FileNotFoundError()])
def test_index_errors_fail_job_and_release_slot(isolated, monkeypatch, error):
    def fail_load(*args):
        raise error
    monkeypatch.setattr(server.store, "load", fail_load)
    job = server._create_job("paper", "question", "user")
    server._execute_job(job["job_id"], "paper", "question", {})
    assert server._read_job(job["job_id"])["status"] == "failed"
    assert server._active_user_jobs() == 0


def test_missing_config_also_fails_job(isolated):
    job = server._create_job("paper", "question")
    server._execute_job(job["job_id"], "paper", "question", {})
    assert server._read_job(job["job_id"])["status"] == "failed"


def test_duplicate_submission_reuses_job_without_charging_again(isolated):
    first_tasks, second_tasks = BackgroundTasks(), BackgroundTasks()
    auth = server.AuthContext("user")
    first = server.ask(first_tasks, "paper", "same question", auth)
    second = server.ask(second_tasks, "paper", "same  question", auth)
    assert first["job_id"] == second["job_id"]
    assert server._read_usage()["jobs_used"] == 1
    assert len(first_tasks.tasks) == 1
    assert len(second_tasks.tasks) == 0
    with pytest.raises(HTTPException) as exc:
        server.ask(BackgroundTasks(), "paper", "different question", server.AuthContext("admin"))
    assert exc.value.status_code == 409


def test_import_lock_prevents_new_question(isolated):
    with server._SUBMISSION_LOCK:
        with pytest.raises(HTTPException) as exc:
            server.ask(BackgroundTasks(), "paper", "question", server.AuthContext("admin"))
    assert exc.value.status_code == 409
    assert server._active_jobs() == []


def test_failed_job_creation_refunds_reservation(isolated, monkeypatch):
    def fail(*args):
        raise OSError("failed write")
    monkeypatch.setattr(server, "_create_job", fail)
    with pytest.raises(OSError):
        server.ask(BackgroundTasks(), "paper", "question", server.AuthContext("user"))
    assert server._read_usage()["jobs_used"] == 0


def test_restart_marks_jobs_interrupted_without_model_replay(isolated, monkeypatch):
    queued = server._create_job("paper", "one", "user")
    running = server._create_job("paper", "two", "admin")
    done = server._create_job("paper", "three", "admin")
    server._update_job(running["job_id"], status="running")
    server._update_job(done["job_id"], status="completed")
    def forbidden(*args):
        pytest.fail("Recovery must not call models")
    monkeypatch.setattr(server, "build_pipeline", forbidden)
    server._recover_interrupted_jobs()
    server._recover_interrupted_jobs()  # Idempotent on another restart.
    for j in (queued, running):
        state = server._read_job(j["job_id"])
        assert state["status"] == "failed" and state["interrupted"]
    assert server._read_job(done["job_id"])["status"] == "completed"
    assert not server._active_jobs()


@pytest.mark.parametrize("mode", ["timeout", "crash", "start_error", "clean_without_result"])
def test_supervisor_stops_worker_and_releases_slot(isolated, monkeypatch, mode):
    class Worker:
        pid = None
        exitcode = 1 if mode == "crash" else 0
        alive = False
        closed = False
        terminated = False
        def start(self):
            if mode == "start_error":
                raise OSError("cannot start")
            self.pid = 123
            self.alive = mode == "timeout"
        def join(self, *args): pass
        def is_alive(self): return self.alive
        def terminate(self): self.terminated = True; self.alive = False
        def close(self): self.closed = True
    worker = Worker()
    monkeypatch.setattr(server.multiprocessing, "get_context", lambda _: SimpleNamespace(Process=lambda **kw: worker))
    job = server._create_job("paper", "question", "user")
    server._run_job(job["job_id"], "paper", "question", {})
    state = server._read_job(job["job_id"])
    assert state["status"] == "failed"
    assert not server._active_jobs()
    if mode == "timeout":
        assert worker.terminated and worker.closed and "300" in state["error"]


@pytest.mark.parametrize("reason,content", [("length", "{}"), ("content_filter", ""), ("stop", "  ")])
def test_transport_rejects_empty_or_truncated_output(reason, content):
    llm = object.__new__(OpenAICompatibleLLM)
    llm.model = "test"
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason=reason, message=SimpleNamespace(content=content))])
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    with pytest.raises(InvalidModelOutput):
        llm.complete("s", "u")
