"""Exercise real spawned workers with a local fake API, never paid model APIs."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from citetrace import server
from citetrace.paper.parser import PaperChunk
from citetrace.paper.store import JsonPaperStore


@pytest.mark.parametrize("mode", ["fallback", "timeout"])
def test_real_worker_lifecycle(tmp_path, monkeypatch, mode):
    calls = []
    request_started = threading.Event()
    release_request = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(data["model"])
            if mode == "timeout":
                request_started.set()
                release_request.wait(20)
            system = data["messages"][0]["content"]
            if data["model"] == "qwen":
                raw = ""  # HTTP 200 with empty output must use the backup.
            elif "plan lexical" in system:
                raw = json.dumps({"queries": ["NVIDIA P100 GPU"]})
            elif "claim-evidence verifier" in system:
                raw = json.dumps({"verdict": "supported", "confidence": 0.99, "reason": "E1 specifies NVIDIA P100."})
            else:
                raw = json.dumps({"answer": "NVIDIA P100 GPUs [E1].", "sufficient": True,
                                  "claims": [{"text": "NVIDIA P100 GPUs were used.", "evidence_ids": ["E1"]}]})
            body = json.dumps({"id": "local-test", "object": "chat.completion", "created": 1,
                               "model": data["model"], "choices": [{"index": 0, "finish_reason": "stop",
                               "message": {"role": "assistant", "content": raw}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(server, "JOBS_PATH", tmp_path / "jobs")
        monkeypatch.setattr(server, "RUNS_PATH", tmp_path / "runs")
        store = JsonPaperStore(tmp_path / "index")
        store.save("paper", [PaperChunk("p1", "paper", 1, "We trained our models with 8 NVIDIA P100 GPUs.")])
        monkeypatch.setattr(server, "store", store)
        config = {"CITETRACE_JOB_TIMEOUT_SECONDS": "10" if mode == "timeout" else "30",
                  "CITETRACE_LLM_MAX_RETRIES": "0", "CITETRACE_LLM_TIMEOUT_SECONDS": "30"}
        for provider in server.PROVIDERS:
            config.update({f"{provider}_API_KEY": "local-test-not-a-secret",
                           f"{provider}_BASE_URL": f"http://127.0.0.1:{http.server_port}/v1",
                           f"{provider}_MODEL": provider.lower()})
        job = server._create_job("paper", "What GPUs were used?", "user")
        started = time.monotonic()
        server._run_job(job["job_id"], "paper", "What GPUs were used?", config)
        state = server._read_job(job["job_id"])
        assert not server._active_jobs()
        if mode == "fallback":
            assert state["status"] == "completed", state
            assert "P100" in state["result"]["answer"]
            assert "deepseek" in calls and "glm" in calls
            assert any(s["stage"] == "answer_fallback" for s in state["stage_history"])
            trace = json.loads((tmp_path / "runs" / f"{state['run_id']}.json").read_text())["trace"]
            assert any(e.get("status") == "error" and e.get("provider") == "qwen" for e in trace)
        else:
            assert request_started.is_set()
            assert state["status"] == "failed"
            assert "10 秒" in state["error"]
            assert time.monotonic() - started < 16
            count = len(calls)
            release_request.set()
            time.sleep(0.2)
            assert len(calls) == count  # No retry from an abandoned background thread.
            assert server._read_job(job["job_id"])["status"] == "failed"
    finally:
        release_request.set()
        http.shutdown()
        http.server_close()
        thread.join(2)
