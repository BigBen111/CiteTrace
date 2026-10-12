from __future__ import annotations

import json
import logging
import multiprocessing
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Annotated
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from openai import APIError, AuthenticationError, PermissionDeniedError, RateLimitError
from pydantic import BaseModel, Field, field_validator
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from citetrace.app import build_pipeline
from citetrace.paper.parser import build_pdf_metadata, parse_pdf
from citetrace.paper.store import JsonPaperStore
from citetrace.llm.validation import InvalidModelOutput
from citetrace.utils.files import atomic_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(os.environ.get("CITETRACE_CONFIG_DIR", PROJECT_ROOT))
ENV_PATH = CONFIG_DIR / ".env"
RUNS_PATH = PROJECT_ROOT / "artifacts" / "runs"
JOBS_PATH = PROJECT_ROOT / "artifacts" / "jobs"
USAGE_PATH = PROJECT_ROOT / "artifacts" / "usage.json"

load_dotenv(ENV_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _recover_interrupted_jobs()
    yield


app = FastAPI(title="CiteTrace", version="0.1.0", lifespan=lifespan)
store = JsonPaperStore(PROJECT_ROOT / "data" / "index")
logger = logging.getLogger(__name__)

PROVIDERS = ("QWEN", "DEEPSEEK", "GLM")
DEFAULTS = {
    "QWEN_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "QWEN_MODEL": "qwen3.7-max",
    "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
    "DEEPSEEK_MODEL": "deepseek-v4-pro",
    "GLM_BASE_URL": "https://open.bigmodel.cn/api/paas/v4/",
    "GLM_MODEL": "glm-5.3",
}
ACCESS_DEFAULTS = {
    "CITETRACE_USER_DAILY_JOB_LIMIT": "20",
    "CITETRACE_USER_MAX_ACTIVE_JOBS": "1",
}
RUNTIME_DEFAULTS = {
    "CITETRACE_PIPELINE_MAX_WORKERS": "4",
    "CITETRACE_LLM_TIMEOUT_SECONDS": "60",
    "CITETRACE_LLM_MAX_RETRIES": "1",
    "CITETRACE_JOB_TIMEOUT_SECONDS": "300",
}

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_MAX_PDF_BYTES = 50 * 1024 * 1024
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_USAGE_LOCK = Lock()
_JOB_LOCK = Lock()
# Single-user admission: imports and new questions cannot race with each other.
_SUBMISSION_LOCK = Lock()


@dataclass(frozen=True)
class AuthContext:
    role: str


class ProviderSettings(BaseModel):
    api_key: str = Field(default="", max_length=500)
    base_url: str = Field(min_length=1, max_length=500)
    model: str = Field(min_length=1, max_length=200)

    @field_validator("api_key", "base_url", "model")
    @classmethod
    def reject_line_breaks(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("Line breaks are not allowed")
        return value


class AccessSettings(BaseModel):
    user_token: str = Field(default="", max_length=500)
    daily_job_limit: int = Field(default=20, ge=1, le=1000)
    max_active_jobs: int = Field(default=1, ge=1, le=20)

    @field_validator("user_token")
    @classmethod
    def reject_token_line_breaks(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("Line breaks are not allowed")
        return value


class SettingsPayload(BaseModel):
    qwen: ProviderSettings
    deepseek: ProviderSettings
    glm: ProviderSettings
    access: AccessSettings


def _admin_token() -> str:
    token = os.environ.get("CITETRACE_ADMIN_TOKEN", "")
    if not token:
        token = secrets.token_urlsafe(32)
        _write_env({"CITETRACE_ADMIN_TOKEN": token})
    return token


def _user_token() -> str:
    token = os.environ.get("CITETRACE_USER_TOKEN", "")
    if not token:
        token = secrets.token_urlsafe(24)
        _write_env({"CITETRACE_USER_TOKEN": token})
    return token


def _require_access(
    x_admin_token: Annotated[str | None, Header()] = None,
) -> AuthContext:
    if x_admin_token and secrets.compare_digest(x_admin_token, _admin_token()):
        return AuthContext(role="admin")
    if x_admin_token and secrets.compare_digest(x_admin_token, _user_token()):
        return AuthContext(role="user")
    raise HTTPException(status_code=401, detail="Invalid access token")


def _require_admin(
    auth: AuthContext = Depends(_require_access),
) -> AuthContext:
    if auth.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    return auth


def _env_values() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "CITETRACE_ADMIN_TOKEN",
            "CITETRACE_USER_TOKEN",
            *ACCESS_DEFAULTS,
            *RUNTIME_DEFAULTS,
        }
        or key.startswith(("QWEN_", "DEEPSEEK_", "GLM_"))
    }


def _write_env(updates: dict[str, str]) -> None:
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    current = _env_values()
    current.update(updates)
    lines = [f"{key}={_dotenv_quote(current[key])}" for key in sorted(current)]
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=ENV_PATH.parent, delete=False
    ) as temp:
        temp.write("\n".join(lines) + "\n")
        temp_path = Path(temp.name)
    temp_path.chmod(0o600)
    temp_path.replace(ENV_PATH)
    for key, value in current.items():
        os.environ[key] = value


def _dotenv_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _pipeline_config() -> dict[str, str]:
    config = _env_values()
    for key, value in DEFAULTS.items():
        config.setdefault(key, value)
    for key, value in RUNTIME_DEFAULTS.items():
        config.setdefault(key, value)
    return config


def _access_config() -> dict[str, str]:
    config = _env_values()
    for key, value in ACCESS_DEFAULTS.items():
        config.setdefault(key, value)
    return config


@app.get("/")
def index():
    path = PROJECT_ROOT / "web" / "index.html"
    if not path.exists():
        return {"name": "CiteTrace", "status": "ok"}
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/session")
def session(auth: AuthContext = Depends(_require_access)):
    return _quota_snapshot(auth.role)


@app.get("/api/settings")
def get_settings(_: AuthContext = Depends(_require_admin)):
    config = _pipeline_config()
    response = {
        provider.lower(): {
            "configured": bool(config.get(f"{provider}_API_KEY")),
            "base_url": config[f"{provider}_BASE_URL"],
            "model": config[f"{provider}_MODEL"],
        }
        for provider in PROVIDERS
    }
    access = _access_config()
    response["access"] = {
        "configured": bool(access.get("CITETRACE_USER_TOKEN")),
        "daily_job_limit": int(access["CITETRACE_USER_DAILY_JOB_LIMIT"]),
        "max_active_jobs": int(access["CITETRACE_USER_MAX_ACTIVE_JOBS"]),
    }
    return response


@app.put("/api/settings")
def save_settings(payload: SettingsPayload, _: AuthContext = Depends(_require_admin)):
    existing = _pipeline_config()
    updates: dict[str, str] = {}
    for provider in PROVIDERS:
        setting = getattr(payload, provider.lower())
        api_key = setting.api_key.strip() or existing.get(f"{provider}_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=400, detail=f"{provider} API key is required"
            )
        updates.update(
            {
                f"{provider}_API_KEY": api_key,
                f"{provider}_BASE_URL": setting.base_url.strip(),
                f"{provider}_MODEL": setting.model.strip(),
            }
        )
    user_token = payload.access.user_token.strip() or existing.get(
        "CITETRACE_USER_TOKEN", ""
    )
    if not user_token:
        user_token = secrets.token_urlsafe(24)
    updates.update(
        {
            "CITETRACE_USER_TOKEN": user_token,
            "CITETRACE_USER_DAILY_JOB_LIMIT": str(payload.access.daily_job_limit),
            "CITETRACE_USER_MAX_ACTIVE_JOBS": str(payload.access.max_active_jobs),
        }
    )
    _write_env(updates)
    return {"status": "saved"}


@app.get("/api/runs/latest")
def latest_run(_: AuthContext = Depends(_require_admin)):
    paths = sorted(RUNS_PATH.glob("*.json"), reverse=True)
    if not paths:
        raise HTTPException(status_code=404, detail="No completed runs")
    return json.loads(paths[0].read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/trace")
def run_trace(run_id: str, _: AuthContext = Depends(_require_admin)):
    if not _SAFE_ID.match(run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id")
    try:
        payload = json.loads((RUNS_PATH / f"{run_id}.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown run_id") from None
    return {"run_id": run_id, "trace": payload.get("trace", [])}


@app.post("/api/papers")
async def upload_paper(
    paper_id: str = Form(...),
    file: UploadFile = File(...),
    _: AuthContext = Depends(_require_access),
):
    request_started = time.monotonic()
    if not _SAFE_ID.match(paper_id):
        raise HTTPException(status_code=400, detail="paper_id must use letters, numbers, ., _, or -")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if not _SUBMISSION_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="正在导入论文或提交任务，请稍后重试")
    path: Path | None = None
    committed = False
    try:
        if _active_jobs():
            raise HTTPException(status_code=409, detail="请等待当前提问完成后再导入论文")
        upload_dir = PROJECT_ROOT / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        # Unique, immutable PDF; the index is the only publication/commit point.
        with NamedTemporaryFile("wb", dir=upload_dir, prefix=f"{paper_id}-", suffix=".pdf", delete=False) as output:
            path = Path(output.name)
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > _MAX_PDF_BYTES:
                    raise HTTPException(status_code=413, detail="PDF must be 50 MB or smaller")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        upload_finished = time.monotonic()
        try:
            chunks = await run_in_threadpool(parse_pdf, path, paper_id)
            if not chunks:
                raise HTTPException(status_code=422, detail="No extractable text found in PDF")
            parse_finished = time.monotonic()
            metadata = await run_in_threadpool(build_pdf_metadata, path, chunks)
        except HTTPException:
            raise
        except Exception:
            logger.exception("PDF parsing failed for %s", paper_id)
            raise HTTPException(status_code=422, detail="PDF 解析失败，请检查文件后重试；已有论文未被修改") from None
        # Do not await across the commit boundary: cancellation after publication
        # must never delete the PDF now referenced by the index.
        store.save(paper_id, chunks, metadata, pdf_filename=path.name)
        committed = True
        index_finished = time.monotonic()
    except OSError:
        logger.exception("Paper import storage failure")
        raise HTTPException(status_code=503, detail="论文保存失败，请稍后重试；已有论文未被修改") from None
    finally:
        try:
            if path is not None and not committed:
                path.unlink(missing_ok=True)
        finally:
            _SUBMISSION_LOCK.release()
    timings = {
        "upload_seconds": round(upload_finished - request_started, 3),
        "parse_seconds": round(parse_finished - upload_finished, 3),
        "index_seconds": round(index_finished - parse_finished, 3),
        "total_seconds": round(index_finished - request_started, 3),
    }
    logger.info(
        "Paper %s imported: bytes=%d chunks=%d timings=%s",
        paper_id,
        total,
        len(chunks),
        timings,
    )
    return {"paper_id": paper_id, "chunks": len(chunks), "timings": timings}


@app.get("/api/papers/{paper_id}")
def paper_status(paper_id: str, _: AuthContext = Depends(_require_access)):
    if not _SAFE_ID.match(paper_id):
        raise HTTPException(status_code=400, detail="Invalid paper_id")
    try:
        chunks = store.load(paper_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown paper_id") from None
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(status_code=422, detail="论文索引无法读取，请重新导入 PDF") from None
    return {"paper_id": paper_id, "chunks": len(chunks), "status": "ready"}


@app.post("/api/ask", status_code=202)
def ask(
    background_tasks: BackgroundTasks,
    paper_id: str = Form(...),
    question: str = Form(...),
    auth: AuthContext = Depends(_require_access),
):
    if not _SAFE_ID.match(paper_id):
        raise HTTPException(status_code=400, detail="Invalid paper_id")
    normalized_question = " ".join(question.split()).strip()
    if not normalized_question:
        raise HTTPException(status_code=400, detail="Question is required")
    if not _SUBMISSION_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="正在导入论文或提交任务，请稍后重试")
    try:
        active = _active_jobs()
        duplicate = next((j for j in active if j.get("paper_id") == paper_id
                          and j.get("question") == normalized_question
                          and j.get("owner_role") == auth.role), None)
        if duplicate is not None:
            job = duplicate
        else:
            if active:
                raise HTTPException(status_code=409, detail="已有提问正在处理，请等待完成或恢复上次提问")
            try:
                store.load(paper_id)
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Unknown paper_id") from None
            except (OSError, ValueError, KeyError, TypeError):
                raise HTTPException(status_code=422, detail="论文索引无法读取，请重新导入 PDF") from None
            config = _pipeline_config()
            _job_timeout(config)
            if auth.role == "user":
                with _USAGE_LOCK:
                    previous_usage = _read_usage()
                    _reserve_user_job()
                    try:
                        job = _create_job(paper_id, normalized_question, auth.role)
                    except Exception:
                        _write_usage(previous_usage)
                        raise
            else:
                job = _create_job(paper_id, normalized_question, auth.role)
            background_tasks.add_task(_run_job, job["job_id"], paper_id, normalized_question, config)
    finally:
        _SUBMISSION_LOCK.release()
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job["stage"],
        "message": job["message"],
        "quota": _quota_snapshot(auth.role),
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, auth: AuthContext = Depends(_require_access)):
    if not _SAFE_ID.match(job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id")
    try:
        job = _read_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown job_id") from None
    if auth.role == "user" and job.get("owner_role") != "user":
        raise HTTPException(status_code=403, detail="This job is not available to users")
    return job


def _run_job(
    job_id: str,
    paper_id: str,
    question: str,
    config: dict[str, str],
) -> None:
    """Supervise one killable worker; a timeout cannot leave model threads alive."""
    started = time.monotonic()
    worker = None
    error: str | None = None
    try:
        timeout = _job_timeout(config)
        worker = multiprocessing.get_context("spawn").Process(
            target=_job_worker,
            args=(job_id, paper_id, question, config, str(JOBS_PATH), str(RUNS_PATH), str(store.root)),
            daemon=True,
            name=f"citetrace-job-{job_id[-8:]}",
        )
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            error = f"任务超过 {timeout:g} 秒，已停止后台执行。请缩小问题范围后重试。"
        elif worker.exitcode != 0:
            error = "后台执行异常退出，请重新运行。"
    except Exception as exc:
        logger.exception("Unable to supervise job %s", job_id)
        error = _job_error_message(exc)
    finally:
        if worker is not None and worker.pid is not None:
            if worker.is_alive():
                worker.terminate()
                worker.join(3)
                if worker.is_alive():
                    worker.kill()
                    worker.join()
            worker.close()
    # Only reconcile after the child is gone; it cannot overwrite a terminal state.
    try:
        job = _read_job(job_id)
        if job.get("status") in {"queued", "running"}:
            message = error or "后台任务未能正常完成，请重新运行。"
            _update_job(job_id, status="failed", message=message, error=message,
                        elapsed_seconds=round(time.monotonic() - started, 3))
    except Exception:
        logger.exception("Unable to persist final state for job %s", job_id)


def _job_timeout(config: dict[str, str]) -> float:
    timeout = float(config.get("CITETRACE_JOB_TIMEOUT_SECONDS", "300"))
    if not 10 <= timeout <= 1800:
        raise ValueError("CITETRACE_JOB_TIMEOUT_SECONDS must be between 10 and 1800")
    return timeout


def _job_worker(job_id: str, paper_id: str, question: str, config: dict[str, str],
                jobs_path: str, runs_path: str, index_path: str) -> None:
    global JOBS_PATH, RUNS_PATH, store
    JOBS_PATH, RUNS_PATH = Path(jobs_path), Path(runs_path)
    store = JsonPaperStore(index_path)
    _execute_job(job_id, paper_id, question, config)


def _execute_job(
    job_id: str,
    paper_id: str,
    question: str,
    config: dict[str, str],
) -> None:
    started = time.monotonic()
    progress_lock = Lock()
    stage_history: list[dict] = []
    active_stage_started: float | None = None

    def stage_snapshot(*, finalize: bool = False) -> list[dict]:
        nonlocal active_stage_started
        with progress_lock:
            if stage_history and active_stage_started is not None:
                stage_history[-1]["duration_seconds"] = round(
                    time.monotonic() - active_stage_started, 3
                )
                if finalize:
                    active_stage_started = None
            return [dict(item) for item in stage_history]

    def progress(stage: str, message: str) -> None:
        nonlocal active_stage_started
        now = time.monotonic()
        with progress_lock:
            if stage_history and stage_history[-1]["stage"] == stage:
                stage_history[-1]["message"] = message
                stage_history[-1]["duration_seconds"] = round(
                    now - (active_stage_started or now), 3
                )
            else:
                if stage_history and active_stage_started is not None:
                    stage_history[-1]["duration_seconds"] = round(
                        now - active_stage_started, 3
                    )
                active_stage_started = now
                stage_history.append(
                    {
                        "stage": stage,
                        "message": message,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "duration_seconds": 0.0,
                    }
                )
            snapshot = [dict(item) for item in stage_history]
        _update_job(
            job_id,
            status="running",
            stage=stage,
            message=message,
            stage_history=snapshot,
        )

    try:
        _update_job(
            job_id, status="running", stage="question_routing",
            message="后台任务已启动", stage_history=[],
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        store.load(paper_id)
        models = {provider.lower(): config[f"{provider}_MODEL"] for provider in PROVIDERS}
        pipeline = build_pipeline(store, config)
        result = pipeline.run(question, paper_id, progress=progress)
        response = asdict(result)
        trace_reader = getattr(pipeline, "trace_snapshot", None)
        trace = trace_reader() if callable(trace_reader) else []
        final_stage_history = stage_snapshot(finalize=True)
        response["stage_history"] = final_stage_history
        response["models_used"] = [
            {
                "agent": entry.get("agent"),
                "provider": entry.get("provider"),
                "model": entry.get("model"),
            }
            for entry in trace
            if entry.get("kind") == "llm_call"
        ]
        run_id = _save_run(paper_id, question, models, response, trace=trace)
        response["run_id"] = run_id
        response["models"] = models
        _update_job(
            job_id,
            status="completed",
            stage="completed",
            message="推理完成",
            run_id=run_id,
            result=response,
            stage_history=final_stage_history,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
    except Exception as exc:
        logger.exception("CiteTrace background job %s failed", job_id)
        _update_job(
            job_id,
            status="failed",
            message=_job_error_message(exc),
            error=_job_error_message(exc),
            stage_history=stage_snapshot(finalize=True),
            elapsed_seconds=round(time.monotonic() - started, 3),
        )


def _job_error_message(exc: Exception) -> str:
    if isinstance(exc, InvalidModelOutput):
        return "模型返回空内容或无效格式，备用模型也未能完成，请重新运行。"
    if isinstance(exc, json.JSONDecodeError):
        return "论文索引或任务记录无法读取，请重新导入 PDF 后重试。"
    if isinstance(exc, FileNotFoundError):
        return "论文索引已不存在，请重新导入 PDF 后重试。"
    if isinstance(exc, AuthenticationError):
        return "Model API authentication failed"
    if isinstance(exc, PermissionDeniedError):
        return f"Model unavailable for this account or quota: {str(exc)[:240]}"
    if isinstance(exc, RateLimitError):
        return "Model API rate limit reached"
    if isinstance(exc, APIError):
        return f"Model API request failed: {str(exc)[:240]}"
    if isinstance(exc, ValueError):
        return str(exc)[:240]
    return f"Unexpected background error: {str(exc)[:240]}"


def _quota_snapshot(role: str) -> dict:
    if role == "admin":
        return {"role": "admin", "unlimited": True}
    with _USAGE_LOCK:
        usage = _read_usage()
        config = _access_config()
        limit = int(config["CITETRACE_USER_DAILY_JOB_LIMIT"])
        active_limit = int(config["CITETRACE_USER_MAX_ACTIVE_JOBS"])
        active = _active_user_jobs()
        used = int(usage.get("jobs_used", 0))
        return {
            "role": "user",
            "unlimited": False,
            "daily_job_limit": limit,
            "jobs_used": used,
            "jobs_remaining": max(0, limit - used),
            "max_active_jobs": active_limit,
            "active_jobs": active,
            "reset_at": _quota_reset_at(),
        }


def _reserve_user_job() -> None:
    usage = _read_usage()
    config = _access_config()
    limit = int(config["CITETRACE_USER_DAILY_JOB_LIMIT"])
    active_limit = int(config["CITETRACE_USER_MAX_ACTIVE_JOBS"])
    used = int(usage.get("jobs_used", 0))
    active = _active_user_jobs()
    if active >= active_limit:
        raise HTTPException(
            status_code=429,
            detail=f"普通用户同时最多运行 {active_limit} 个任务，请等待当前任务完成",
        )
    if used >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"普通用户今日 {limit} 次提问额度已用完，额度将在 {_quota_reset_at()} 重置",
        )
    usage["jobs_used"] = used + 1
    _write_usage(usage)


def _active_user_jobs() -> int:
    return sum(job.get("owner_role") == "user" for job in _active_jobs())


def _active_jobs() -> list[dict]:
    active = []
    for path in JOBS_PATH.glob("*.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if job.get("status") in {
            "queued",
            "running",
        }:
            active.append(job)
    return active


def _read_usage() -> dict:
    today = datetime.now(_SHANGHAI).date().isoformat()
    try:
        payload = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        payload = {}
    if payload.get("date") != today:
        payload = {"date": today, "jobs_used": 0}
    return payload


def _write_usage(payload: dict) -> None:
    USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=USAGE_PATH.parent, delete=False
    ) as temp:
        json.dump(payload, temp, ensure_ascii=False, indent=2)
        temp.write("\n")
        temp_path = Path(temp.name)
    temp_path.chmod(0o600)
    temp_path.replace(USAGE_PATH)


def _quota_reset_at() -> str:
    now = datetime.now(_SHANGHAI)
    return (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).isoformat()


def _create_job(paper_id: str, question: str, owner_role: str = "admin") -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_id = f"{stamp}-{secrets.token_hex(4)}"
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "job_id": job_id,
        "status": "queued",
        "stage": "question_routing",
        "message": "任务已创建，等待后台执行",
        "stage_history": [],
        "created_at": now,
        "updated_at": now,
        "paper_id": paper_id,
        "question": question,
        "owner_role": owner_role,
    }
    with _JOB_LOCK:
        _write_job(payload)
    return payload


def _read_job(job_id: str) -> dict:
    return json.loads((JOBS_PATH / f"{job_id}.json").read_text(encoding="utf-8"))


def _update_job(job_id: str, **updates) -> dict:
    with _JOB_LOCK:
        payload = _read_job(job_id)
        payload.update(updates)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_job(payload)
    return payload


def _recover_interrupted_jobs() -> None:
    # Never replay paid API requests automatically after an uncertain shutdown.
    for job in _active_jobs():
        message = "服务重启导致任务中断，未自动重复调用模型。请重新运行。"
        _update_job(
            job["job_id"], status="failed", message=message, error=message,
            interrupted=True,
        )


def _write_job(payload: dict) -> None:
    atomic_json(JOBS_PATH / f"{payload['job_id']}.json", payload)


def _save_run(
    paper_id: str,
    question: str,
    models: dict[str, str],
    result: dict,
    trace: list[dict] | None = None,
) -> str:
    RUNS_PATH.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{secrets.token_hex(4)}"
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_id": paper_id,
        "question": question,
        "models": models,
        "result": result,
        "trace": trace or [],
    }
    path = RUNS_PATH / f"{run_id}.json"
    atomic_json(path, payload)
    return run_id
