from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from citetrace.app import build_pipeline
from citetrace.paper.parser import parse_pdf
from citetrace.paper.store import JsonPaperStore


load_dotenv()
app = FastAPI(title="CiteTrace", version="0.1.0")
store = JsonPaperStore()
pipeline = None

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = build_pipeline(store)
    return pipeline


@app.get("/")
def index():
    path = Path("web/index.html")
    if not path.exists():
        return {"name": "CiteTrace", "status": "ok"}
    return FileResponse(path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/papers")
async def upload_paper(paper_id: str = Form(...), file: UploadFile = File(...)):
    if not _SAFE_ID.match(paper_id):
        raise HTTPException(status_code=400, detail="paper_id must use letters, numbers, ., _, or -")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{paper_id}.pdf"
    path.write_bytes(await file.read())

    chunks = parse_pdf(path, paper_id)
    store.save(paper_id, chunks)
    return {"paper_id": paper_id, "chunks": len(chunks)}


@app.post("/api/ask")
def ask(paper_id: str = Form(...), question: str = Form(...)):
    try:
        result = get_pipeline().run(question, paper_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown paper_id") from None
    return asdict(result)
