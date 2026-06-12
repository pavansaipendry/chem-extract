"""Live web UI for the chemistry extraction pipeline.

Drag a document in; the server streams each stage back as NDJSON so the browser
can show what the system is doing in real time — reading passes, validation
flagging, and unit inference with reasoning.

Run:  .venv/bin/uvicorn webapp.server:app --reload --port 8000
"""
import json
import tempfile
import traceback
from pathlib import Path

import fitz  # PyMuPDF
from anthropic import Anthropic
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from chemextract import rag
from chemextract.extract import extract_page
from chemextract.infer import infer_unit
from chemextract.pipeline import CONFIDENCE_THRESHOLD, _consistency_flags, _obs_key
from chemextract.reconstruct import reconstruct
from chemextract.schema import DocumentExtraction, Observation
from chemextract.validate import validate_observation

app = FastAPI()
STATIC = Path(__file__).parent / "static"
client = Anthropic()


def _event(**kw) -> str:
    return json.dumps(kw) + "\n"


def to_image(path: str, page: int = 0) -> tuple[str, int]:
    """If the upload is a PDF, render one page to a PNG the vision API can read.
    Returns (image_path, page_count). Non-PDFs pass through unchanged."""
    if not path.lower().endswith(".pdf"):
        return path, 1
    doc = fitz.open(path)
    page_count = doc.page_count
    pix = doc[min(page, page_count - 1)].get_pixmap(dpi=160)
    out = path + f".p{page}.png"
    pix.save(out)
    return out, page_count


def stream_extract(upload_path: str, n_runs: int = 2, page: int = 0, source_name: str = "upload"):
    """Yield NDJSON stage events as the pipeline runs."""
    try:
        yield from _stream_extract_inner(upload_path, n_runs, page, source_name)
    except Exception as e:  # never leave the UI hanging on "Reading…"
        traceback.print_exc()
        yield _event(stage="error", message=f"{type(e).__name__}: {e}")


def _stream_extract_inner(upload_path: str, n_runs: int, page: int, source_name: str = "upload"):
    image_path, page_count = to_image(upload_path, page)
    yield _event(stage="start", runs=n_runs, pages=page_count,
                 rendered=("pdf" if upload_path.lower().endswith(".pdf") else "image"))

    # --- reading passes (one call each so we can report per-run) ---
    runs = []
    for i in range(n_runs):
        yield _event(stage="reading", run=i + 1, runs=n_runs)
        raw = extract_page(image_path, n_runs=1, client=client)[0]
        runs.append(raw)
        yield _event(stage="read_done", run=i + 1, runs=n_runs, transcript=raw.transcript)

    primary, others = runs[0], runs[1:]
    mismatched = _consistency_flags(primary, others) if others else set()

    # --- validation ---
    yield _event(stage="validating", n=len(primary.observations))
    observations: list[Observation] = []
    for raw in primary.observations:
        flags = validate_observation(raw)
        if _obs_key(raw) in mismatched:
            flags.append("self_consistency_mismatch")
        if raw.confidence < CONFIDENCE_THRESHOLD:
            flags.append("low_confidence")
        observations.append(Observation(
            quantity_type=raw.quantity_type, value=raw.value, unit=raw.unit,
            quote=raw.quote, confidence=raw.confidence, flags=flags,
            needs_review=bool(flags),
        ))

    yield _event(
        stage="validated",
        observations=[json.loads(o.model_dump_json()) for o in observations],
    )

    # --- inference for missing-unit fields ---
    for idx, obs in enumerate(observations):
        if "missing_unit" in obs.flags:
            yield _event(stage="inferring", index=idx, value=obs.value, quote=obs.quote)
            obs.inferred = infer_unit(obs, primary.transcript, client=client)
            yield _event(stage="inferred", index=idx,
                         inferred=json.loads(obs.inferred.model_dump_json()))

    # --- chemistry-grounded reconstruction (corrected, fully-specified rewrite) ---
    yield _event(stage="reconstructing")
    doc = DocumentExtraction(source_image="upload", transcript=primary.transcript,
                             observations=observations)
    recon = reconstruct(doc, client=client)
    yield _event(stage="reconstructed", reconstruction=json.loads(recon.model_dump_json()))

    # --- index into the RAG store so the document becomes queryable ---
    yield _event(stage="indexing")
    try:
        n_chunks = rag.ingest_document(
            doc_id=source_name,
            source=source_name,
            transcript=primary.transcript,
            canonical_text=recon.canonical_text,
            observations=[json.loads(o.model_dump_json()) for o in observations],
        )
        yield _event(stage="indexed", chunks=n_chunks, **rag.status())
    except Exception as e:  # RAG indexing must never sink the extraction itself
        traceback.print_exc()
        yield _event(stage="indexed", chunks=0, error=f"{type(e).__name__}: {e}")

    # --- summary ---
    total = len(observations)
    review = sum(1 for o in observations if o.needs_review)
    unresolved = sum(1 for a in recon.ambiguities if not a.resolved)
    yield _event(stage="done", transcript=primary.transcript,
                 total=total, review=review, unresolved=unresolved,
                 observations=[json.loads(o.model_dump_json()) for o in observations])


@app.post("/extract")
async def extract(file: UploadFile, runs: int = 2, page: int = 0):
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(await file.read())
    tmp.close()
    return StreamingResponse(
        stream_extract(tmp.name, n_runs=runs, page=page,
                       source_name=file.filename or "upload"),
        media_type="application/x-ndjson")


# ---------------- RAG: ask questions answered ONLY from indexed documents ----


class AskRequest(BaseModel):
    question: str
    k: int = 5


class FinalizeRequest(BaseModel):
    source: str
    final_text: str


@app.post("/ask")
async def ask(req: AskRequest):
    try:
        return rag.answer(req.question.strip(), k=req.k, client=client)
    except Exception as e:
        traceback.print_exc()
        return {"answer": f"Something went wrong: {type(e).__name__}: {e}",
                "sources": [], "refused": False, "error": True}


@app.post("/rag/finalize")
async def rag_finalize(req: FinalizeRequest):
    """Index the human-reviewed final text — the highest-trust version."""
    n = rag.ingest_document(doc_id=req.source + "::final", source=req.source,
                            final_text=req.final_text)
    return {"chunks": n, **rag.status()}


@app.get("/rag/status")
async def rag_status():
    try:
        return rag.status()
    except Exception:
        return {"chunks": 0, "documents": []}


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
