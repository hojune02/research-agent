import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File, Form

from app.config import settings

from app.schemas import (
    LLMRequest,
    LLMResponse,
    UploadResponse,
    ParsePDFRequest,
    ParsePDFResponse,
    ParsePDFPreviewResponse,
)
from app.llm.client import generate_answer
from app.tools.file_utils import (
    build_upload_path,
    validate_pdf_upload,
    get_uploaded_pdf_path,
)

from app.rag.pdf_parser import parse_pdf_into_chunks

app = FastAPI(
    title="Soundable Research Agent",
    description="Local multi-user research automation agent with RAG, tool calling, memory, and local LLM serving.",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "Sounable Research Agent backend is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mock_llm": settings.MOCK_LLM,
        "llm_base_url": settings.LLM_BASE_URL,
        "llm_model": settings.LLM_MODEL,
        "upload_dir": settings.UPLOAD_DIR,
        "chroma_path": settings.CHROMA_PATH,
        "sqlite_path": settings.SQLITE_PATH,
        "top_k": settings.TOP_K,
    }

@app.post("/debug/llm", response_model=LLMResponse)
def debug_llm(request: LLMRequest):
    try:
        return generate_answer(
            prompt=request.prompt,
            context=request.context,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    
@app.post("/upload/raw", response_model=UploadResponse)
def upload_raw_pdf(
    user_id: str = Form(...),
    project_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Phase 3 endpoint.

    Saves an uploaded PDF under:
        data/uploads/{user_id}/{project_id}/{filename}

    This does NOT parse, chunk, embed, or index the PDF yet.
    That starts in Phase 4.
    """
    try:
        validate_pdf_upload(file)

        target_path = build_upload_path(
            upload_dir=settings.UPLOAD_DIR,
            user_id=user_id,
            project_id=project_id,
            filename=file.filename or "upload.pdf",
        )

        target_path.parent.mkdir(parents=True, exist_ok=True)

        with target_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        size_bytes = target_path.stat().st_size

        return UploadResponse(
            status="saved",
            user_id=user_id,
            project_id=project_id,
            filename=target_path.name,
            saved_path=str(target_path),
            size_bytes=size_bytes,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {exc}")

    finally:
        file.file.close()

@app.post("/parse/pdf", response_model=ParsePDFResponse)
def parse_uploaded_pdf(request: ParsePDFRequest):
    """
    Phase 4 endpoint.

    Takes an already-uploaded PDF and parses it into text chunks.
    Does NOT store chunks in a vector database yet.
    """
    try:
        pdf_path = get_uploaded_pdf_path(
            upload_dir=settings.UPLOAD_DIR,
            user_id=request.user_id,
            project_id=request.project_id,
            filename=request.filename,
        )

        if not pdf_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Uploaded PDF not found: {pdf_path}",
            )

        pages_parsed, chunks = parse_pdf_into_chunks(
            pdf_path=pdf_path,
            user_id=request.user_id,
            project_id=request.project_id,
            source_filename=pdf_path.name,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        return ParsePDFResponse(
            status="parsed",
            user_id=request.user_id,
            project_id=request.project_id,
            filename=pdf_path.name,
            pages_parsed=pages_parsed,
            chunks_created=len(chunks),
            chunks=chunks,
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {exc}")
    
@app.post("/parse/pdf/preview", response_model=ParsePDFPreviewResponse)
def preview_parse_uploaded_pdf(request: ParsePDFRequest):
    """
    Phase 4 preview endpoint.

    Parses PDF but returns only the first few chunks.
    Useful for debugging large PDFs.
    """
    try:
        pdf_path = get_uploaded_pdf_path(
            upload_dir=settings.UPLOAD_DIR,
            user_id=request.user_id,
            project_id=request.project_id,
            filename=request.filename,
        )

        if not pdf_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Uploaded PDF not found: {pdf_path}",
            )

        pages_parsed, chunks = parse_pdf_into_chunks(
            pdf_path=pdf_path,
            user_id=request.user_id,
            project_id=request.project_id,
            source_filename=pdf_path.name,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        return ParsePDFPreviewResponse(
            status="parsed_preview",
            user_id=request.user_id,
            project_id=request.project_id,
            filename=pdf_path.name,
            pages_parsed=pages_parsed,
            chunks_created=len(chunks),
            preview_chunks=chunks[:5],
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF preview: {exc}")