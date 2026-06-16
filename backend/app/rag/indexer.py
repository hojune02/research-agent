from app.config import settings
from app.rag.pdf_parser import parse_pdf_into_chunks
from app.rag.vectorstore import index_chunks
from app.schemas import IndexPDFResponse
from app.tools.file_utils import get_uploaded_pdf_path


def index_uploaded_pdf(
    user_id: str,
    project_id: str,
    filename: str,
) -> IndexPDFResponse:
    """
    Parse an uploaded PDF and index its chunks into Chroma.
    """
    pdf_path = get_uploaded_pdf_path(
        upload_dir=settings.UPLOAD_DIR,
        user_id=user_id,
        project_id=project_id,
        filename=filename,
    )

    if not pdf_path.exists():
        raise FileNotFoundError(f"Uploaded PDF not found: {pdf_path}")

    pages_parsed, chunks = parse_pdf_into_chunks(
        pdf_path=pdf_path,
        user_id=user_id,
        project_id=project_id,
        source_filename=pdf_path.name,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    chunks_indexed = index_chunks(chunks)

    return IndexPDFResponse(
        status="indexed",
        user_id=user_id,
        project_id=project_id,
        filename=pdf_path.name,
        collection=settings.CHROMA_COLLECTION,
        pages_parsed=pages_parsed,
        chunks_indexed=chunks_indexed,
        embedding_model=settings.EMBEDDING_MODEL,
    )