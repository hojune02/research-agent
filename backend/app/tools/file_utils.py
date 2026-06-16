import re
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


ALLOWED_PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
}


def sanitize_filename(filename: str) -> str:
    """
    Remove path components and unsafe characters from uploaded filename.
    Example:
        '../../../evil.pdf' -> 'evil.pdf'
        'my paper (final).pdf' -> 'my_paper_final.pdf'
    """
    raw_name = Path(filename).name
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw_name)
    return safe_name.strip("._") or f"upload_{uuid4().hex}.pdf"


def validate_pdf_upload(file: UploadFile) -> None:
    """
    Minimal MVP PDF validation.
    We check extension and content type.
    Later, Phase 4 will actually parse the file.
    """
    filename = file.filename or ""

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are allowed.")

    if file.content_type and file.content_type not in ALLOWED_PDF_CONTENT_TYPES:
        raise ValueError(f"Invalid content type: {file.content_type}. Only PDF files are allowed.")


def build_upload_path(upload_dir: str, user_id: str, project_id: str, filename: str) -> Path:
    """
    Build upload path:
        data/uploads/{user_id}/{project_id}/{filename}
    """
    safe_user_id = re.sub(r"[^a-zA-Z0-9._-]+", "_", user_id).strip("._")
    safe_project_id = re.sub(r"[^a-zA-Z0-9._-]+", "_", project_id).strip("._")
    safe_filename = sanitize_filename(filename)

    if not safe_user_id:
        raise ValueError("user_id cannot be empty after sanitization.")

    if not safe_project_id:
        raise ValueError("project_id cannot be empty after sanitization.")

    return Path(upload_dir) / safe_user_id / safe_project_id / safe_filename