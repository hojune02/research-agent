import re
from pathlib import Path

from pypdf import PdfReader

from app.schemas import DocumentChunk


def normalize_whitespace(text: str) -> str:
    """
    Cleans repeated whitespace while preserving readable text.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages_from_pdf(pdf_path: Path) -> list[tuple[int, str]]:
    """
    Extract text from a PDF page by page.

    Returns:
        [(page_number, text), ...]
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []

    for idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        clean_text = normalize_whitespace(raw_text)

        if clean_text:
            pages.append((idx, clean_text))

    return pages


def split_text_into_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Simple character-based chunking.

    Example:
        chunk_size=900, chunk_overlap=150

    This means:
        chunk 1: chars 0-900
        chunk 2: chars 750-1650
        chunk 3: chars 1500-2400

    The overlap helps preserve context across chunk boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    text = text.strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - chunk_overlap

    return chunks


def parse_pdf_into_chunks(
    pdf_path: Path,
    user_id: str,
    project_id: str,
    source_filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[int, list[DocumentChunk]]:
    """
    Parse PDF into metadata-rich chunks.

    Returns:
        pages_parsed, chunks
    """
    pages = extract_pages_from_pdf(pdf_path)
    all_chunks: list[DocumentChunk] = []

    source_stem = Path(source_filename).stem
    safe_source_stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", source_stem).strip("._")

    for page_number, page_text in pages:
        page_chunks = split_text_into_chunks(
            text=page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
            chunk_id = f"{safe_source_stem}_p{page_number}_c{chunk_index}"

            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    user_id=user_id,
                    project_id=project_id,
                    source=source_filename,
                    page=page_number,
                    text=chunk_text,
                    char_count=len(chunk_text),
                )
            )

    return len(pages), all_chunks