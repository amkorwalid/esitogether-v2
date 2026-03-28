"""
PDF → Markdown converter using Docling + PyMuPDF.
Processes large PDFs in batches to manage memory usage.
"""

from pathlib import Path
import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter


def docs_to_markdown(pdf_path: Path, output_dir: Path, tmp_dir: Path, batch_size: int = 10) -> bool:
    """
    Convert a PDF document to Markdown and save it to output_dir.

    Args:
        pdf_path:   Path to the source PDF file.
        output_dir: Directory where the .md file will be written.
        tmp_dir:    Directory for intermediate per-batch PDFs.
        batch_size: Number of pages processed per batch (default 10).

    Returns:
        True on success, False on any error.
    """

    try:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
    except Exception as exc:
        print(f"[converter] Error: {exc}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{pdf_path.stem}.md"
    if output_file.exists():
        try:
            output_file.unlink()
        except Exception as exc:
            print(f"[converter] Cannot remove existing md file: {exc}")
            return False

    try:
        output_file.write_text("\n", encoding="utf-8")
    except Exception as exc:
        print(f"[converter] Cannot create md file: {exc}")
        return False

    try:
        converter = DocumentConverter()
    except Exception as exc:
        print(f"[converter] DocumentConverter init error: {exc}")
        return False

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"[converter] Converting '{pdf_path.name}' ({total_pages} pages) in batches of {batch_size}...")

    for i in range(0, total_pages, batch_size):
        start_page = i
        end_page = min(i + batch_size - 1, total_pages - 1)

        batch_doc = fitz.open()
        batch_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)

        temp_pdf = tmp_dir / f"{pdf_path.stem}_tmp_{start_page}_{end_page}.pdf"
        batch_doc.save(temp_pdf)
        batch_doc.close()

        try:
            result = converter.convert(temp_pdf)
            markdown = result.document.export_to_markdown()
        except Exception as exc:
            print(f"[converter] Conversion error on pages {start_page}-{end_page}: {exc}")
            doc.close()
            return False

        try:
            with open(output_file, "a", encoding="utf-8") as fh:
                fh.write(markdown)
        except Exception as exc:
            print(f"[converter] Write error: {exc}")
            doc.close()
            return False

        try:
            temp_pdf.unlink()
        except Exception as exc:
            print(f"[converter] Cannot remove temp PDF: {exc}")

    doc.close()
    print(f"[converter] Saved markdown to {output_file}")
    return True
