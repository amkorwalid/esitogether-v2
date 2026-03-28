from docling.document_converter import DocumentConverter
from pathlib import Path
import fitz  # PyMuPDF

def docs_to_markdown(pdf_path: Path, output_dir: Path, batch_size: int = 10):
    """
    Convert a PDF document to Markdown format and save it to the specified output directory.
    Args:
        pdf_path (Path): The path to the PDF document to be converted.
        output_dir (Path): The directory where the converted Markdown file will be saved.
        
    Returns:
        bool: True if the conversion and saving are successful, False otherwise.
    """
    
    # Check if the PDF file exists
    try:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"The specified PDF file does not exist: {pdf_path}")
    except Exception as e:
        print(f"Error: {e}")
        return False  
    
    # Create the output directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Remove the output file if it already exists to avoid appending to an old file
    output_file = output_dir / f"{pdf_path.stem}.md"
    if output_file.exists():
        try:
            output_file.unlink()
        except Exception as e:
            print(f"Error occurred while removing existing Markdown file: {e}")
            return False
    
    # Create a new empty Markdown file
    try:
        with open(output_dir / f"{pdf_path.stem}.md", "w", encoding="utf-8") as f:
            f.write("\n")
    except Exception as e:
        print(f"Error occurred while saving Markdown file: {e}")
        return False
    
    # Initialize the DocumentConverter
    try:
        converter = DocumentConverter()
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    # Process the PDF in batches of pages to manage memory usage
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    for i in range(0, total_pages, batch_size):
        
        # Create a new empty PDF for the current batch
        new_doc = fitz.open()
        
        # Calculate the page range for this batch
        start_page = i
        end_page = min(i + batch_size - 1, total_pages - 1)
        
        # Copy the range of pages from the source to the new document
        # from_page and to_page are both 0-indexed and inclusive
        new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
        
        # save the new document to a temporary file
        temp_pdf_path = output_dir / 'tmp' /f"{pdf_path.stem}_temp_{start_page}_{end_page}.pdf"
        new_doc.save(temp_pdf_path)
        
        # Convert the new document to Markdown
        try:
            result = converter.convert(temp_pdf_path)
            markdown = result.document.export_to_markdown()
        except Exception as e:
            print(f"Error occurred during conversion of pages {start_page} to {end_page}: {e}")
            new_doc.close()
            return False
        
        # Append the converted Markdown to the output file
        try: 
            with open(output_dir / f"{pdf_path.stem}.md", "a", encoding="utf-8") as f:
                f.write(markdown)
        except Exception as e:
            print(f"Error occurred while saving Markdown file: {e}")
            new_doc.close()
            return False
        
        new_doc.close()
        
        # Remove the temporary PDF file
        try:
            temp_pdf_path.unlink()
        except Exception as e:
            print(f"Error occurred while removing temporary PDF file: {e}")

    doc.close()
    
    return True

