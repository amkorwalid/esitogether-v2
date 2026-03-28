from pathlib import Path
from docs_to_markdown import docs_to_markdown


folder_path_raw = Path(__file__).parent.parent / "data" / "raw"
if not folder_path_raw.exists():
    raise FileNotFoundError(f"The specified raw data folder does not exist: {folder_path_raw}")

folder_path_clean = Path(__file__).parent.parent / "data" / "clean"
if not folder_path_clean.exists():
    folder_path_clean.mkdir(parents=True)
    
    
files = [f.name for f in folder_path_raw.iterdir() if f.is_file()]

for file in files:
    
    input_path = folder_path_raw / file
    output_dir = folder_path_clean
    
    try:
        
        success = docs_to_markdown(input_path, output_dir)
        if success:
            print(f"Successfully converted {input_path} to Markdown and saved to {output_dir}")
        else:
            print(f"Failed to convert {input_path} to Markdown.")

    except Exception as e:
        print(f"An error occurred while processing {input_path}: {e}")


