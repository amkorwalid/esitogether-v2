import tiktoken
from pathlib import Path

def count_tokens_markdown(path: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    
    text = Path(path).read_text(encoding="utf-8")
    tokens = encoding.encode(text)
    return len(tokens)

folder_path = Path(__file__).parent.parent / "data" / "clean"
files = [f.name for f in folder_path.iterdir() if f.is_file()]
for file in files:
    md_path = folder_path / file
    num_tokens = count_tokens_markdown(md_path)
    print(f"File: {md_path}, Tokens: {num_tokens}")