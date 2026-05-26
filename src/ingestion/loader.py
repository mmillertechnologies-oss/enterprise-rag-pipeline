import hashlib
import logging
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv", ".rst"}


def load_directory(path: str) -> Generator[dict, None, None]:
    """Yield document dicts from all supported files under path."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    for file_path in sorted(root.rglob("*")):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        doc = _load_file(file_path)
        if doc:
            yield doc


def load_file(path: str) -> dict:
    result = _load_file(Path(path))
    if result is None:
        raise ValueError(f"Could not load file: {path}")
    return result


def _load_file(path: Path) -> dict | None:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            return None
        return {
            "content": content,
            "source": str(path),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "doc_id": hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16],
            "size_bytes": path.stat().st_size,
        }
    except Exception as exc:
        logger.warning("Skipping %s: %s", path, exc)
        return None
