"""Load CSV, JSON, PDF, plain text → list[str] document units."""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

# Supported extensions
_PLAIN_EXTS = {".txt", ".md"}
_STRUCTURED_EXTS = {".csv", ".json", ".pdf"}
_ALL_EXTS = _PLAIN_EXTS | _STRUCTURED_EXTS


class DocumentLoader:
    """Load documents from various file formats into a list of text strings.

    Each returned string is one *document unit* — a row (CSV), a page (PDF),
    an item (JSON), or an entire file (TXT/MD).
    """

    def load(self, source: Union[str, Path], *, column: str | None = None) -> list[str]:
        """Load document(s) from *source* path.

        Parameters
        ----------
        source : str | Path
            Path to a file on disk.
        column : str | None
            For CSV files, the name of the column containing text.
            If ``None``, the loader will auto-detect the first text-like column.

        Returns
        -------
        list[str]
            One string per document unit.

        Raises
        ------
        ValueError
            If the file extension is unsupported.
        FileNotFoundError
            If *source* does not exist.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

        ext = path.suffix.lower()
        if ext not in _ALL_EXTS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported extensions: {sorted(_ALL_EXTS)}"
            )

        loader = {
            ".csv": self._load_csv,
            ".json": self._load_json,
            ".pdf": self._load_pdf,
            ".txt": self._load_plain,
            ".md": self._load_plain,
        }[ext]

        return loader(path, column=column)

    # ── Private loaders ─────────────────────────────────────────────────────

    @staticmethod
    def _load_plain(path: Path, **_kwargs) -> list[str]:
        text = path.read_text(encoding="utf-8")
        return [text] if text.strip() else []

    @staticmethod
    def _load_csv(path: Path, *, column: str | None = None, **_kwargs) -> list[str]:
        """Read CSV and return text from the specified or auto-detected column."""
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return []

            if column and column in reader.fieldnames:
                col = column
            else:
                # Auto-detect: pick the first column whose name looks text-like
                text_hints = {"text", "content", "body", "description", "message", "document"}
                col = next(
                    (c for c in reader.fieldnames if c.lower() in text_hints),
                    reader.fieldnames[0],
                )
                if column and column != col:
                    logger.warning(
                        "Column '%s' not found in CSV; falling back to '%s'", column, col
                    )

            docs: list[str] = []
            for row in reader:
                val = (row.get(col) or "").strip()
                if val:
                    docs.append(val)
            return docs

    @staticmethod
    def _load_json(path: Path, **_kwargs) -> list[str]:
        """Flatten a JSON file to a list of text strings.

        Supports:
        - A list of strings
        - A list of dicts (extracts first string-valued field from each)
        - A single string
        """
        raw = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(raw, str):
            return [raw] if raw.strip() else []

        if isinstance(raw, list):
            docs: list[str] = []
            for item in raw:
                if isinstance(item, str):
                    if item.strip():
                        docs.append(item.strip())
                elif isinstance(item, dict):
                    # Try common text keys first, then first string value
                    text_keys = ["text", "content", "body", "document", "message"]
                    val = None
                    for k in text_keys:
                        if k in item and isinstance(item[k], str):
                            val = item[k].strip()
                            break
                    if val is None:
                        # Fallback: first string value
                        for v in item.values():
                            if isinstance(v, str) and v.strip():
                                val = v.strip()
                                break
                    if val:
                        docs.append(val)
            return docs

        raise ValueError(f"Unsupported JSON structure: {type(raw).__name__}")

    @staticmethod
    def _load_pdf(path: Path, **_kwargs) -> list[str]:
        """Extract text per page using pdfminer.six."""
        try:
            from pdfminer.high_level import extract_text
            from pdfminer.layout import LAParams
        except ImportError:
            raise ImportError(
                "pdfminer.six is required for PDF loading. "
                "Install it with: pip install pdfminer.six"
            )

        # pdfminer doesn't expose simple per-page extraction natively,
        # so we extract the full text and split on form-feed characters
        full_text = extract_text(str(path), laparams=LAParams())
        pages = full_text.split("\x0c")
        return [p.strip() for p in pages if p.strip()]
