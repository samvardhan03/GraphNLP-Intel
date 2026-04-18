"""Parse raw MIME emails, extract: sender, subject, body, attachments, amounts."""

from __future__ import annotations

import email
import email.policy
import re
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Union

# Regex for monetary amounts like $1,234.56 or $50
_MONEY_RE = re.compile(r"\$[\d,]+\.?\d*")

# Common merchant / vendor patterns (rough heuristic for non-NER extraction)
_MERCHANT_RE = re.compile(
    r"(?:from|at|to|paid|payment\s+to|merchant[:\s])\s+([A-Z][A-Za-z0-9\s&'.,-]{2,30})",
    re.IGNORECASE,
)


@dataclass
class ParsedEmail:
    """Structured representation of a parsed email."""

    message_id: str = ""
    sender: str = ""
    recipients: list[str] = field(default_factory=list)
    subject: str = ""
    body: str = ""
    date: datetime | None = None
    amounts: list[float] = field(default_factory=list)
    merchants: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Flatten email to a single text string for pipeline ingestion."""
        parts = []
        if self.subject:
            parts.append(f"Subject: {self.subject}")
        if self.sender:
            parts.append(f"From: {self.sender}")
        if self.recipients:
            parts.append(f"To: {', '.join(self.recipients)}")
        if self.date:
            parts.append(f"Date: {self.date.isoformat()}")
        if self.body:
            parts.append(f"\n{self.body}")
        return "\n".join(parts)


class EmailParser:
    """Parse raw MIME email strings or .eml files into structured data."""

    def parse(self, source: Union[str, Path]) -> ParsedEmail:
        """Parse a MIME email from a raw string or file path.

        Parameters
        ----------
        source : str | Path
            Either a raw MIME email string or a path to an ``.eml`` file.

        Returns
        -------
        ParsedEmail
            Structured representation with extracted metadata and entities.
        """
        path = Path(source) if not isinstance(source, str) or "\n" not in source else None

        if path is not None and path.exists() and path.suffix.lower() == ".eml":
            raw = path.read_text(encoding="utf-8", errors="replace")
        else:
            raw = str(source)

        msg = email.message_from_string(raw, policy=email.policy.default)
        return self._extract(msg)

    def _extract(self, msg: email.message.Message) -> ParsedEmail:
        """Extract fields from a parsed email.message.Message."""
        result = ParsedEmail()

        # Message ID
        result.message_id = msg.get("Message-ID", "") or ""

        # Sender
        _, addr = parseaddr(msg.get("From", ""))
        result.sender = addr or msg.get("From", "")

        # Recipients
        for header in ("To", "Cc"):
            raw = msg.get(header, "")
            if raw:
                for _, addr in email.utils.getaddresses([raw]):
                    if addr:
                        result.recipients.append(addr)

        # Subject
        result.subject = msg.get("Subject", "") or ""

        # Date
        date_str = msg.get("Date")
        if date_str:
            try:
                result.date = parsedate_to_datetime(date_str)
            except (ValueError, TypeError):
                pass

        # Body — prefer plain text over HTML
        result.body = self._get_body(msg)

        # Attachments
        for part in msg.walk():
            filename = part.get_filename()
            if filename:
                result.attachments.append(filename)

        # Extract monetary amounts from body and subject
        text = f"{result.subject} {result.body}"
        result.amounts = self._extract_amounts(text)
        result.merchants = self._extract_merchants(text)

        return result

    @staticmethod
    def _get_body(msg: email.message.Message) -> str:
        """Extract the best plain-text body from a (possibly multipart) message."""
        if not msg.is_multipart():
            ct = msg.get_content_type()
            if ct == "text/plain":
                payload = msg.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="replace").strip()
            elif ct == "text/html":
                payload = msg.get_payload(decode=True)
                if payload:
                    # Strip HTML tags as a rough fallback
                    html = payload.decode("utf-8", errors="replace")
                    return re.sub(r"<[^>]+>", " ", html).strip()
            return ""

        # Multipart — prefer text/plain
        plain_parts: list[str] = []
        html_parts: list[str] = []

        for part in msg.walk():
            ct = part.get_content_type()
            if part.get_filename():
                continue  # skip attachments
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            text = payload.decode("utf-8", errors="replace").strip()
            if ct == "text/plain":
                plain_parts.append(text)
            elif ct == "text/html":
                html_parts.append(text)

        if plain_parts:
            return "\n".join(plain_parts)
        if html_parts:
            return re.sub(r"<[^>]+>", " ", "\n".join(html_parts)).strip()
        return ""

    @staticmethod
    def _extract_amounts(text: str) -> list[float]:
        """Extract monetary amounts from text using regex."""
        amounts: list[float] = []
        for match in _MONEY_RE.finditer(text):
            raw = match.group().replace("$", "").replace(",", "")
            try:
                amounts.append(float(raw))
            except ValueError:
                continue
        return amounts

    @staticmethod
    def _extract_merchants(text: str) -> list[str]:
        """Extract merchant names from text using regex heuristics."""
        merchants: list[str] = []
        seen: set[str] = set()
        for match in _MERCHANT_RE.finditer(text):
            name = match.group(1).strip().rstrip(".,")
            normalized = name.lower()
            if normalized not in seen and len(name) > 2:
                merchants.append(name)
                seen.add(normalized)
        return merchants
