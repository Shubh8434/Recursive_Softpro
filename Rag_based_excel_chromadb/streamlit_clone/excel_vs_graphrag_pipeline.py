"""Fallback utilities for spreadsheet row lookups used by the Streamlit app."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

_SEPARATOR = "\n- "


def _normalize_tokens(text: str) -> List[str]:
    """Return lowercased alphanumeric tokens for rough matching."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _keyword_score(document_tokens: Iterable[str], keywords: Iterable[str]) -> int:
    """Score a document by counting keyword overlaps."""
    doc_set = set(document_tokens)
    return sum(1 for kw in keywords if kw in doc_set)


def row_lookup_records(documents: List[str], question: str, top_k: int = 3) -> List[Dict[str, str | int]]:
    """Return structured metadata for the best matching rows."""
    question_tokens = [tok for tok in _normalize_tokens(question) if len(tok) > 2]
    if not question_tokens:
        return []

    scored = []
    for idx, doc in enumerate(documents):
        doc_tokens = _normalize_tokens(doc)
        score = _keyword_score(doc_tokens, question_tokens)
        if score > 0:
            scored.append((score, idx, doc))

    if not scored:
        return []

    scored.sort(reverse=True)
    top = scored[:top_k]
    records: List[Dict[str, str | int]] = []
    for rank, (score, idx, doc) in enumerate(top, start=1):
        records.append(
            {
                "rank": rank,
                "row_index": idx + 1,
                "score": score,
                "text": doc,
            }
        )
    return records


def row_lookup(documents: List[str], question: str, top_k: int = 3) -> str:
    """Return a formatted string describing the best row matches."""
    records = row_lookup_records(documents, question, top_k=top_k)
    if not records:
        if not _normalize_tokens(question):
            return "Please provide a more descriptive question for row lookup."
        return "No matching rows found for the supplied question."

    lines = ["Top matching rows:"]
    for record in records:
        lines.append(f"{record['rank']}. Row #{record['row_index']} (score={record['score']})")
        lines.append(str(record["text"]))
    return _SEPARATOR.join(lines)
