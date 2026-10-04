#!/usr/bin/env python3
"""Chunk and index a folder of text.

    python scripts/ingest.py --path ./corpus --source crr-575-2013

Chunking is paragraph-aware with a sentence overlap. Fixed-width chunking cuts
statutes mid-subsection, and a retrieved half-clause is worse than no hit at
all because the grader cannot tell it is truncated.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import uuid
from pathlib import Path

from rag_agent.graph.state import Document
from rag_agent.retrieval.qdrant_store import QdrantStore

MAX_CHARS = 1200
OVERLAP_SENTENCES = 1


def chunk(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= MAX_CHARS:
            buffer = f"{buffer}\n\n{para}" if buffer else para
            continue
        if buffer:
            chunks.append(buffer)
            tail = re.split(r"(?<=[.!?])\s+", buffer)[-OVERLAP_SENTENCES:]
            buffer = " ".join(tail) + "\n\n" + para
        else:
            buffer = para
    if buffer:
        chunks.append(buffer)
    return chunks


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--source", required=True)
    args = ap.parse_args()

    store = QdrantStore()
    await store.ensure_collection()

    docs: list[Document] = []
    for page, file in enumerate(sorted(Path(args.path).glob("*.txt")), start=1):
        for piece in chunk(file.read_text(encoding="utf-8")):
            docs.append(Document(id=str(uuid.uuid4()), text=piece,
                                 source=args.source, page=page))

    for i in range(0, len(docs), 128):
        await store.upsert(docs[i : i + 128])
        print(f"upserted {min(i + 128, len(docs))}/{len(docs)}")


if __name__ == "__main__":
    asyncio.run(main())
