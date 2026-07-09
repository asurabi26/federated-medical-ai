"""
Each simulated hospital keeps its own private vector store of clinical text (case notes,
de-identified report excerpts). Stores never merge - this mirrors the same constraint
that motivates federated learning: institutions can't pool raw patient data, text or
image. Only retrieved snippets (not whole documents, not the store itself) ever leave
a hospital's boundary, and only in response to a specific query.

Uses FAISS for local similarity search - same library as the earlier MedReport-QA
project, swapped from a single global index to N independent per-hospital indexes.
"""

import os
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedSnippet:
    hospital_id: str
    text: str
    score: float


class HospitalNoteStore:
    def __init__(self, hospital_id: str, embedding_model: SentenceTransformer):
        self.hospital_id = hospital_id
        self.embedding_model = embedding_model
        self.texts: List[str] = []
        self.index = None

    def ingest(self, documents: List[str], chunk_size: int = 300):
        chunks = []
        for doc in documents:
            words = doc.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():
                    chunks.append(chunk)

        if not chunks:
            return

        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)
        self.texts.extend(chunks)

    def query(self, query_text: str, top_k: int = 3) -> List[RetrievedSnippet]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_emb = self.embedding_model.encode([query_text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedSnippet(hospital_id=self.hospital_id, text=self.texts[idx], score=float(score)))
        return results
