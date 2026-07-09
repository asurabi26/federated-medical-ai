"""
Queries every hospital's private note store independently (no store ever sees another
store's contents), then combines the retrieved snippets - this is "federated RAG."

Two things this layer does that a single-store RAG system doesn't have to think about:
1. Ranking across sources fairly (a naive concat would let whichever hospital has more
   documents dominate results regardless of relevance)
2. Detecting when different hospitals' retrieved snippets appear to disagree, so that
   conflict can be surfaced explicitly rather than silently resolved by picking one -
   this is the "honest about uncertainty" principle carried through from the RAG design
   discussed earlier, now applied to a genuinely multi-source setting.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from rag.hospital_notes_store import HospitalNoteStore, RetrievedSnippet


@dataclass
class FederatedRetrievalResult:
    snippets: List[RetrievedSnippet]
    hospitals_queried: List[str]
    possible_conflict: bool
    conflict_note: str = ""


class FederatedRetriever:
    def __init__(self, hospital_stores: Dict[str, HospitalNoteStore]):
        self.hospital_stores = hospital_stores

    def retrieve(self, query: str, top_k_per_hospital: int = 3, top_k_final: int = 5) -> FederatedRetrievalResult:
        all_snippets: List[RetrievedSnippet] = []
        for hospital_id, store in self.hospital_stores.items():
            results = store.query(query, top_k=top_k_per_hospital)
            all_snippets.extend(results)

        all_snippets.sort(key=lambda s: s.score, reverse=True)
        top_snippets = all_snippets[:top_k_final]

        conflict, note = self._check_conflict(top_snippets)

        return FederatedRetrievalResult(
            snippets=top_snippets,
            hospitals_queried=list(self.hospital_stores.keys()),
            possible_conflict=conflict,
            conflict_note=note,
        )

    def _check_conflict(self, snippets: List[RetrievedSnippet]) -> Tuple[bool, str]:
        """
        Lightweight heuristic conflict flag: if top snippets come from multiple distinct
        hospitals AND contain simple negation-pattern mismatches (e.g. one contains
        'no history of X' and another contains 'history of X' for overlapping terms),
        flag it for the agent layer to handle explicitly rather than silently merge.
        This is deliberately simple - a real system would use an NLI model here: this
        heuristic exists so the orchestration logic has something concrete to act on,
        with a clear seam to swap in a proper contradiction-detection model later.
        """
        distinct_hospitals = {s.hospital_id for s in snippets}
        if len(distinct_hospitals) < 2:
            return False, ""

        texts_lower = [s.text.lower() for s in snippets]
        negation_markers = ["no history of", "no evidence of", "denies", "negative for"]
        for i, text_a in enumerate(texts_lower):
            for marker in negation_markers:
                if marker in text_a:
                    subject = text_a.split(marker, 1)[1].strip().split(".")[0][:40]
                    for j, text_b in enumerate(texts_lower):
                        if i != j and subject and subject in text_b and marker not in text_b:
                            return True, (
                                f"Snippets from different hospitals may conflict regarding '{subject.strip()}' - "
                                f"one source notes its absence, another does not. Review both sources directly."
                            )
        return False, ""
