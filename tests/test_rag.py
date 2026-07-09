import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


def sentence_transformers_available():
    try:
        import sentence_transformers  # noqa
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not sentence_transformers_available(), reason="sentence-transformers not installed")
def test_federated_retriever_queries_all_hospitals():
    from rag.federated_retriever import FederatedRetriever
    from rag.ingest import build_hospital_stores

    stores = build_hospital_stores()
    retriever = FederatedRetriever(stores)

    result = retriever.retrieve("pituitary mass prolactin", top_k_final=5)

    assert len(result.hospitals_queried) == len(stores)
    assert len(result.snippets) > 0
    # the top result should be relevant to pituitary/prolactin, coming from one of the
    # hospitals that actually has a matching case note
    assert any("pituitary" in s.text.lower() or "prolactin" in s.text.lower() for s in result.snippets)


@pytest.mark.skipif(not sentence_transformers_available(), reason="sentence-transformers not installed")
def test_empty_store_returns_no_snippets():
    from sentence_transformers import SentenceTransformer

    from rag.hospital_notes_store import HospitalNoteStore

    model = SentenceTransformer("all-MiniLM-L6-v2")
    store = HospitalNoteStore("empty_hospital", model)

    results = store.query("any query", top_k=3)
    assert results == []
