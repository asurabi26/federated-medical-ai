"""
Builds per-hospital HospitalNoteStore instances from text documents. Until the real
clinical text dataset is provided, uses a small set of synthetic placeholder case notes
so the federated RAG layer is fully runnable and testable end to end. Swap
`load_synthetic_notes()` for a real loader (e.g. reading de-identified case reports per
institution) without touching anything downstream.
"""

from typing import Dict, List

from sentence_transformers import SentenceTransformer

from rag.hospital_notes_store import HospitalNoteStore

# Placeholder clinical notes per simulated hospital - intentionally includes one
# deliberate cross-hospital disagreement (hospital_2 vs hospital_4 on family history)
# so the conflict-detection logic in federated_retriever.py has something real to catch.
SYNTHETIC_NOTES: Dict[str, List[str]] = {
    "hospital_0": [
        "Patient presented with recurrent headaches over three weeks. MRI showed a mass "
        "in the left temporal lobe, consistent with glioma. No history of prior brain "
        "surgery. Family history of migraine but no history of malignancy.",
        "Follow-up imaging six weeks post-resection shows no evidence of residual tumor. "
        "Patient reports mild fatigue, no new neurological deficits.",
    ],
    "hospital_1": [
        "Case involves a well-circumscribed lesion at the meninges, favoring meningioma "
        "over glioma given imaging characteristics. Patient asymptomatic, lesion found "
        "incidentally on unrelated imaging.",
        "Radiologist notes slow growth over 18 months of surveillance imaging, consistent "
        "with a benign meningioma course.",
    ],
    "hospital_2": [
        "Patient with pituitary mass, elevated prolactin levels on labs. History of "
        "malignancy in first-degree relative (mother, breast cancer). No history of "
        "radiation exposure.",
        "Endocrine workup consistent with prolactinoma. Recommend dopamine agonist "
        "therapy prior to considering surgical intervention.",
    ],
    "hospital_3": [
        "Routine screening MRI in asymptomatic patient shows no abnormal findings. "
        "No history of neurological symptoms reported.",
    ],
    "hospital_4": [
        "Patient referred with similar presentation to a prior pituitary case, elevated "
        "prolactin. No history of malignancy in family reported at intake, patient "
        "declined further family history disclosure.",
        "Imaging shows small pituitary microadenoma, likely non-functioning. "
        "Conservative management recommended with repeat imaging in 12 months.",
    ],
}


def build_hospital_stores(embedding_model_name: str = "all-MiniLM-L6-v2") -> Dict[str, HospitalNoteStore]:
    embedding_model = SentenceTransformer(embedding_model_name)
    stores = {}
    for hospital_id, notes in SYNTHETIC_NOTES.items():
        store = HospitalNoteStore(hospital_id, embedding_model)
        store.ingest(notes)
        stores[hospital_id] = store
    return stores
