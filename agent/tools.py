"""
Tools available to the orchestrating agent. Each tool wraps a piece of the system we
already built - the trained federated model for classification, and the federated
retriever for pulling relevant clinical text. The agent decides which tool(s) a given
query needs; it never has direct access to raw hospital data, only to what these tools
choose to return.
"""

from typing import Dict, List, Optional

import torch

from data.dataset_loader import DatasetInfo
from rag.federated_retriever import FederatedRetriever


class ImageClassifierTool:
    def __init__(self, model: torch.nn.Module, dataset_info: DatasetInfo, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.dataset_info = dataset_info
        self.device = device

    def classify(self, image_tensor: torch.Tensor) -> Dict:
        """image_tensor: shape (C, H, W) or (1, C, H, W)."""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_prob, top_idx = probs.max(dim=0)
        return {
            "predicted_class": self.dataset_info.class_names[top_idx.item()],
            "confidence": round(top_prob.item(), 4),
            "all_class_probs": {
                name: round(probs[i].item(), 4) for i, name in enumerate(self.dataset_info.class_names)
            },
        }


class FederatedRetrievalTool:
    def __init__(self, retriever: FederatedRetriever):
        self.retriever = retriever

    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        result = self.retriever.retrieve(query, top_k_final=top_k)
        return {
            "snippets": [
                {"hospital": s.hospital_id, "text": s.text, "relevance_score": round(s.score, 4)}
                for s in result.snippets
            ],
            "hospitals_queried": result.hospitals_queried,
            "possible_conflict": result.possible_conflict,
            "conflict_note": result.conflict_note,
        }


TOOL_DESCRIPTIONS = """
Available tools:

1. classify_scan(image) - runs the federated-trained image classifier on a medical scan.
   Use when the user provides or references an image and wants a classification/finding.

2. retrieve_case_notes(query) - searches clinical notes across all simulated hospitals'
   private stores independently and combines results. Use when the user asks about
   similar documented cases, patterns across cases, or clinical context/history.
   Always check the `possible_conflict` field in the result - if true, you MUST
   surface the conflict_note in your answer rather than silently picking one source.

Use both tools together when the query needs both an image finding AND supporting
clinical context. Never fabricate a finding or a case note - only report what the
tools actually returned.
"""
