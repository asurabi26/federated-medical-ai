"""
Serves the trained federated model, the federated RAG retriever, and the LangGraph
agent through a small API. Models/stores are loaded once at startup; the /chat
endpoint is the main demo surface - it's the same orchestration used in the interview
pitch: "the agent decides which tools to call, grounds its answer in real results."
"""

import io
import os
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from agent.orchestrator import build_agent_graph, run_agent
from agent.tools import FederatedRetrievalTool, ImageClassifierTool
from data.dataset_loader import DatasetInfo
from federated.model import SimpleMedicalCNN
from rag.federated_retriever import FederatedRetriever
from rag.ingest import build_hospital_stores

app = FastAPI(
    title="Federated Medical AI System",
    description="Federated image classification + federated RAG over private clinical "
    "notes, orchestrated by a LangGraph agent. Research/portfolio project - not a "
    "diagnostic tool.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/checkpoints/global_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_state = {"model": None, "dataset_info": None, "classifier_tool": None,
          "retriever_tool": None, "agent_graph": None}


@app.on_event("startup")
def load_everything():
    dataset_info = DatasetInfo(
        num_classes=4,
        class_names=["glioma", "meningioma", "pituitary", "no_tumor"],
        input_shape=(1, 64, 64),
    )
    model = SimpleMedicalCNN(in_channels=dataset_info.input_shape[0], num_classes=dataset_info.num_classes)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"[startup] loaded trained weights from {MODEL_PATH}")
    else:
        print(f"[startup] WARNING: no checkpoint found at {MODEL_PATH} - "
              f"using randomly initialized weights. Run federated/train_federated.py "
              f"and save the model first.")

    _state["model"] = model
    _state["dataset_info"] = dataset_info
    _state["classifier_tool"] = ImageClassifierTool(model, dataset_info, device=DEVICE)

    try:
        hospital_stores = build_hospital_stores()
        retriever = FederatedRetriever(hospital_stores)
        _state["retriever_tool"] = FederatedRetrievalTool(retriever)
        print("[startup] federated RAG stores built")
    except Exception as e:
        print(f"[startup] WARNING: could not build RAG stores: {e}")
        _state["retriever_tool"] = None

    if os.environ.get("OPENAI_API_KEY"):
        _state["agent_graph"] = build_agent_graph(_state["classifier_tool"], _state["retriever_tool"])
        print("[startup] agent graph built")
    else:
        print("[startup] WARNING: OPENAI_API_KEY not set - /chat endpoint will be unavailable")


_image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    tools_used: list
    classification_result: Optional[dict] = None
    retrieval_result: Optional[dict] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": _state["model"] is not None,
        "rag_loaded": _state["retriever_tool"] is not None,
        "agent_available": _state["agent_graph"] is not None,
    }


@app.post("/classify")
async def classify_scan(file: UploadFile = File(...)):
    if _state["classifier_tool"] is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    tensor = _image_transform(image)
    result = _state["classifier_tool"].classify(tensor)
    return result


@app.get("/retrieve")
def retrieve_notes(query: str, top_k: int = 5):
    if _state["retriever_tool"] is None:
        raise HTTPException(status_code=503, detail="Federated RAG not loaded")
    return _state["retriever_tool"].retrieve(query, top_k=top_k)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, file: Optional[UploadFile] = File(None)):
    if _state["agent_graph"] is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not available - set OPENAI_API_KEY and restart the API",
        )

    image_tensor = None
    if file is not None:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_tensor = _image_transform(image)

    result = run_agent(_state["agent_graph"], request.query, image_tensor=image_tensor)
    return ChatResponse(**result)
