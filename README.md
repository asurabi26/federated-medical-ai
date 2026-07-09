# Federated Medical AI: Federated Learning + Federated RAG + Agent Orchestration, with a Custom CUDA Kernel

A research-grade system combining three things that are each individually uncommon on
an entry-level portfolio, wired together as one coherent architecture rather than
three disconnected demos:

1. **Federated learning** for medical image classification across simulated hospitals
   with realistic non-IID data, using FedAvg
2. **A custom CUDA kernel** that quantizes model weight updates before transmission,
   cutting communication cost - the real bottleneck federated learning has to deal with
3. **Federated RAG + a LangGraph agent** that retrieves from each hospital's private
   clinical notes independently and orchestrates between the classifier and the
   retriever, surfacing cross-source conflicts explicitly instead of hiding them

**This is a research/portfolio project, not a diagnostic tool.** No component of this
system should be used, or is designed, to make real clinical decisions. Every output
is framed as "the model flagged" or "the model's prediction," never as a diagnosis.

## Why this combination, not three separate projects

Federated learning and federated RAG exist for the same underlying reason: hospitals
can't pool raw patient data (images or text) across institutions, for real legal and
ethical reasons (HIPAA). Building both under one system, with an agent that decides
which one a given query needs, demonstrates understanding of *why* the constraint
exists, not just how to call the relevant libraries.

## Architecture

```
                     ┌─────────────────────────────────────┐
                     │         5 simulated hospitals         │
                     │   (non-IID image + private notes)     │
                     └───────────────┬───────────────────────┘
                                      │
            ┌─────────────────────────┴──────────────────────────┐
            │                                                      │
   ┌────────▼─────────┐                                 ┌─────────▼──────────┐
   │  Federated image   │                                 │   Federated RAG     │
   │  classification     │                                 │   (per-hospital     │
   │  (FedAvg, local      │                                 │   FAISS stores,     │
   │  training per round) │                                 │   queried           │
   │                       │                                 │   independently)    │
   │  Weight updates       │                                 └─────────┬──────────┘
   │  compressed via        │                                          │
   │  CUSTOM CUDA KERNEL     │                                          │
   │  before aggregation      │                                        │
   └────────┬─────────────────┘                                        │
            │                                                          │
            └───────────────────┬──────────────────────────────────────┘
                                 │
                       ┌─────────▼──────────┐
                       │  LangGraph agent     │
                       │  orchestrator        │
                       │  (routes query to     │
                       │  classifier, retriever,│
                       │  or both; surfaces     │
                       │  conflicts explicitly)  │
                       └─────────┬───────────────┘
                                 │
                       ┌─────────▼──────────┐
                       │   FastAPI serving    │
                       │  /classify /retrieve  │
                       │  /chat                │
                       └────────────────────────┘
```

## Project structure

```
data/             pluggable dataset interface - swap in the real dataset here
federated/        FedAvg training loop, non-IID partitioning, model, compression
cuda_kernels/      custom CUDA quantization kernel + benchmark harness
notebooks/         Colab notebook to build/run/benchmark the CUDA kernel (needs a GPU)
rag/               per-hospital private vector stores + federated retriever
agent/             LangGraph orchestrator + tool definitions
api/               FastAPI serving layer
tests/             pytest suite
```

## Running the federated learning + RAG + agent system (no GPU required for this part)

```bash
pip install -r requirements.txt

# Run federated training on the synthetic placeholder dataset
python -m federated.train_federated

# Run the API (agent /chat endpoint requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
uvicorn api.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs.

## Running the CUDA kernel (requires a GPU - use Google Colab)

This is the one part of the system that needs an actual NVIDIA GPU to compile and run.
Since not everyone has one locally, use the provided Colab notebook:

1. Open `notebooks/run_all_colab.ipynb` in Google Colab (this is the tested, working
   version - it avoids the numpy/ninja pitfalls documented below)
2. Runtime → Change runtime type → GPU (the free T4 tier is enough)
3. Upload this repo (as a zip, or clone from GitHub once pushed) and run the notebook cells top to bottom

The notebook: compiles the kernel, sanity-checks it, runs the federated training loop
with GPU-accelerated compression, then benchmarks naive-GPU vs. custom-CUDA
quantization and plots the results.

**Verified results (Google Colab, Tesla T4 GPU, 2026-07-09):** the custom CUDA kernel
ran **1.5x-2.7x faster than PyTorch's own quantization ops**, on the same GPU, for the
same operation, across tensor sizes from 10K to 5M elements:

| Tensor size | Naive GPU (PyTorch ops) | Custom CUDA kernel | Speedup |
|---|---|---|---|
| 10,000 | 0.330 ms | 0.138 ms | **2.39x** |
| 100,000 | 0.251 ms | 0.168 ms | **1.50x** |
| 1,000,000 | 0.408 ms | 0.206 ms | **1.98x** |
| 5,000,000 | 1.629 ms | 0.607 ms | **2.68x** |

This comparison is deliberately GPU-vs-GPU, not GPU-vs-CPU - a GPU-vs-CPU number would
mostly just prove "GPUs are fast," not "this kernel is well-written." Isolating the
naive-GPU baseline against the custom kernel, both on the same device, is what actually
demonstrates the kernel implementation itself is better, not just the hardware.

**The output plot (`cuda_benchmark_fair.png`) is the single most interview-relevant
artifact in this repo** - save it, it's the concrete evidence behind the "I wrote a
custom CUDA kernel and benchmarked it" claim.

### Common setup gotchas (already fixed in this repo, documented for reference)

- **Colab's pre-installed numpy conflicts with `requirements.txt`'s pinned version.**
  Don't run `pip install -r requirements.txt` on Colab - instead install only the
  packages not already present (see `notebooks/run_all_colab.ipynb`, which does this
  correctly) and let pip resolve versions compatible with what Colab ships.
- **`ninja` isn't preinstalled on Colab** and is required to JIT-compile the CUDA
  extension: `pip install ninja` before calling `torch.utils.cpp_extension.load(...)`.
- **The custom kernel's `.so` build can silently fail inside a Jupyter cell** with a
  misleading `cannot open shared object file` error instead of the real compiler error.
  If this happens, run the build as a subprocess and capture stdout/stderr to a file
  to see the real message (see troubleshooting notes in the notebook).
- **Keep the comparison strictly on-GPU.** An earlier version of `cuda_compressor.py`
  round-tripped through CPU inside `compress()`/`decompress()` to simulate network
  transfer - this made the custom kernel look *slower* than the naive-GPU baseline,
  because the baseline never paid that same transfer cost. The current version stays
  on-GPU throughout, matching the baseline, so the benchmark isolates actual kernel
  compute rather than an artifact of uneven transfer overhead.

## Swapping in the real dataset

Everything is built against the `MedicalImageDataset` interface in
`data/dataset_loader.py`. To use a real dataset:

1. Write a new subclass of `MedicalImageDataset` that loads and returns
   `(train_dataset, test_dataset, DatasetInfo)`
2. Register it in `get_dataset()`
3. Nothing in `federated/`, `agent/`, or `api/` needs to change

## What this demonstrates, by role

- **ML Engineer**: distributed training coordination, non-IID data handling, FedAvg
  implementation, MLflow-style experiment tracking (add if desired), model serving
- **AI Engineer**: RAG (federated variant), LangGraph agent orchestration, grounding
  and conflict-surfacing as explicit design requirements, not afterthoughts
- **Software/Systems (NVIDIA-relevant)**: custom CUDA C++ kernel with `pybind11`
  bindings exposed to PyTorch, GPU memory management, honest three-way benchmarking
  methodology (CPU vs. naive GPU vs. custom kernel) rather than an unverified speedup claim

## Honest scope notes

- The CNN in `federated/model.py` is deliberately lightweight so federated rounds run
  quickly in a demo/Colab context. Swap in a real transfer-learning backbone
  (ResNet50/DenseNet201, as used in the earlier brain tumor classification project)
  once training at real scale on real data.
- The cross-hospital conflict detection in `rag/federated_retriever.py` uses a simple
  keyword-based heuristic, documented as such in the code. A production version would
  use a proper NLI (natural language inference) model - the seam to swap one in is
  intentionally clean.
- This project does not cover data-engineering-style skills (streaming pipelines,
  orchestration schedulers) - see the separate e-commerce pipeline project for that.
