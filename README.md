# Sparse Document Attention RAG (SDAG)

Sparse Document Attention RAG (SDAG) is a defense mechanism for Retrieval-Augmented Generation (RAG) against corpus knowledge poisoning attacks. SDAG isolates each retrieved document during generation by avoiding cross-attention between documents, reducing the model’s ability to mix malicious content with benign evidence.

> **Goal:** make RAG robust to poisoned or adversarially injected content in the retrieval corpus.

---

## Table of Contents
- [Overview](#overview)
- [Key Ideas](#key-ideas)
- [Project Layout](#project-layout)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Attack & Defense Modes](#attack--defense-modes)
- [Data](#data)
- [Results & Outputs](#results--outputs)
- [Tips](#tips)
- [Citation](#citation)

---

## Overview
The repository implements an experimental pipeline for evaluating poisoning attacks on RAG systems and comparing defenses. SDAG introduces **document isolation** in the generation step: instead of allowing the model to attend across all retrieved documents, attention is constrained per document (optionally with a small neighborhood), reducing the impact of malicious content.

The pipeline supports:
- **Corpus poisoning attacks** (malicious document injection, document corruption).
- **Multiple retrieval backends** (dense, sparse, hybrid).
- **Defensive strategies** (none, RAGDefender, Discern-and-Answer, SDAG isolation).

---

## Key Ideas
- **Sparse Document Attention:** enforce per-document attention during generation so that evidence is processed independently, minimizing cross-document contamination.
- **Attack simulation:** inject or corrupt documents before generation to emulate corpus knowledge poisoning.
- **Consistent evaluation:** measure attack success, false answer rates, and general QA performance across different defenses and retrieval settings.

---

## Project Layout
```
.
├── data/                      # Sample datasets and attack results
├── src/
│   └── pipeline/
│       ├── attack/            # Malicious document generation & corruption
│       ├── defenses/          # Defense implementations (RAGDefender, Discern, etc.)
│       ├── models/            # Datamodels and helpers
│       ├── retrieval/         # Dense / sparse / hybrid retrievers
│       ├── sparse_attention_RAG/  # SDAG generation logic
│       ├── utils/             # Utilities: metrics, parsing, prompt templates
│       ├── config.py          # Central config defaults
│       └── main.py            # End-to-end pipeline entrypoint
└── README.md
```

---

## Quickstart
> These steps outline the main pipeline. Adjust for your environment and models.

1. **Create a virtual environment** (recommended).
2. **Install dependencies** (examples):
   ```bash
   pip install torch transformers sentence-transformers faiss-cpu pyserini tqdm
   ```
3. **Run the pipeline** with default settings:
   ```bash
   python -m src.pipeline.main
   ```
4. **Run with a JSON config override**:
   ```bash
   python -m src.pipeline.main path/to/config.json
   ```

---

## Configuration
Configuration is centralized in `src/pipeline/config.py`. You can override any field at runtime by passing a JSON file to `main.py`. Example override:

```json
{
  "TOP_K": [5, 10],
  "ADD_ATTACK_IN_RANK": [1, 3],
  "RETRIEVER_BACKEND": "dense",
  "ATTACK_VARIANT": "malicious_doc",
  "DEFENSE_BACKEND": "none",
  "DOC_NEIGHBORS_K": 0
}
```

Key settings to explore:
- `RETRIEVER_BACKEND`: `dense`, `sparse`, or `sparse_and_dense`.
- `ATTACK_VARIANT`: `malicious_doc` or `doc_corruption`.
- `DEFENSE_BACKEND`: `none`, `ragdefender`, or `discern_and_answer`.
- `DOC_NEIGHBORS_K`: number of neighbor documents allowed to attend during SDAG generation (0 = strict isolation).

---

## Attack & Defense Modes
- **Malicious document injection**: generate adversarial documents and insert them into the ranked list at specified positions.
- **Document corruption**: replace ground-truth spans inside retrieved documents with false content.
- **No defense**: baseline generation on the retrieved list.
- **RAGDefender**: model-based filtering of malicious documents.
- **Discern-and-Answer**: external classifier to flag poisoned content.
- **SDAG**: document isolation during generation via sparse attention.

---

## Data
The `data/` directory includes:
- Pre-sampled query sets (Natural Questions, HotpotQA, TriviaQA).
- Example attack results (CSV) from prior runs.

You can point `CSV_INPUT_PATH` to your own query/answer files if you prefer a custom dataset.

---

## Results & Outputs
Each run outputs:
- A CSV file containing per-query results.
- A JSON file containing metrics and a configuration snapshot.

Output naming is derived from:
- `OUTPUT_CSV_BASE`
- `TOP_K`
- `ADD_ATTACK_IN_RANK`

---

## Tips
- Start with `RETRIEVER_BACKEND="dense"` and a small `SAMPLE_SIZE` to validate the pipeline.
- Use `DOC_NEIGHBORS_K=0` for the strongest isolation baseline.
- For reproducibility, keep `SEED` fixed and record config JSONs per run.

---

## Citation
If you use this repository in academic work, please cite the project:

```bibtex
@misc{sdar2024,
  title={Sparse Document Attention RAG (SDAG)},
  author={Your Name or Lab},
  year={2024},
  note={Defense against corpus poisoning attacks in RAG systems}
}
```
