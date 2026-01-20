# Sparse RAG under Corpus Knowledge Poisoning Attacks (SDAG)

This repository implements an experimental pipeline for **Retrieval-Augmented Generation (RAG)** under **Corpus Knowledge Poisoning** and evaluates **Sparse Document Attention RAG (SDAG)** against the standard causal-attention baseline (**CARG**) and additional defense baselines (e.g., RAGDefender, Discern-and-Answer), including **SDAG + baseline integration**.

SDAG is a **block-sparse attention mechanism** that **disallows cross-attention between retrieved documents**, mitigating harmful cross-document interactions when the retrieved set contains conflicting or adversarial evidence. SDAG is applied as a **minimal inference-time change** (attention mask only), with **no fine-tuning** and no architectural changes required.

---

## Paper framing: RAG, threat model, and what this repo evaluates

### RAG formulation
Given a question \(q\), a retriever \(R\) retrieves a set of \(k\) documents  
\(D := R(q, C) = \{d_1, \dots, d_k\}\) from corpus \(C\).
A generator \(G\) then produces an answer
\(y = G(p, q, D)\), where \(p\) denotes instructions and optional context.

After tokenization, each retrieved document corresponds to a token **block** \(B_i\).
The instruction + query block is \(B_T\), and optional generator context is \(B_C\).

### Threat model: Corpus Knowledge Poisoning
The attacker injects adversarial documents crafted to steer the generator toward a false target answer.

Two evaluation settings are supported:
- **In-corpus**: adversarial documents are injected into the corpus and may or may not be retrieved.
- **In-set**: adversarial documents are assumed to appear in the retrieved set, isolating generator-side effects.

### Adversarial document selection
Adversarial documents can be selected using:
- **Random**
- **Near** (close to benign-doc centroid in embedding space)
- **Far** (far from benign-doc centroid)

Single-adversarial-document attacks are the primary setting, matching the paper.

---

## SDAG vs CARG (generation behavior)

- **CARG (baseline)**: standard causal attention over the entire prompt, allowing unrestricted cross-document interactions.
- **SDAG**: enforces document isolation by masking attention across different document blocks \(B_i\), while preserving causal attention within \(B_T\) and \(B_C\).

This directly targets cross-document contamination and knowledge conflicts during generation.

---

## What this repository provides

The pipeline supports:
1. Query loading / sampling (Natural Questions, HotpotQA, TriviaQA, or CSV).
2. Retrieval using **dense**, **sparse (BM25)**, or **hybrid** retrievers.
3. **Corpus poisoning attacks**:
   - malicious document injection
   - document corruption variants
4. **Defenses**:
   - none (CARG baseline)
   - SDAG (document-isolated generation)
   - RAGDefender
   - Discern-and-Answer
   - integrations (e.g., SDAG + baseline defenses)
5. Evaluation using **Accuracy (ACC)** and **Attack Success Rate (ASR)**, plus detailed retrieval–ground-truth analyses.

---

## Repository layout

```
.
├── data/                         # Optional datasets, sampled queries, outputs
├── src/
│   └── pipeline/
│       ├── attack/               # Adversarial document generation & selection (Random/Near/Far)
│       ├── defenses/             # Defense baselines and integrations
│       ├── retrieval/            # Dense / sparse / hybrid retrieval
│       ├── sdag/                 # Sparse Document Attention (block-sparse masks)
│       ├── metrics/              # ACC/ASR and detailed retrieval-ground-truth statistics
│       ├── prompts/              # Prompt templates
│       ├── config.py             # Central configuration (paper-aligned knobs)
│       └── main.py               # End-to-end pipeline entrypoint
└── README.md
```

---

## Installation

Example dependencies (adjust as needed):

```bash
pip install torch transformers sentence-transformers faiss-cpu pyserini tqdm numpy
```

---

## Quickstart

Run with default configuration:

```bash
python -m src.pipeline.main
```

Run with a JSON config override:

```bash
python -m src.pipeline.main path/to/config.json
```

---

## Configuration (paper-aligned)

Important parameters:
- `TOP_K`: number of retrieved documents \(k\).
- `RETRIEVER_BACKEND`: dense / sparse / hybrid.
- `ATTACK_VARIANT`: malicious document injection or corruption.
- `MALICIOUS_DOC_SELECTION_STRATEGY`: random / near / far.
- `DEFENSE_BACKEND`: none (CARG), sdag, ragdefender, discern_and_answer.
- `DOC_NEIGHBORS_K`: number of neighboring documents allowed to attend (0 = strict SDAG).

Example override:

```json
{
  "TOP_K": [5, 10],
  "ADD_ATTACK_IN_RANK": [1, 3],
  "RETRIEVER_BACKEND": "dense",
  "ATTACK_VARIANT": "malicious_doc",
  "DEFENSE_BACKEND": "sdag",
  "DOC_NEIGHBORS_K": 0
}
```

---

## Outputs

For each `(top_k, attacker_pos)` configuration, the pipeline produces:
- **CSV**: per-query outputs (answers, retrieved docs, match flags).
- **JSON**: aggregated metrics including:
  - ACC / ASR statistics,
  - retrieval–ground-truth bucket analyses,
  - iso vs no-iso overlap behavior,
  - a full **run configuration snapshot** for reproducibility.

Output directories are created automatically.

---

## Citation

If you use this repository in academic work, please cite:

```bibtex
@misc{sdag_sparse_rag_poisoning,
  title={Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention},
  note={Under review},
  year={2025}
}
```
