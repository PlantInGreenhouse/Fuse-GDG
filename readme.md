# Fuse-GDG

> **Graph + Domain + Global (GDG) fusion for link prediction**

This repository contains the minimal, reproducible code used in our paper to evaluate link prediction on **Hetionet** and **SuppKG** using a lightweight fusion model that combines:

- **Text embeddings** (e.g., PubMedBERT sentence/entity representations)
- **Graph/Knowledge embeddings** (e.g., Poincaré/MuRP)
- **Global/context embeddings** (Clustered by hierarchical leigen clustering.)

The implementation is intentionally compact for ease of reproduction and ablation.

---

## Highlights

- **Single‑command training & evaluation**
- Works with **precomputed embeddings** (no heavy model training here)
- Supports **MRR, MR, Hits\@K** with filtered evaluation
- **CUDA 12.4** ready (PyTorch & DGL pinned)

---

## Repository Structure

```
FUSE_GDG/
├─ hetionet/
│  ├─ entity_with_definition/
│  │  ├─ train_aligned.json
│  │  ├─ valid_aligned.json
│  │  └─ test_aligned.json
│  ├─ entity2index.pkl
│  ├─ index2entity.pkl
│  ├─ index2relation.pkl
│  ├─ relation2index.pkl
│  ├─ pubmedbert_embeddings_768.npy      # Files downloaded from Google Drive
│  ├─ poincare_embeddings.npy
│  ├─ global_embeddings.npy              # Files downloaded from Google Drive
│  └─ train.tsv | valid.tsv | test.tsv   # Triples: head \t relation \t tail
├─ suppkg/
│  ├─ (same layout as hetionet/)
├─ data_loader.py
├─ model.py
├─ myutils.py
├─ main.py
└─ requirements.txt
```

> **Note**: The `entity_with_definition/` The folder is optional during runtime and you can see how the names, types, and definitions of the entities are configured.

---

## Download Precomputed Embeddings

Due to file size limitations, precomputed embedding files (`.npy` and `.pth`) are not included directly in this repository. You can download them from Google Drive:

🔗 [Download Embeddings from Google Drive](https://drive.google.com/drive/folders/1cZyn_SXkwAWW397MixsXUhDPK7H0scwz?usp=drive_link)

## Pretrained Model Checkpoints

We also provide trained checkpoints for direct evaluation or warm start.

🔗 [Download Model Checkpoints (Google Drive)](https://drive.google.com/drive/folders/1cZyn_SXkwAWW397MixsXUhDPK7H0scwz?usp=drive_link)

## Environment

- **Python**: 3.10
- **CUDA**: 12.4
- **Key packages** (pinned in `requirements.txt`):
  - `torch==2.4.0+cu124` (pypi)
  - `dgl==2.4.0+cu124`  (pypi)

### Quick setup (conda + pip)

```bash
conda create -n fuse_gdg python=3.10 -y
conda activate fuse_gdg
# If you already have CUDA 12.4 drivers/toolkit
pip install -r requirements.txt
```

> If you use a different CUDA version, install the **matching** wheels for both PyTorch and DGL.

---

## Data Format

- `train.tsv`, `valid.tsv`, `test.tsv` contain triples in **TSV**: `head\trelation\ttail` (no header).
- `entity2index.pkl`, `relation2index.pkl` map **string IDs → integer indices**.
- Embedding `.npy` files are arrays whose **row order** matches the integer indices.

### Required files per dataset

- `pubmedbert_embeddings_768.npy`  → `--text_embedding_file`
- `poincare_embeddings.npy`        → `--graph_embedding_file`
- `global_embeddings.npy`          → `--global_embedding_file`

---

## Quickstart

Single command to run training + periodic evaluation:

```bash
python main.py \
  --data hetionet \
  --text_embedding_file pubmedbert_embeddings_768.npy \
  --knowledge_embedding_file poincare_embeddings.npy \
  --global_embedding_file global_embeddings.npy \
  --w_text 0.3 --w_domain 0.5 --w_global 0.2 \
  --num_hidden_layers 2 \
  --iterations 40000 \
  --evaluate_every 1000 \
  --neg_sample_size_eval 100 \
  --model_state_file hetionet_model_state352.pth
```

Switching to **SuppKG** only changes the `--data` folder and output filename:

```bash
python main.py --data suppkg ... --model_state_file suppkg_model_stateXXXX.pth
```

---


### Key Arguments

- `--w_text`, `--w_domain`, `--w_global`: fusion weights (must sum to 1.0 is **not** required, but recommended).
- `--iterations`: total training steps.
- `--evaluate_every`: evaluation frequency (steps). Set high to evaluate only at the end.
- `--neg_sample_size_eval`: negatives per query during evaluation.
- `--model_state_file`: output checkpoint filename.

---

## Metrics

- **MRR**, **MR**, **Hits\@1/3/10** (filtered protocol).
- Logs are printed to stdout; checkpoints are saved to `--model_state_file`.

---

## Reproducibility Tips

- Fix seeds if needed (see `myutils.py`).
- Keep `entity2index.pkl` and embedding row orders in sync.
- Use the same CUDA build for **both** PyTorch and DGL (see Troubleshooting).

---

## Troubleshooting

**DGL/PyTorch CUDA mismatch**

- Symptom: import errors or runtime CUDA failures
- Fix: install wheels that target the **same** CUDA (e.g., `+cu124` for both)

**OOM during eval**

- Reduce `--neg_sample_size_eval`
- Evaluate less frequently via larger `--evaluate_every`

---

## Citation

```bibtex
@misc{fuse_gdg_2025,
  title        = {Fuse-GDG: Leveraging graph structural, domain knowledge, global context to enhance GNN-based link prediction on Biomedical knowledge graphs},
  author       = {DaeHo Kim, TaeHeon Seong, SoYeop Yoo and OkRan Jeong},
  year         = {2025},
  url          = {https://github.com/PlantInGreenhouse/Fuse-GDG}
}
```

---

## License


---

## Acknowledgments

- 

