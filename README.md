# Reranker Effectiveness Demo

This project demonstrates how a trained reranker can improve search quality when combined with a lightweight encoder, potentially matching or exceeding the performance of a high-complexity encoder while maintaining speed advantages.

## Overview

The demo compares three approaches for semantic search:
1. **High-complexity encoder** (baseline quality)
2. **Low-complexity encoder** (fast but lower quality)
3. **Low-complexity encoder + trained reranker** (fast retrieval + refined ranking)

## Components

### Models Used

- **High Encoder**: `sentence-transformers/all-MiniLM-L12-v2` (33M params)
- **Low Encoder**: `sentence-transformers/static-retrieval-mrl-en-v1` (lightweight)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L6-v2` (fine-tuned on GooAQ dataset)

### Dataset

- **Training/Evaluation**: GooAQ dataset (`sentence-transformers/gooaq`)
- **Split**: 80% train / 20% test
- **Training samples**: 10,000 (with 3 epochs)
- **Test samples**: 2,000

## Project Structure

```
RerankerEffectivenessDemo/
├── train_reranker.py          # Train the reranker model
├── evaluate_encoders.py       # Compare high vs low encoders
├── evaluate_all.py            # Compare all three approaches
├── demo_app.py                # Interactive Streamlit web demo
├── trained_reranker/          # Saved reranker model (Git LFS)
├── test_dataset/              # Held-out test set
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- Virtual environment recommended

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install sentence-transformers datasets torch streamlit
```

## Usage

### 1. Train the Reranker

Train a reranker on the GooAQ dataset:

```bash
# Full training (10K samples, 3 epochs)
python train_reranker.py

# Quick test run (1K samples, 1 epoch, no save)
python train_reranker.py --dry-run
```

**Training Configuration:**
- Batch size: 256
- Epochs: 3 (1 for dry-run)
- Learning rate: 2e-5
- Evaluation: Every 100 steps on 100 queries
- Loss: Binary Cross-Entropy

### 2. Evaluate Encoders

Compare high-complexity vs low-complexity encoders:

```bash
python evaluate_encoders.py
```

This evaluates both encoders on the test set and reports:
- MRR, Precision@k, Recall@k, NDCG@k
- Encoding and search times
- Performance differences

### 3. Comprehensive Evaluation

Compare all three approaches:

```bash
python evaluate_all.py
```

This shows:
- How the reranker improves the low encoder's results
- Whether low encoder + reranker matches high encoder quality
- Speed vs quality trade-offs

### 4. Interactive Web Demo

Launch the interactive Streamlit demo:

```bash
streamlit run demo_app.py
```

**Demo Features:**
- Adjustable k (number of results: 1-20)
- Search bar with example queries
- Side-by-side comparison:
  - Left: Low encoder retrieves top-k directly
  - Right: Low encoder retrieves top-100, reranker selects top-k
- Visual indicators for results promoted by reranker
- Timing and overlap metrics

## Key Results

Based on evaluation on 200 test queries:

### Encoder Comparison (without reranker)

| Metric | High Encoder | Low Encoder | Difference |
|--------|-------------|-------------|------------|
| NDCG@10 | 0.9926 | 0.9816 | +1.12% |
| MRR | 0.9900 | 0.9772 | +1.31% |
| Speed (encoding) | 0.54s | 0.02s | **30x faster** |
| Speed (search/query) | 10.38ms | 0.37ms | **28x faster** |

### With Reranker

The reranker refines the low encoder's top candidates to close the quality gap while maintaining the encoding speed advantage. The reranking step adds minimal overhead compared to the initial retrieval.

## How It Works

### Training Pipeline

1. **Load GooAQ dataset** - Question-answer pairs
2. **Train/test split** - 80/20, saves test set for evaluation
3. **Create training pairs** - Positive (Q, correct A) and negative (Q, random A) pairs
4. **Train cross-encoder** - Binary classification: relevant vs irrelevant
5. **Save model** - Stored in `trained_reranker/`

### Retrieval Pipeline

**Encoder-only approach:**
1. Encode corpus once (offline)
2. Encode query
3. Find top-k most similar documents by cosine similarity

**Encoder + Reranker approach:**
1. Encode corpus once (offline)
2. Encode query
3. Retrieve top-100 candidates with encoder (fast)
4. Score all 100 candidates with reranker (slow but accurate)
5. Return top-k re-ranked results

## Customization

### Adjust Training Parameters

Edit configuration in `train_reranker.py`:

```python
MAX_TRAIN_SAMPLES = 10000  # Number of training samples
NUM_EPOCHS = 3             # Training epochs
BATCH_SIZE = 256           # Batch size
NUM_NEGATIVES = 5          # Negatives per positive
EVAL_STEPS = 100           # Evaluate every N steps
```

### Change Models

Update model names in scripts:

```python
HIGH_ENCODER_NAME = "sentence-transformers/all-MiniLM-L12-v2"
LOW_ENCODER_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"
RERANKER_BASE_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
```

## Git LFS

Large model files are stored using Git LFS:

```bash
# Install Git LFS
git lfs install

# Files tracked by LFS:
# - *.safetensors
# - *.pt, *.pth, *.bin
# - model config files
```

## Troubleshooting

### Network Issues

If you can't download models from Hugging Face:
- Models cache to `~/.cache/huggingface/`
- Download models separately and specify local paths
- Use offline mode: `export HF_HUB_OFFLINE=1`

### Memory Issues

If training fails with OOM:
- Reduce `BATCH_SIZE` in training script
- Reduce `MAX_TRAIN_SAMPLES`
- Use `--dry-run` for testing

### MPS/GPU Issues

The static-retrieval model doesn't support MPS (Apple Silicon):
- Automatically falls back to CPU
- This is expected and handled in the code

## References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Training Rerankers Tutorial](https://huggingface.co/blog/train-reranker)
- [GooAQ Dataset](https://huggingface.co/datasets/sentence-transformers/gooaq)
- [Cross-Encoder Models](https://www.sbert.net/examples/applications/cross-encoder/README.html)

## License

This project is for educational purposes. Model licenses follow their respective repositories.

## Citation

If you use this demo in your research, please cite the underlying libraries:

```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on EMNLP",
    year = "2019",
    publisher = "Association for Computational Linguistics",
}
```
