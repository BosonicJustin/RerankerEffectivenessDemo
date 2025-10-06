"""
Evaluate and compare two encoder approaches on the GooAQ test set:
1. High-complexity encoder (all-MiniLM-L12-v2)
2. Low-complexity encoder (all-MiniLM-L6-v2)

Metrics: Recall@k, Precision@k, NDCG@k, MRR
"""

import numpy as np
import logging
from datetime import datetime
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, util
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'evaluation_encoders_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
HIGH_ENCODER_NAME = "sentence-transformers/all-MiniLM-L12-v2"
LOW_ENCODER_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"
TEST_DATASET_DIR = "./test_dataset"
TOP_K_VALUES = [1, 3, 5, 10, 20, 50, 100]
EVAL_SAMPLES = 500  # Number of test samples to evaluate

logger.info("=" * 80)
logger.info("ENCODER COMPARISON EVALUATION")
logger.info("=" * 80)

# Load test dataset
logger.info("\n[Step 1] Loading test dataset...")
test_dataset = load_from_disk(TEST_DATASET_DIR)
logger.info(f"Test dataset size: {len(test_dataset)}")

# Limit evaluation samples
eval_samples = min(EVAL_SAMPLES, len(test_dataset))
test_dataset_eval = test_dataset.select(range(eval_samples))
logger.info(f"Evaluating on {eval_samples} samples")

# Create corpus and queries
logger.info("\n[Step 2] Preparing corpus and queries...")
corpus = [example["answer"] for example in test_dataset]
queries = [example["question"] for example in test_dataset_eval]

logger.info(f"Corpus size: {len(corpus)}")
logger.info(f"Queries: {len(queries)}")

# Metrics calculation functions
def calculate_metrics(relevant_doc_idx, ranked_indices, k_values):
    """
    Calculate Precision, Recall, NDCG, and MRR for a single query.
    relevant_doc_idx: index of the relevant document
    ranked_indices: list of retrieved document indices in ranked order
    """
    metrics = {}

    # Find rank of relevant document (1-indexed)
    try:
        rank = ranked_indices.index(relevant_doc_idx) + 1
    except ValueError:
        rank = float('inf')  # Not found

    # MRR
    metrics['mrr'] = 1.0 / rank if rank != float('inf') else 0.0

    for k in k_values:
        top_k = ranked_indices[:k]

        # Precision@k and Recall@k
        relevant_in_top_k = 1 if relevant_doc_idx in top_k else 0
        metrics[f'precision@{k}'] = relevant_in_top_k / k
        metrics[f'recall@{k}'] = relevant_in_top_k  # Only 1 relevant doc per query

        # NDCG@k
        if relevant_doc_idx in top_k:
            position = top_k.index(relevant_doc_idx) + 1
            dcg = 1.0 / np.log2(position + 1)
            idcg = 1.0 / np.log2(2)  # Best case: relevant doc at position 1
            metrics[f'ndcg@{k}'] = dcg / idcg
        else:
            metrics[f'ndcg@{k}'] = 0.0

    return metrics

def aggregate_metrics(all_metrics):
    """Aggregate metrics across all queries."""
    aggregated = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        aggregated[metric_name] = np.mean(values)

    return aggregated

# Evaluate High-Complexity Encoder
logger.info("\n[Step 3] Evaluating High-Complexity Encoder...")
logger.info(f"Loading model: {HIGH_ENCODER_NAME}")
high_encoder = SentenceTransformer(HIGH_ENCODER_NAME)

logger.info("Encoding corpus with high-complexity encoder...")
start = time.time()
high_corpus_embeddings = high_encoder.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
high_encode_time = time.time() - start
logger.info(f"Corpus encoding time: {high_encode_time:.2f}s")

high_metrics = []
high_search_time = 0

logger.info("Evaluating queries...")
for i, query in enumerate(queries):
    if (i + 1) % 100 == 0:
        logger.info(f"  Processed {i + 1}/{len(queries)} queries...")

    relevant_doc_idx = i  # Query i corresponds to document i

    start = time.time()
    query_embedding = high_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, high_corpus_embeddings, top_k=max(TOP_K_VALUES))[0]
    high_search_time += time.time() - start

    ranked_indices = [hit['corpus_id'] for hit in hits]
    high_metrics.append(calculate_metrics(relevant_doc_idx, ranked_indices, TOP_K_VALUES))

high_results = aggregate_metrics(high_metrics)
logger.info(f"Average search time: {high_search_time / len(queries) * 1000:.2f}ms per query")

# Evaluate Low-Complexity Encoder
logger.info("\n[Step 4] Evaluating Low-Complexity Encoder...")
logger.info(f"Loading model: {LOW_ENCODER_NAME}")
# Use CPU for static-retrieval models (MPS not supported for embedding_bag operation)
low_encoder = SentenceTransformer(LOW_ENCODER_NAME, device="cpu")

logger.info("Encoding corpus with low-complexity encoder (using CPU)...")
start = time.time()
low_corpus_embeddings = low_encoder.encode(corpus, convert_to_tensor=False, show_progress_bar=True)
low_encode_time = time.time() - start
logger.info(f"Corpus encoding time: {low_encode_time:.2f}s")

low_metrics = []
low_search_time = 0

logger.info("Evaluating queries...")
for i, query in enumerate(queries):
    if (i + 1) % 100 == 0:
        logger.info(f"  Processed {i + 1}/{len(queries)} queries...")

    relevant_doc_idx = i

    start = time.time()
    query_embedding = low_encoder.encode(query, convert_to_tensor=False)
    hits = util.semantic_search(query_embedding, low_corpus_embeddings, top_k=max(TOP_K_VALUES))[0]
    low_search_time += time.time() - start

    ranked_indices = [hit['corpus_id'] for hit in hits]
    low_metrics.append(calculate_metrics(relevant_doc_idx, ranked_indices, TOP_K_VALUES))

low_results = aggregate_metrics(low_metrics)
logger.info(f"Average search time: {low_search_time / len(queries) * 1000:.2f}ms per query")

# Print Results
logger.info("\n" + "=" * 80)
logger.info("RESULTS")
logger.info("=" * 80)

logger.info("\n1. HIGH-COMPLEXITY ENCODER (all-MiniLM-L12-v2)")
logger.info("-" * 80)
logger.info(f"MRR: {high_results['mrr']:.4f}")
for k in TOP_K_VALUES:
    logger.info(f"Precision@{k}: {high_results[f'precision@{k}']:.4f}  |  Recall@{k}: {high_results[f'recall@{k}']:.4f}  |  NDCG@{k}: {high_results[f'ndcg@{k}']:.4f}")
logger.info(f"Corpus encoding time: {high_encode_time:.2f}s")
logger.info(f"Average search time: {high_search_time / len(queries) * 1000:.2f}ms per query")

logger.info("\n2. LOW-COMPLEXITY ENCODER (static-retrieval-mrl-en-v1)")
logger.info("-" * 80)
logger.info(f"MRR: {low_results['mrr']:.4f}")
for k in TOP_K_VALUES:
    logger.info(f"Precision@{k}: {low_results[f'precision@{k}']:.4f}  |  Recall@{k}: {low_results[f'recall@{k}']:.4f}  |  NDCG@{k}: {low_results[f'ndcg@{k}']:.4f}")
logger.info(f"Corpus encoding time: {low_encode_time:.2f}s")
logger.info(f"Average search time: {low_search_time / len(queries) * 1000:.2f}ms per query")

# Performance Comparison
logger.info("\n" + "=" * 80)
logger.info("PERFORMANCE COMPARISON")
logger.info("=" * 80)

logger.info("\nPerformance difference (High vs Low):")
logger.info(f"MRR: High={high_results['mrr']:.4f}, Low={low_results['mrr']:.4f}, Diff={((high_results['mrr'] - low_results['mrr']) / low_results['mrr'] * 100) if low_results['mrr'] > 0 else 0:+.2f}%")

for k in TOP_K_VALUES:
    high_ndcg = high_results[f'ndcg@{k}']
    low_ndcg = low_results[f'ndcg@{k}']
    diff_pct = ((high_ndcg - low_ndcg) / low_ndcg * 100) if low_ndcg > 0 else 0
    logger.info(f"NDCG@{k}: High={high_ndcg:.4f}, Low={low_ndcg:.4f}, Diff={diff_pct:+.2f}%")

logger.info(f"\nSpeed comparison:")
logger.info(f"Corpus encoding: High={high_encode_time:.2f}s, Low={low_encode_time:.2f}s, Speedup={high_encode_time/low_encode_time:.2f}x")
logger.info(f"Search time: High={high_search_time/len(queries)*1000:.2f}ms, Low={low_search_time/len(queries)*1000:.2f}ms")

logger.info("\n" + "=" * 80)
logger.info("EVALUATION COMPLETE!")
logger.info("=" * 80)
