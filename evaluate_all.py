"""
Comprehensive evaluation comparing three approaches on the GooAQ test set:
1. High-complexity encoder (all-MiniLM-L12-v2)
2. Low-complexity encoder (static-retrieval-mrl-en-v1)
3. Low-complexity encoder + trained reranker

All approaches are evaluated on the same task: finding the correct answer in the full corpus.

Metrics: Recall@k, Precision@k, NDCG@k, MRR
"""

import numpy as np
import logging
from datetime import datetime
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'evaluation_all_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
HIGH_ENCODER_NAME = "sentence-transformers/all-MiniLM-L12-v2"
LOW_ENCODER_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"
TRAINED_RERANKER_PATH = "./trained_reranker"
TEST_DATASET_DIR = "./test_dataset"
TOP_K_RETRIEVAL = 100  # Retrieve top-100 with encoder
RERANK_TOP_K = 10  # Rerank top-10 with cross-encoder
TOP_K_VALUES = [1, 3, 5, 10, 20, 50, 100]
EVAL_SAMPLES = 500  # Number of test samples to evaluate

logger.info("=" * 80)
logger.info("COMPREHENSIVE EVALUATION: Encoders vs Encoder + Reranker")
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

# Approach 1: High-Complexity Encoder
logger.info("\n[Step 3] Evaluating Approach 1: High-Complexity Encoder...")
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

    relevant_doc_idx = i

    start = time.time()
    query_embedding = high_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, high_corpus_embeddings, top_k=max(TOP_K_VALUES))[0]
    high_search_time += time.time() - start

    ranked_indices = [hit['corpus_id'] for hit in hits]
    high_metrics.append(calculate_metrics(relevant_doc_idx, ranked_indices, TOP_K_VALUES))

high_results = aggregate_metrics(high_metrics)
logger.info(f"Average search time: {high_search_time / len(queries) * 1000:.2f}ms per query")

# Approach 2: Low-Complexity Encoder
logger.info("\n[Step 4] Evaluating Approach 2: Low-Complexity Encoder...")
logger.info(f"Loading model: {LOW_ENCODER_NAME}")
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

# Approach 3: Low-Complexity Encoder + Reranker
logger.info("\n[Step 5] Evaluating Approach 3: Low-Complexity Encoder + Reranker...")
logger.info(f"Loading reranker: {TRAINED_RERANKER_PATH}")
reranker = CrossEncoder(TRAINED_RERANKER_PATH)

low_rerank_metrics = []
low_rerank_search_time = 0
rerank_time = 0

logger.info(f"Evaluating queries (retrieve top-{TOP_K_RETRIEVAL}, rerank top-{RERANK_TOP_K})...")
for i, query in enumerate(queries):
    if (i + 1) % 100 == 0:
        logger.info(f"  Processed {i + 1}/{len(queries)} queries...")

    relevant_doc_idx = i

    # Step 1: Retrieve with low encoder
    start = time.time()
    query_embedding = low_encoder.encode(query, convert_to_tensor=False)
    hits = util.semantic_search(query_embedding, low_corpus_embeddings, top_k=TOP_K_RETRIEVAL)[0]
    low_rerank_search_time += time.time() - start

    # Step 2: Rerank top-k candidates
    start = time.time()
    top_candidates = hits[:RERANK_TOP_K]
    rerank_pairs = [[query, corpus[hit['corpus_id']]] for hit in top_candidates]
    rerank_scores = reranker.predict(rerank_pairs)

    # Sort by reranker scores
    reranked_candidates = sorted(zip(top_candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    reranked_indices = [hit['corpus_id'] for hit, _ in reranked_candidates]

    # Append remaining candidates that weren't reranked
    remaining_indices = [hit['corpus_id'] for hit in hits[RERANK_TOP_K:]]
    final_ranked_indices = reranked_indices + remaining_indices
    rerank_time += time.time() - start

    low_rerank_metrics.append(calculate_metrics(relevant_doc_idx, final_ranked_indices, TOP_K_VALUES))

low_rerank_results = aggregate_metrics(low_rerank_metrics)
logger.info(f"Average search time: {low_rerank_search_time / len(queries) * 1000:.2f}ms per query")
logger.info(f"Average rerank time: {rerank_time / len(queries) * 1000:.2f}ms per query")
logger.info(f"Total time: {(low_rerank_search_time + rerank_time) / len(queries) * 1000:.2f}ms per query")

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

logger.info("\n3. LOW-COMPLEXITY ENCODER + TRAINED RERANKER")
logger.info("-" * 80)
logger.info(f"MRR: {low_rerank_results['mrr']:.4f}")
for k in TOP_K_VALUES:
    logger.info(f"Precision@{k}: {low_rerank_results[f'precision@{k}']:.4f}  |  Recall@{k}: {low_rerank_results[f'recall@{k}']:.4f}  |  NDCG@{k}: {low_rerank_results[f'ndcg@{k}']:.4f}")
logger.info(f"Corpus encoding time: {low_encode_time:.2f}s (same as approach 2)")
logger.info(f"Average search time: {low_rerank_search_time / len(queries) * 1000:.2f}ms per query")
logger.info(f"Average rerank time: {rerank_time / len(queries) * 1000:.2f}ms per query")
logger.info(f"Total time: {(low_rerank_search_time + rerank_time) / len(queries) * 1000:.2f}ms per query")

# Performance Comparison
logger.info("\n" + "=" * 80)
logger.info("PERFORMANCE COMPARISON")
logger.info("=" * 80)

logger.info("\nLow Encoder + Reranker vs Low Encoder alone:")
for metric in ['mrr'] + [f'ndcg@{k}' for k in [1, 3, 5, 10]]:
    improvement = ((low_rerank_results[metric] - low_results[metric]) / low_results[metric] * 100) if low_results[metric] > 0 else 0
    logger.info(f"  {metric.upper()}: {low_results[metric]:.4f} -> {low_rerank_results[metric]:.4f} ({improvement:+.2f}%)")

logger.info("\nLow Encoder + Reranker vs High Encoder:")
for metric in ['mrr'] + [f'ndcg@{k}' for k in [1, 3, 5, 10]]:
    diff = ((low_rerank_results[metric] - high_results[metric]) / high_results[metric] * 100) if high_results[metric] > 0 else 0
    logger.info(f"  {metric.upper()}: Low+Rerank={low_rerank_results[metric]:.4f}, High={high_results[metric]:.4f} ({diff:+.2f}%)")

logger.info("\n" + "=" * 80)
logger.info("EVALUATION COMPLETE!")
logger.info("=" * 80)
logger.info("\nKey Takeaway:")
logger.info("The reranker refines the low encoder's top candidates to improve ranking quality")
logger.info("while maintaining the speed advantage of the lightweight encoder.")
