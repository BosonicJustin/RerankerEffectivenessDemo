"""
Train a reranker model on GooAQ dataset using sentence-transformers.

This script:
1. Loads the GooAQ dataset
2. Splits into train/test sets
3. Creates training data with random negatives (simplified for Mac)
4. Trains a cross-encoder reranker
5. Saves the trained model weights and test dataset
"""

import os
import random
import logging
import argparse
from datetime import datetime
from datasets import load_dataset
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments
)
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

# Parse arguments
parser = argparse.ArgumentParser(description='Train a reranker model on GooAQ dataset')
parser.add_argument('--dry-run', action='store_true', help='Run with limited data for testing (1000 samples, 1 epoch)')
args = parser.parse_args()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
RERANKER_BASE_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
OUTPUT_DIR = "./trained_reranker"
TEST_DATASET_DIR = "./test_dataset"
TRAIN_TEST_SPLIT = 0.8
NUM_NEGATIVES = 5

# Adjust parameters based on dry-run mode
if args.dry_run:
    MAX_TRAIN_SAMPLES = 1000
    NUM_EPOCHS = 1
    logger.info("DRY-RUN MODE ENABLED")
else:
    MAX_TRAIN_SAMPLES = 10000
    NUM_EPOCHS = 3

BATCH_SIZE = 256
LEARNING_RATE = 2e-5
EVAL_STEPS = 100  # Evaluate every 100 steps

logger.info("=" * 80)
logger.info("RERANKER TRAINING DEMO")
logger.info("=" * 80)
logger.info(f"Configuration: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}, learning rate {LEARNING_RATE}")
logger.info(f"Max training samples: {MAX_TRAIN_SAMPLES}, Negatives per query: {NUM_NEGATIVES}")

# Step 1: Load dataset
logger.info("\n[Step 1] Loading GooAQ dataset...")
dataset = load_dataset("sentence-transformers/gooaq", split="train")
logger.info(f"Total samples: {len(dataset)}")

# Limit dataset size for faster training
if len(dataset) > MAX_TRAIN_SAMPLES:
    dataset = dataset.select(range(MAX_TRAIN_SAMPLES))
    logger.info(f"Limited to {MAX_TRAIN_SAMPLES} samples for faster training on Mac")

# Step 2: Train/test split
logger.info("\n[Step 2] Splitting into train/test sets...")
split_dataset = dataset.train_test_split(test_size=1-TRAIN_TEST_SPLIT, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]
logger.info(f"Train samples: {len(train_dataset)}")
logger.info(f"Test samples: {len(test_dataset)}")

# Save test set for later evaluation
test_dataset.save_to_disk(TEST_DATASET_DIR)
logger.info(f"Saved test dataset to {TEST_DATASET_DIR}")

# Step 3: Prepare training data with random negatives (lightweight approach)
logger.info("\n[Step 3] Preparing training data with random negatives...")

def create_training_pairs(examples):
    """
    Create training pairs: (question, answer) with label=1 and (question, negative_answer) with label=0.
    Uses random sampling for negatives - much faster than semantic similarity search.
    """
    questions = examples["question"]
    answers = examples["answer"]

    texts1 = []
    texts2 = []
    labels = []

    for i in range(len(questions)):
        question = questions[i]
        answer = answers[i]

        # Add positive pair
        texts1.append(question)
        texts2.append(answer)
        labels.append(1.0)

        # Add random negative pairs
        neg_indices = [j for j in range(len(answers)) if j != i]
        random.seed(42 + i)
        selected_neg_indices = random.sample(neg_indices, min(NUM_NEGATIVES, len(neg_indices)))

        for neg_idx in selected_neg_indices:
            texts1.append(question)
            texts2.append(answers[neg_idx])
            labels.append(0.0)

    return {"text1": texts1, "text2": texts2, "label": labels}

logger.info("Creating training pairs (this may take a moment)...")
train_data = train_dataset.map(
    create_training_pairs,
    batched=True,
    remove_columns=train_dataset.column_names
)

logger.info(f"Training pairs created: {len(train_data)}")
logger.info(f"Expected pairs per query: {1 + NUM_NEGATIVES} (1 positive + {NUM_NEGATIVES} negatives)")

# Step 4: Prepare evaluation data - evaluate on full corpus (realistic scenario)
logger.info("\n[Step 4] Preparing evaluation data (full corpus)...")

eval_samples = []

# Create full corpus from all test samples
full_corpus = [example["answer"] for example in test_dataset]

# Use subset of test dataset for faster evaluation during training
eval_size = min(100, len(test_dataset))  # Evaluate on 100 queries but with full corpus
logger.info(f"Using {eval_size} queries for evaluation (each against {len(full_corpus)} documents)")

for i in range(eval_size):
    example = test_dataset[i]
    question = example["question"]
    answer = example["answer"]

    # Use all OTHER answers as negative documents (realistic full-corpus search)
    documents = [full_corpus[j] for j in range(len(full_corpus)) if j != i]

    eval_samples.append({
        "query": question,
        "positive": [answer],
        "documents": documents
    })

logger.info(f"Created {eval_size} evaluation samples with full corpus ({len(full_corpus)} documents each)")

evaluator = CrossEncoderRerankingEvaluator(
    samples=eval_samples,
    name="gooaq_eval",
    batch_size=BATCH_SIZE,
    show_progress_bar=True
)

# Step 5: Load reranker model
logger.info(f"\n[Step 5] Loading reranker model: {RERANKER_BASE_NAME}")
model = CrossEncoder(RERANKER_BASE_NAME, num_labels=1)
logger.info("Model loaded successfully")

# Step 6: Configure training
logger.info("\n[Step 6] Configuring training...")
training_args = CrossEncoderTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    logging_steps=50,
    save_steps=1000,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_gooaq_eval_ndcg@10",
    greater_is_better=True,
)
logger.info(f"Training configuration: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}, eval every {EVAL_STEPS} steps")

# Step 7: Train the model
logger.info("\n[Step 7] Training reranker...")
logger.info("=" * 80)

trainer = CrossEncoderTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=None,
    evaluator=evaluator,
)

logger.info("Starting training...")
trainer.train()
logger.info("Training finished!")

# Step 8: Save the trained model (skip in dry-run mode)
if not args.dry_run:
    logger.info("\n[Step 8] Saving trained model...")
    model.save(OUTPUT_DIR)
    logger.info(f"Model saved to {OUTPUT_DIR}")
else:
    logger.info("\n[Step 8] Skipping model save (dry-run mode)")

logger.info("\n" + "=" * 80)
logger.info("TRAINING COMPLETE!")
logger.info("=" * 80)

if not args.dry_run:
    logger.info(f"\nTrained reranker saved to: {OUTPUT_DIR}")
    logger.info(f"Test dataset saved to: {TEST_DATASET_DIR}")
    logger.info("\nNext steps:")
    logger.info("1. Run evaluate_demo.py to compare performance")
else:
    logger.info(f"\nDry-run complete - model not saved")
    logger.info(f"Test dataset saved to: {TEST_DATASET_DIR}")
