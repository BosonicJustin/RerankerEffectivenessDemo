---
language:
- en
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:48000
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
datasets:
- sentence-transformers/gooaq
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- map
- mrr@10
- ndcg@10
model-index:
- name: CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2
  results:
  - task:
      type: cross-encoder-reranking
      name: Cross Encoder Reranking
    dataset:
      name: gooaq eval
      type: gooaq_eval
    metrics:
    - type: map
      value: 0.9491
      name: Map
    - type: mrr@10
      value: 0.9491
      name: Mrr@10
    - type: ndcg@10
      value: 0.9616
      name: Ndcg@10
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) on the [gooaq](https://huggingface.co/datasets/sentence-transformers/gooaq) dataset using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Training Dataset:**
    - [gooaq](https://huggingface.co/datasets/sentence-transformers/gooaq)
- **Language:** en
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['what is an algebraic expression ks2?', 'algebraic expression. • a mathematical phrase combining numbers and/or variables. • an expression does not contain equality or inequality signs. but may include other operators and grouping symbols.'],
    ['what is an algebraic expression ks2?', "As mentioned, corn isn't considered a paleo-friendly food. While it is pretty low in calories and is technically plant-based, it's a whole grain and the phytic acid found within corn can cause inflammation in the gut and mess with your blood sugar."],
    ['what is an algebraic expression ks2?', 'The Jacobs Engine Brake® (also known as the "Jake Brake®") is a diesel engine retarder that uses the engine to aid in slowing and controlling the vehicle. When activated, the engine brake alters the operation of the engine\'s exhaust valves so that the engine works as a power-absorbing air compressor.'],
    ['what is an algebraic expression ks2?', 'DMV opens earlier, closes earlier on Thursdays. The new hours are 7:45 a.m. to 4 p.m. on Tuesday, Wednesday and Friday. ... Up until this change, the DMV has opened at 8 a.m., Monday through Saturday and closed at 6:30 p.m. on Thursday. Full-service branches will remain closed on Mondays, as they are now.'],
    ['what is an algebraic expression ks2?', 'The religions in both Mesopotamia and ancient Egypt were polytheistic, meaning they believed in multiple gods and goddesses, and were based on nature. Both civilizations had gods of the sky, earth, freshwater, and the sun, as well as gods devoted to human emotions and the underworld.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'what is an algebraic expression ks2?',
    [
        'algebraic expression. • a mathematical phrase combining numbers and/or variables. • an expression does not contain equality or inequality signs. but may include other operators and grouping symbols.',
        "As mentioned, corn isn't considered a paleo-friendly food. While it is pretty low in calories and is technically plant-based, it's a whole grain and the phytic acid found within corn can cause inflammation in the gut and mess with your blood sugar.",
        'The Jacobs Engine Brake® (also known as the "Jake Brake®") is a diesel engine retarder that uses the engine to aid in slowing and controlling the vehicle. When activated, the engine brake alters the operation of the engine\'s exhaust valves so that the engine works as a power-absorbing air compressor.',
        'DMV opens earlier, closes earlier on Thursdays. The new hours are 7:45 a.m. to 4 p.m. on Tuesday, Wednesday and Friday. ... Up until this change, the DMV has opened at 8 a.m., Monday through Saturday and closed at 6:30 p.m. on Thursday. Full-service branches will remain closed on Mondays, as they are now.',
        'The religions in both Mesopotamia and ancient Egypt were polytheistic, meaning they believed in multiple gods and goddesses, and were based on nature. Both civilizations had gods of the sky, earth, freshwater, and the sun, as well as gods devoted to human emotions and the underworld.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Reranking

* Dataset: `gooaq_eval`
* Evaluated with [<code>CrossEncoderRerankingEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator) with these parameters:
  ```json
  {
      "at_k": 10,
      "always_rerank_positives": true
  }
  ```

| Metric      | Value                |
|:------------|:---------------------|
| map         | 0.9491 (+0.9491)     |
| mrr@10      | 0.9491 (+0.9491)     |
| **ndcg@10** | **0.9616 (+0.9616)** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### gooaq

* Dataset: [gooaq](https://huggingface.co/datasets/sentence-transformers/gooaq) at [b089f72](https://huggingface.co/datasets/sentence-transformers/gooaq/tree/b089f728748a068b7bc5234e5bcf5b25e3c8279c)
* Size: 48,000 training samples
* Columns: <code>text1</code>, <code>text2</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | text1                                                                                          | text2                                                                                           | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                         | string                                                                                          | float                                                          |
  | details | <ul><li>min: 20 characters</li><li>mean: 42.74 characters</li><li>max: 73 characters</li></ul> | <ul><li>min: 58 characters</li><li>mean: 252.8 characters</li><li>max: 370 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.17</li><li>max: 1.0</li></ul> |
* Samples:
  | text1                                             | text2                                                                                                                                                                                                                                                                                                                      | label            |
  |:--------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>what is an algebraic expression ks2?</code> | <code>algebraic expression. • a mathematical phrase combining numbers and/or variables. • an expression does not contain equality or inequality signs. but may include other operators and grouping symbols.</code>                                                                                                        | <code>1.0</code> |
  | <code>what is an algebraic expression ks2?</code> | <code>As mentioned, corn isn't considered a paleo-friendly food. While it is pretty low in calories and is technically plant-based, it's a whole grain and the phytic acid found within corn can cause inflammation in the gut and mess with your blood sugar.</code>                                                      | <code>0.0</code> |
  | <code>what is an algebraic expression ks2?</code> | <code>The Jacobs Engine Brake® (also known as the "Jake Brake®") is a diesel engine retarder that uses the engine to aid in slowing and controlling the vehicle. When activated, the engine brake alters the operation of the engine's exhaust valves so that the engine works as a power-absorbing air compressor.</code> | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 256
- `learning_rate`: 2e-05
- `warmup_steps`: 100
- `load_best_model_at_end`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 256
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 100
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: True
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss | gooaq_eval_ndcg@10 |
|:------:|:----:|:-------------:|:------------------:|
| 0.2660 | 50   | 0.113         | -                  |
| 0.5319 | 100  | 0.0266        | 0.9571 (+0.9571)   |
| 0.7979 | 150  | 0.0209        | -                  |
| 1.0638 | 200  | 0.0167        | 0.9651 (+0.9651)   |
| 1.3298 | 250  | 0.0128        | -                  |
| 1.5957 | 300  | 0.0146        | 0.9650 (+0.9650)   |
| 1.8617 | 350  | 0.0138        | -                  |
| 2.1277 | 400  | 0.0132        | 0.9653 (+0.9653)   |
| 2.3936 | 450  | 0.0102        | -                  |
| 2.6596 | 500  | 0.0082        | 0.9616 (+0.9616)   |
| 2.9255 | 550  | 0.0095        | -                  |


### Framework Versions
- Python: 3.10.0
- Sentence Transformers: 5.1.1
- Transformers: 4.57.0
- PyTorch: 2.8.0
- Accelerate: 1.10.1
- Datasets: 4.1.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->