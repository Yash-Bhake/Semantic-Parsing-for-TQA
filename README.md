# Text-to-SQL Generator with T5 and QLoRA

A lightweight semantic parsing system that converts natural language questions into SQL queries using T5-Large with QLoRA fine-tuning on the Spider dataset.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Known Issues](#known-issues)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [References](#references)

## Overview

This project implements a complete Text-to-SQL pipeline that:
- Takes natural language questions and database schemas as input
- Generates syntactically correct SQL queries
- Uses parameter-efficient fine-tuning (QLoRA) for training
- Includes schema retrieval, entity linking, and constrained decoding

**Key Stats:**
- Base Model: T5-Large (770M parameters)
- Trainable Parameters: ~18.8M (2.5% of total)
- Training Data: Spider dataset (5,000 samples + augmentation)
- Memory: ~4GB GPU RAM
- Inference: ~100-200ms per query

## Features

### Core Components

1. **Schema Retriever**
   - Uses sentence-transformers for semantic similarity
   - Retrieves top-K relevant tables/columns
   - Reduces context for efficient processing

2. **Entity Linker**
   - Fuzzy string matching for value normalization
   - Handles abbreviations (NYC → New York City)
   - Links question entities to database values

3. **Data Augmentation**
   - Synonym replacement
   - Implicit operation expansion (oldest → MAX)
   - Paraphrase generation

4. **PICARD-Style Decoder**
   - Constrained SQL generation
   - Syntax validation during decoding
   - Prevents invalid token sequences

5. **Comprehensive Evaluation**
   - Exact match accuracy
   - Token-level F1 score
   - Error categorization (joins, aggregations, etc.)
   - Robustness testing

6. **Interactive GUI**
   - Gradio interface for live inference
   - Example queries included
   - Schema visualization

## Architecture

```
Input Question + Schema
        ↓
[Schema Retriever] ← Sentence Transformers
        ↓
[Entity Linker] ← Fuzzy Matching
        ↓
[T5-Large Encoder] ← QLoRA Fine-tuned
        ↓
[PICARD Decoder] ← Constrained Generation
        ↓
SQL Query Output
```

### Model Architecture

- **Base**: T5-Large (encoder-decoder transformer)
- **Fine-tuning**: QLoRA with LoRA adapters
  - LoRA rank (r): 32
  - LoRA alpha: 64
  - Target modules: q, k, v, o (attention layers)
  - Dropout: 0.1
- **Precision**: FP16 mixed precision training

## 🔧 Installation

### Requirements

```bash
# Python 3.8+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip install transformers datasets accelerate peft bitsandbytes
pip install sentence-transformers faiss-cpu rapidfuzz
pip install sqlparse evaluate sacrebleu rouge_score
pip install gradio  # For GUI
```

### Google Colab Setup

```python
!pip install -q transformers datasets accelerate peft bitsandbytes
!pip install -q sqlparse faiss-cpu rapidfuzz sentence-transformers evaluate
!pip install -q sacrebleu rouge_score gradio

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
```

## 🚀 Usage

### Quick Start

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# Load model
tokenizer = AutoTokenizer.from_pretrained('./sql_model_final')
model = AutoModelForSeq2SeqLM.from_pretrained('./sql_model_final')
schema_retriever = SchemaRetriever()

# Define schema
schema = {
    'table_names_original': ['students', 'courses'],
    'column_names_original': [
        (-1, '*'),
        (0, 'id'), (0, 'name'), (0, 'age'),
        (1, 'id'), (1, 'title')
    ],
    'column_types': ['text', 'number', 'text', 'number', 'number', 'text']
}

# Generate SQL
question = "What are the names of students older than 20?"
reduced_schema = schema_retriever.retrieve_top_k(question, schema, k=10)
input_text = f"question: {question} schema: {reduced_schema}"

outputs = model.generate(**tokenizer(input_text, return_tensors='pt'))
sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated SQL: {sql}")
```

### Using the GUI

```python
# Launch Gradio interface
demo.launch(share=True)
```

The GUI provides:
- Natural language question input
- Schema definition interface
- Real-time SQL generation
- Example queries

## Project Structure

```
CS772_Text_to_SQL/
│
├── sql_model_final/          # Saved model checkpoint
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
│
├── evaluation_results.json   # Evaluation metrics
├── analysis_report.md        # Detailed analysis
│
├── modules/
│   ├── schema_retriever.py   # Schema retrieval
│   ├── entity_linker.py      # Entity linking
│   ├── augmenter.py          # Data augmentation
│   ├── picard_decoder.py     # Constrained decoding
│   └── evaluator.py          # Evaluation metrics
│
└── notebooks/
    └── CS772_SP_for_TQA1.ipynb  # Main notebook
```

## Training

### Training Configuration

```python
training_args = TrainingArguments(
    output_dir='./sql_model',
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy='steps',
    eval_steps=200,
    save_steps=200,
)
```

### Training Process

1. **Data Loading**: Spider dataset (train + validation splits)
2. **Augmentation**: Generates ~1.8x more samples via:
   - Synonym replacement
   - Implicit operation expansion
3. **Fine-tuning**: 5 epochs with cosine schedule
4. **Evaluation**: Every 200 steps on validation set

### Training Time

- **Full training**: ~5-6 hours on T4 GPU (5,000 samples)
- **Per epoch**: ~1 hour
- **Memory usage**: ~3.5GB GPU RAM

## Evaluation

### Metrics

1. **Exact Match (EM)**: Binary correctness after normalization
2. **Token F1**: Token-level precision/recall
3. **Execution Accuracy**: Query result matching (optional)

### Error Categories

The system analyzes failures across:
- Schema linking errors
- Join operation errors
- Aggregation function errors
- Filter (WHERE) clause errors
- GROUP BY errors
- ORDER BY errors
- Nested query errors

### Running Evaluation

```python
# Evaluate on test set
predictions, references, results = evaluate_model(
    model, tokenizer, spider_dev, schema_retriever, num_samples=100
)

# Generate failure analysis
analyzer = FailureAnalyzer()
failure_report = analyzer.generate_report(questions, predictions, references)
```

## Known Issues

### Critical Issues in Current Code

1. **Model Name Variable Missing**
   ```python
   # Line in QLoRA Setup cell is incomplete:
   model = AutoModelForSeq2SeqLM.from_pretrained(
       model_name,  # ❌ 'model_name' is not defined!
       ...
   )
   
   # Fix: Add before model loading
   model_name = 't5-large'
   ```

2. **Training Interrupted**
   - Training stopped at step 129/2880 (4.5% complete)
   - Model never reached convergence
   - This explains 0% exact match accuracy

3. **Schema Formatting**
   - Some examples may have misaligned table-column indices
   - Could cause schema retrieval errors

4. **Memory Management**
   - Batch size of 1 with 16 gradient accumulation steps
   - May cause slow training on free Colab tier
   - Consider using Colab Pro for full training

### Recommendations to Fix

```python
# 1. Define model name
model_name = 't5-large'  # or 't5-base' for faster training

# 2. Reduce training samples for faster testing
spider_train = spider_train.select(range(1000))  # Start smaller

# 3. Add checkpointing
training_args.save_total_limit = 5
training_args.load_best_model_at_end = True

# 4. Monitor training
training_args.logging_steps = 10
training_args.report_to = 'tensorboard'
```

## Results

### Current Performance (Incomplete Training)

```
Exact Match Accuracy: 0.00%
Token F1 Score: 0.1903
```

**Note**: These results are from an interrupted training run. Complete training should achieve:
- Expected EM: 30-40% (Spider dev set)
- Expected F1: 0.60-0.70

### Error Distribution (100 test samples)

| Error Type | Count | Percentage |
|------------|-------|------------|
| Filter errors | 51 | 21.8% |
| Join errors | 47 | 20.1% |
| Aggregation errors | 47 | 20.1% |
| Nested query errors | 43 | 18.4% |
| GROUP BY errors | 26 | 11.1% |
| ORDER BY errors | 18 | 7.7% |
| Schema linking | 2 | 0.9% |

## Future Improvements

### Short Term

1. **Complete Training**
   - Run full 5 epochs
   - Monitor validation loss
   - Save best checkpoint

2. **Model Enhancements**
   - Increase LoRA rank to 64
   - Add prefix tuning
   - Experiment with T5-3B

3. **Data Augmentation**
   - Add back-translation
   - Generate hard negatives
   - Include domain-specific examples

### Long Term

1. **Multi-Stage Reasoning**
   - Implement DIN-SQL style decomposition
   - Add intermediate representation (IR)
   - Chain-of-thought prompting

2. **Execution-Guided Decoding**
   - Verify queries against actual database
   - Retry on execution errors
   - Self-correction mechanism

3. **Advanced Techniques**
   - Schema value caching
   - Cross-database generalization
   - Few-shot learning support

## References

### Papers

1. **T5**: Raffel et al. (2020) - "Exploring the Limits of Transfer Learning"
2. **Spider**: Yu et al. (2018) - "Spider: A Large-Scale Human-Labeled Dataset"
3. **LoRA**: Hu et al. (2021) - "Low-Rank Adaptation of Large Language Models"
4. **PICARD**: Scholak et al. (2021) - "PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding"

### Datasets

- **Spider**: https://yale-lily.github.io/spider
- **WikiSQL**: https://github.com/salesforce/WikiSQL

### Libraries

- **Transformers**: https://huggingface.co/transformers
- **PEFT**: https://github.com/huggingface/peft
- **Sentence-Transformers**: https://www.sbert.net

## Citation

If you use this code, please cite:

```bibtex
@misc{text2sql_t5_qlora,
  title={Text-to-SQL Generation with T5 and QLoRA},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/text-to-sql}}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue or contact [yashbhake1@gmail.com]

---

**Note**: This is a research/educational project.