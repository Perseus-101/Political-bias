# Article Bias Prediction

This repository is a fork of [ramybaly/Article-Bias-Prediction](https://github.com/ramybaly/Article-Bias-Prediction). It enhances the original work by implementing and evaluating Triplet Loss Pre-training (TLP) for political bias detection in news articles. The goal is to improve the model's ability to capture nuanced political biases through better representation learning using pre-trained models from HuggingFace.

## Table of Contents
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
- [Models and Approaches](#models-and-approaches)
  - [Baseline Model](#baseline-model)
  - [Triplet Loss Pre-training (TLP)](#triplet-loss-pre-training-tlp)
  - [LLM Bias Detection](#llm-bias-detection)
- [Running Experiments](#running-experiments)
- [Parameter Guidelines](#parameter-guidelines)
- [Performance Comparison](#performance-comparison)
- [Citation](#citation)

## Dataset

The dataset consists of 37,554 news articles crawled from www.allsides.com, available in the `./data` folder along with different evaluation splits.

Each article is stored as a JSON object in the `./data/jsons` directory with the following fields:

1. **ID**: Alphanumeric identifier
2. **topic**: Topic discussed in the article
3. **source**: Name of the article's source (e.g., New York Times)
4. **source_url**: URL to the source's homepage (e.g., www.nytimes.com)
5. **url**: Link to the actual article
6. **date**: Publication date
7. **authors**: Comma-separated list of authors
8. **title**: Article title
9. **content_original**: Original body of the article (from newspaper3k library)
10. **content**: Processed and tokenized content used as model input
11. **bias_text**: Political bias label (left, center, or right)
12. **bias**: Numeric encoding of political bias (0, 1, or 2)

The `./data/splits` directory contains two types of evaluation splits:
- **random**: Articles randomly split into train/validation/test sets
- **media-based**: Split based on news sources to test generalization to unseen media outlets

## Setup Instructions

### Virtual Environment Setup

Choose one of the following methods:

#### Option 1: Using Conda
```bash
# Create and activate environment
conda create -n bias-detection python=3.9
conda activate bias-detection

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using venv
```bash
# Create and activate environment
python -m venv bias-env
bias-env\Scripts\activate  # Windows
source bias-env/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Models and Approaches

This repository implements two main approaches for political bias detection:

### Baseline Model

A standard transformer-based classifier that directly fine-tunes a pre-trained model on the bias classification task.

**Key features:**
- Uses HuggingFace transformer models (default: DistilBERT)
- Simple fine-tuning approach
- Directly predicts political bias labels

### Triplet Loss Pre-training (TLP)

An enhanced approach that uses triplet loss to improve representation learning before fine-tuning on the classification task.

**Key features:**
- Two-phase training process (pre-training + fine-tuning)
- Maps articles with similar political biases closer together in embedding space
- Pushes articles with different biases further apart
- Supports different mining strategies for triplet selection

**How it works:**
1. **Pre-training Phase**: The model learns to map articles with similar political biases closer together in the embedding space while pushing articles with different biases further apart.
2. **Fine-tuning Phase**: After pre-training, the model is fine-tuned on the classification task to predict the political bias of articles.

### LLM Bias Detection

A new feature that allows benchmarking Large Language Models (LLMs) for political bias by analyzing their responses to politically sensitive topics.

**Key features:**
- Uses fine-tuned bias detection models to evaluate LLM-generated content
- Supports multiple LLM models from HuggingFace
- Provides quantitative bias scores and distributions
- Includes a set of politically sensitive topics for consistent evaluation

**How it works:**
1. **Response Generation**: The system prompts LLMs with politically sensitive topics to generate responses
2. **Bias Analysis**: A fine-tuned bias detection model evaluates the political leaning of each response
3. **Benchmarking**: Results are aggregated to produce bias scores and distributions for each LLM

## Running Experiments

### 1. Running the Baseline Model

```bash
python main.py \
  --model_name distilbert-base-uncased \
  --model_type baseline \
  --num_epochs 3 \
  --batch_size 32 \
  --max_length 256 \
  --learning_rate 2e-5 \
  --split_type random
```

### 2. Running the TLP Model

#### Complete TLP Pipeline (Pre-training + Fine-tuning)

```bash
python run_triplet_pretraining.py \
  --model_name distilbert-base-uncased \
  --pretrain_epochs 5 \
  --finetune_epochs 3 \
  --pretrain_batch_size 16 \
  --finetune_batch_size 32 \
  --max_length 256 \
  --split_type random
```

#### Advanced TLP Configuration

```bash
python run_triplet_pretraining.py \
  --model_name distilbert-base-uncased \
  --pretrain_batch_size 8 \
  --pretrain_epochs 3 \
  --pretrain_lr 2e-5 \
  --max_length 256 \
  --split_type random \
  --margin 0.5 \
  --mining_strategy hard \
  --finetune_batch_size 16 \
  --finetune_epochs 5 \
  --finetune_lr 5e-5
```

### 3. Using Different HuggingFace Models

You can experiment with different pre-trained models by changing the `--model_name` parameter:

```bash
# Using BERT base
python run_triplet_pretraining.py \
  --model_name bert-base-uncased \
  --pretrain_batch_size 8 \
  --pretrain_epochs 3 \
  --pretrain_lr 2e-5 \
  --finetune_batch_size 16 \
  --finetune_epochs 5 \
  --finetune_lr 5e-5 \
  --max_length 256 \
  --split_type random

# Using RoBERTa base
python run_triplet_pretraining.py \
  --model_name roberta-base \
  --pretrain_batch_size 8 \
  --pretrain_epochs 3 \
  --pretrain_lr 2e-5 \
  --finetune_batch_size 16 \
  --finetune_epochs 5 \
  --finetune_lr 5e-5 \
  --max_length 256 \
  --split_type random
```

### 4. Comparing Results

To run both models sequentially and compare their results:

```bash
# Step 1: Run the baseline model
python main.py \
  --model_name distilbert-base-uncased \
  --model_type baseline \
  --num_epochs 3 \
  --batch_size 32 \
  --max_length 256 \
  --split_type random

# Step 2: Run the TLP model
python run_triplet_pretraining.py \
  --model_name distilbert-base-uncased \
  --pretrain_epochs 5 \
  --finetune_epochs 3 \
  --pretrain_batch_size 16 \
  --finetune_batch_size 32 \
  --max_length 256 \
  --split_type random

# Step 3: View the consolidated results
cat results/consolidated_results.csv
```

All results are stored in a single CSV file (`results/consolidated_results.csv`) that gets updated with each new experiment, making it easy to track and compare performance metrics.

### 5. Running the LLM Bias Benchmark

The LLM Bias Benchmark tool allows you to evaluate the political bias of various Large Language Models:

```bash
# Basic usage with default models
python run_llm_bias_benchmark.py

# Specify custom models to benchmark
python run_llm_bias_benchmark.py \
  --models meta-llama/Llama-3-8b-hf mistralai/Mistral-7B-Instruct-v0.2 \
  --bias_model_path ./results/roberta-base-bias-model.pt \
  --output_dir ./llm_benchmark_results

# Use a specific device
python run_llm_bias_benchmark.py --device cuda
```

The benchmark will:
1. Load the specified LLM models
2. Generate responses for politically sensitive topics (defined in `topics.json`)
3. Analyze the political bias of each response
4. Produce a summary of bias distributions and scores

Results are saved in the specified output directory, with generated responses and a CSV file containing bias predictions.

## Parameter Guidelines

### Common Parameters
- **model_name**: Any HuggingFace transformer model (recommended: distilbert-base-uncased)
- **max_length**: Maximum sequence length (recommended: 256-512)
- **split_type**: Either "random" or "media-based"

### Baseline Model Parameters
- **batch_size**: 8-32 (GPU memory dependent)
- **learning_rate**: 1e-5 to 5e-5
- **num_epochs**: 3-5 (can be increased for better performance)

### TLP-Specific Parameters

#### Pre-training Phase Parameters
- **pretrain_batch_size**: 8-32 (GPU memory dependent)
  - Smaller batch sizes (8-16) work better for hard mining strategies
  - Larger batch sizes (16-32) provide more triplet combinations for random mining
- **pretrain_epochs**: 3-5
  - More epochs generally improve representation quality but risk overfitting
  - Monitor validation performance to determine optimal stopping point
- **pretrain_lr**: 1e-5 to 5e-5
  - Lower learning rates (1e-5 to 2e-5) provide more stable training
  - Higher rates (3e-5 to 5e-5) may converge faster but risk overshooting

#### Fine-tuning Phase Parameters
- **finetune_batch_size**: 16-32
  - Standard classification batch sizes apply here
  - Larger batches generally improve stability of gradient updates
- **finetune_epochs**: 3-5
  - Fewer epochs are typically needed after effective pre-training
  - Early stopping based on validation performance is recommended
- **finetune_lr**: 1e-5 to 5e-5
  - Can be set slightly higher than pre-training learning rate
  - 3e-5 to 5e-5 often works well for the classification task

#### Triplet Loss Specific Parameters
- **margin**: Margin for triplet loss (recommended: 0.5-1.0)
  - Controls the minimum distance between different classes in embedding space
  - Smaller margins (0.3-0.5) create tighter clusters but may not separate classes well
  - Larger margins (0.7-1.0) enforce stronger separation but may be harder to train
- **mining_strategy**: Strategy for mining triplets
  - **random**: Randomly selects triplets (anchor, positive, negative)
    - Fastest but least effective strategy
    - Good for initial experiments or very large datasets
  - **semi-hard**: Selects triplets where negatives are neither too easy nor too hard
    - Balances training difficulty and stability
    - Often provides the best trade-off between performance and training time
  - **hard**: Selects the most challenging triplets (negatives closest to anchor)
    - Most effective for learning but can lead to unstable training
    - May require smaller learning rates and batch sizes

#### Advanced Configuration Tips
- For best results, combine hard mining with smaller batch sizes and learning rates
- Semi-hard mining works well with medium batch sizes and learning rates
- Random mining benefits from larger batch sizes to increase triplet diversity
- Consider starting with random mining and then switching to hard mining in later epochs

### LLM Bias Benchmark Parameters

- **models**: List of HuggingFace model IDs to benchmark (e.g., 'meta-llama/Llama-3-8b-hf')
  - Use models that are available locally or through HuggingFace with appropriate access
  - Smaller models (7B-8B parameters) run faster but may have less nuanced responses
  - Larger models (>30B parameters) provide more sophisticated responses but require more GPU memory

- **bias_model_name**: Pre-trained model used for the bias detector (default: 'roberta-base')
  - Should match the model used to train your bias detection weights

- **bias_model_path**: Path to fine-tuned bias detection model weights
  - Must point to a model trained on the political bias classification task

- **output_dir**: Directory to save benchmark results (default: './llm_benchmark_results')
  - Will be created if it doesn't exist
  - Contains generated responses and bias analysis results

- **topics_file**: Path to JSON file containing topics (default: './topics.json')
  - Should contain a list of politically sensitive topics
  - Can be customized to focus on specific political issues

## Performance Comparison

### Results on Random Split (Ranked by Macro F1)

| Model | Approach | Macro F1 | Accuracy | MAE |
|-------|----------|----------|----------|-----|
| FacebookAI/roberta-base-baseline-max-performance | Baseline | 86.09 | 85.85 | 0.23 |
| FacebookAI/roberta-base-baseline | Baseline | 85.24 | 85.15 | 0.23 |
| roberta-base-baseline | Baseline | 85.02 | 84.92 | 0.23 |
| FacebookAI/roberta-base-baseline-high-learing-rate-5e-6 | Baseline | 81.05 | 80.85 | 0.31 |
| roberta-base-baseline | Baseline | 77.67 | 77.31 | 0.36 |
| FacebookAI/roberta-base-baseline | Baseline | 77.83 | 77.54 | 0.35 |
| roberta-base-triplet-pretrained | TLP | 78.7 | 77.77 | 0.38 |
| FacebookAI/roberta-base-triplet-pretrained | TLP | 76.19 | 75.69 | 0.4 |
| BERT Large | TLP | 76.43 | 75.85 | 0.40 |
| BERT Large | Baseline | 75.30 | 74.69 | 0.41 |
| PoliticalBiasBERT | Baseline | 75.08 | 75.00 | 0.40 |
| DistilBERT | Unsupervised SimCSE | 74.53 | 73.62 | 0.44 |
| BERT Base | Baseline | 73.25 | 72.92 | 0.42 |
| DistilBERT | TLP | 73.13 | 73.00 | 0.45 |
| DistilBERT | Hybrid Pretrained | 71.70 | 71.38 | 0.45 |
| DistilBERT | Baseline | 70.03 | 69.15 | 0.51 |

### Results on Media-Based Split (Ranked by Macro F1)

| Model | Approach | Macro F1 | Accuracy | MAE |
|-------|----------|----------|----------|-----|
| BERT Large | Baseline | 38.25 | 43.85 | 0.78 |
| BERT Large | TLP | 21.03 | 46.08 | 0.85 |

## Citation

```
@inproceedings{baly2020we,
  author      = {Baly, Ramy and Da San Martino, Giovanni and Glass, James and Nakov, Preslav},
  title       = {We Can Detect Your Bias: Predicting the Political Ideology of News Articles},
  booktitle   = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  series      = {EMNLP~'20},
  NOmonth     = {November},
  year        = {2020}
  pages       = {4982--4991},
  NOpublisher = {Association for Computational Linguistics}
}
```
