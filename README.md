# Fine-tuning Small Language Models for Summarization in Portuguese

## Overview
This project focuses on fine-tuning a small language model (SLM) for text summarization in Portuguese. It involves:
- Collecting Wikipedia articles in Portuguese
- Generating high-quality target summaries using a large pre-trained model
- Fine-tuning the Qwen2.5-0.5B-Instruct model using LoRA
- Evaluating performance using **ROUGE** and **BERTScore**

Our experiments show that fine-tuning significantly improves summary quality and that the fine-tuned model performs comparably to a model twice its size.

## Repository Structure
```
📂 root
│-- 📂 src               # Useful python modules
│
│-- 📂 data              # Generated datasets
│
│-- 📂 notebooks         # Jupyter notebooks for experiments
│   │-- 01-EDA.ipynb                   # Exploratory Data Analysis
│   │-- 02-dataset_generation.ipynb    # Generating datasets
│   │-- 03-generate_target_summaries.ipynb # Target summary generation
│   │-- 04-analysing_generation.ipynb  # Analysis of generated summaries
│   │-- 05-sft.ipynb                    # Supervised fine-tuning
│   │-- 06-score.ipynb                   # Scoring metrics
│   │-- 07-hyperparameter_search.ipynb   # Hyperparameter optimization
│   │-- 08-results.ipynb                 # Experiment results
│   │-- 09-llm-as-a-judge.ipynb          # LLM-based qualitative evaluation
│
│-- 📂 scripts               # Standalone scripts for running experiments
│   │-- generate_target.py   # Bash script for training the model
│   │-- hyperparameter_tuning.py     # Script for model inference
│   │-- llm_as_a_judge.py             # LLM-as-a-Judge evaluation
|
│-- 📜 requirements.txt  # Dependencies
```

## Requirements
To set up the environment, install the necessary dependencies using:
```sh
pip install -r requirements.txt
```
Ensure you have **PyTorch** and **HuggingFace Transformers** installed. The model also requires **LoRA** fine-tuning with the `unsloth` package.

## Fine-tuning the Model
To fine-tune the model, run the following command:
```sh
python scripts/hyperparameter_tuning.py
```
The training script supports LoRA tuning with configurable rank parameters.

## Running Inference
To generate summaries using the fine-tuned model:
```sh
python scripts/generate_target.py --model_name {MODEL_NAME} --chat_template {unsloth chat template} --n_words {n_words} --split {train or test}
```
```

## Results
- The fine-tuned model improves **ROUGE-Lsum** from **0.29 → 0.41**
- BERTScore improves from **0.74 → 0.80**
- The model produces significantly more structured and informative summaries

```

## Acknowledgments
This work was conducted using computational resources from **École Polytechnique**. We thank the open-source NLP community for their contributions to model development and dataset curation.

