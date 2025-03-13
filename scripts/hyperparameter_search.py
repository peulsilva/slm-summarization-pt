import os

while "notebooks" in os.getcwd():
    os.chdir("..")


import sys
sys.path.insert(0, './')

import evaluate
from datasets import load_dataset, Dataset
from huggingface_hub import notebook_login
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only, get_chat_template
import aiohttp
import asyncio
from tqdm.asyncio import tqdm
import torch
from IPython.display import clear_output
from src.pytorch_utils import count_parameters
from src.train_test_split import stratified_train_test_split
from unsloth import FastLanguageModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("script.log"),
])
logger = logging.getLogger(__name__)

tqdm.pandas()
## Loading base model
max_seq_length = 6000
dtype = None 
load_in_4bit = True

notebook_login()
torch.cuda.set_per_process_memory_fraction(0.6, device=0)

base_data = pd.read_json("data/wikipedia_dataset.json")
completions_df = pd.read_pickle("data/generated_dataset_100_Meta-Llama-3.1-8B-Instruct-bnb-4bit_2.pkl")
text_df = pd.merge(
    base_data,
    completions_df,
    on = 'id'
)[['id', 'text', 'num_tokens', 'generated_text']]
text_df.head()  
train_df , temp = stratified_train_test_split(text_df, test_size=0.4)
test_df, val_df = stratified_train_test_split(temp, test_size=0.5)
n_words = 100

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2-0.5B-Instruct", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    cache_dir = '/Data'
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
)

def format_row(row):
    prompt = f'''
        Você é um assistente virtual que deve gerar resumos de textos em português. 
        Seu resumo deve ter, no máximo {n_words} palavras e conter todas as informações principais do texto.
        Esse é o texto:

        {row['text']}
        
        Faça um resumo de no máximo {n_words} palavras do texto acima.
    '''

    row['conversations'] = [{'role': 'user', "content": prompt}, {'role': 'assistant', 'content': row['generated_text']}]
    return row


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

def generate_and_tokenize_prompt(text):
    # Generate the full prompt with the assistant's response
    tokenized_full_prompt = tokenizer(text, return_tensors='pt')

    # Clone the input_ids to create labels
    labels = tokenized_full_prompt.input_ids.clone()

    # Find the position of "<assistant>" in the prompt
    prompt_text = text[:text.find(">assistant")] + ">assistant"
    end_prompt_idx = len(tokenizer(prompt_text, return_tensors="pt")["input_ids"][0])

    # Mask all tokens before "<assistant>" with -100
    labels[:, :end_prompt_idx] = -100
    # labels = labels[:, end_prompt_idx:]

    return {
        'input_ids': tokenized_full_prompt.input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': tokenized_full_prompt.attention_mask.flatten(),
    }

def preprocess_df(df):
    temp_df = df.progress_apply(format_row, axis =1)
    temp_df = formatting_prompts_func(temp_df)

    tokens = []
    for t in tqdm(temp_df['text']):
        tokens.append(generate_and_tokenize_prompt(t))

    
    tokens_dataset = Dataset.from_list(tokens)
    return tokens_dataset
train_dataset = preprocess_df(train_df)
val_dataset = preprocess_df(val_df)
test_dataset = preprocess_df(test_df)

logger.info("Processing DF")
# Supervised Fine Tuning (SFT)


for lora_rank in [8, 16, 32, 64]:
    logger.info(f"Finetuning for Lora Rank = {lora_rank}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2-0.5B-Instruct", # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        cache_dir = '/Data'
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 2 * lora_rank,
        lora_dropout = 0.05, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    count_parameters(model)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5",
    )

    trainer = SFTTrainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset= val_dataset,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        # compute_metrics = compute_metrics,
        args = SFTConfig(
            # eval_strategy = 'steps',
            # eval_steps = 1,
            per_device_train_batch_size = 16,
            gradient_accumulation_steps = 4,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            warmup_steps = 5,
            num_train_epochs = 4, # Set this for 1 full training run.
            learning_rate = 2e-5,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            # hub_model_id= "peulsilva/slm-pt",
            # save_total_limit=1,
            # output_dir = "/Data",
            # report_to = "none", # Use this for WandB etc,
            # push_to_hub=True,
            # save_steps = 10,
            # hub_strategy="every_save",
            max_seq_length = max_seq_length,
            # dataset_text_field = 'text',
            batch_eval_metrics = True,
            # padding = True,
        ),
    )


    trainer.train()
    trainer.model.push_to_hub_merged(f"peulsilva/qwen-0.5b-instruct-summary-pt-rank{lora_rank}")
    trainer.tokenizer.push_to_hub(f"peulsilva/qwen-0.5b-instruct-summary-pt-rank{lora_rank}")