import argparse
parser = argparse.ArgumentParser(description='Generates samples of a given instruction on a given dataset.')
parser.add_argument("--model_name", help="Model name", required=True)
parser.add_argument("--chat_template", help="Chat template", required=True)
parser.add_argument("--n_words", help="Number of words in summary", required=True)
parser.add_argument("--split", help="Split (train vs test)", required=True)
args = parser.parse_args()

import os
import pandas as pd
import torch
from datasets import load_dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from IPython.display import clear_output
import logging

import os

while "notebooks" in os.getcwd():
    os.chdir("..")


import sys
sys.path.insert(0, './')

from src.train_test_split import stratified_train_test_split

model_name = args.model_name
chat_template = args.chat_template
n_words = args.n_words
split = args.split

dataset = pd.read_json("data/wikipedia_dataset.json")

train_df , test_df = stratified_train_test_split(dataset, test_size=0.4)
train_idx = train_df.id.tolist()
test_idx = test_df.id.tolist()

if split == 'train':
    mask = dataset.id.isin(train_idx) 
    dataset = dataset[mask] 

if split == 'test':
    mask = dataset.id.isin(test_idx) 
    dataset = dataset[mask] 

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", cache_dir = '/Data')

# def update_short_entries(df):
#     for idx, row in df.iterrows():
#         paragraphs = row['base_text'].split('\n\n')
#         para_index = 1  # Start from the second paragraph

#         while row['num_tokens'] < 50 and para_index < len(paragraphs):
#             df.at[idx, 'text'] += '\n' + paragraphs[para_index]  # Modify the DataFrame directly
#             df.at[idx, 'num_tokens'] = len(tokenizer.encode(df.at[idx, 'text']))  # Recompute token count
#             para_index += 1

#     return df

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("script.log"),
])
logger = logging.getLogger(__name__)

# dataset = update_short_entries(dataset)

torch.cuda.set_per_process_memory_fraction(0.6, device=0)

logger.info("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=6_000,
    dtype=None,
    load_in_4bit=True,
    fast_inference=True,
    cache_dir='/Data'
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template=chat_template,
)

FastLanguageModel.for_inference(model)

out_dir = f"data/generated_dataset_{split}_{n_words}_{model_name.split('/')[-1]}.pkl"

generated = []

# Check if output file exists and load existing data
if os.path.exists(out_dir):
    existing_data = pd.read_pickle(out_dir)
    processed_ids = set(existing_data['id'])
    generated.extend(existing_data.to_dict('records'))
    logger.info(f"Resuming from {len(existing_data)} already processed entries.")
    out_dir = f"data/generated_dataset_{n_words}_{model_name.split('/')[-1]}_2.pkl"
else:
    processed_ids = set()

try:
    for i, (_, row) in tqdm(enumerate(dataset.sort_values("num_tokens", ascending=False).iterrows()), total=len(dataset)):
        if row['id'] in processed_ids:
            continue  # Skip already processed IDs
        
        prompt = f'''
            Você é um assistente virtual que deve gerar resumos de textos em português. 
            Seu resumo deve ter, no máximo {n_words} palavras e conter todas as informações principais do texto.
            Esse é o texto:
            
            {row["text"]}
            
            Faça um resumo de no máximo {n_words} palavras do texto acima.
        '''

        message = [{'role': 'user', 'content': prompt}]
        inputs = tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = model.generate(inputs, max_new_tokens=500)
        generated_text = tokenizer.decode(generated_ids[0, inputs.shape[1]:])

        generated.append({'id': row['id'], 'generated_text': generated_text})
        print(f"Generated text for ID {row['id']}")
        print(generated_text)

        if i % 100 == 0:
            pd.DataFrame(generated).to_pickle(out_dir)  # Save progress every 100 iterations
            logger.info(f"Saved progress at {i} entries.")
except Exception as e:
    logger.error(f"Error during text generation: {e}")
    logger.info(prompt)

pd.DataFrame(generated).to_pickle(out_dir)