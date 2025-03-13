import os

if 'notebooks' in os.getcwd():
    os.chdir("..")


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
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import aiohttp
import asyncio
from tqdm.asyncio import tqdm
import torch
import scienceplots
plt.style.use(['science', 'no-latex'])
from IPython.display import clear_output
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from langdetect import detect

import sys
sys.path.insert(0, './')

from src.train_test_split import stratified_train_test_split

import evaluate




tqdm.pandas()


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", cache_dir = '/Data')
torch.cuda.set_per_process_memory_fraction(0.6, device=0)

base_data = pd.read_json("data/wikipedia_dataset.json")

train_df , temp_df = stratified_train_test_split(base_data, test_size=0.4)
val_df , test_df = stratified_train_test_split(temp_df, test_size=0.5)

train_idx = train_df.id.tolist()
val_idx = val_df.id.tolist()
test_idx = test_df.id.tolist()

base_path = "data/generated_dataset_test_100_qwen-0.5b-instruct-summary-pt-rank{lora_rank}.pkl"

all_df = []
raw_model_df = pd.read_pickle("data/generated_dataset_Qwen2.5-0.5B-Instruct.pkl")
raw_model_df['model_name'] = "Qwen-0.5B-Instruct"

all_df.append(raw_model_df)

raw_model_df = pd.read_pickle("data/generated_dataset_test_100_Llama-3.2-1B-Instruct-bnb-4bit.pkl")
raw_model_df['model_name'] = "Llama-1B-Instruct"

all_df.append(raw_model_df)

raw_model_df = pd.read_pickle("data/generated_dataset_test_100_Qwen2.5-3B-Instruct-unsloth-bnb-4bit.pkl")
raw_model_df['model_name'] = "Qwen-3B-Instruct"

all_df.append(raw_model_df)

reference_summary = pd.read_pickle("data/generated_dataset_100_Meta-Llama-3.1-8B-Instruct-bnb-4bit_2.pkl")\
    .rename(columns = {'generated_text': 'reference_summary'})


for lora_rank in [64]:
    temp = pd.read_pickle(base_path.format(lora_rank = lora_rank))
    
    temp['model_name'] = "Finetuned Model"
    all_df.append(temp)
temp_df = pd.concat(all_df, ignore_index=True)
base_data['text'].apply(lambda x: len(x.split()))
df = pd.merge(
    base_data[['id', 'text']],
    temp_df
).query(f"id in {test_idx}")

df = pd.merge(
    base_data[['id', 'text']],
    temp_df,
    on='id'
).query(f"id in {test_idx}")

df = pd.merge(df, reference_summary, on='id')


model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name, # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = 6_000,
    dtype = None,
    load_in_4bit = True,
    fast_inference=True,
    cache_dir = '/Data'
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

base_prompt = '''
    Você é um assistente útil que classifica resumos.  
    Fornecerei o texto e dois resumos (0 e 1) de aproximadamente 100 palavras desse texto em português.  

    Você deve indicar qual deles é o melhor resumo, com base tanto na qualidade do resumo quanto na qualidade do texto em português e no tamanho do texto (deve ter aproximadamente 100 palavras).
    Um resumo que tenha mais de 200 ou menos de 30 palavras deve ser considerado ruim.

    Aqui está o texto:  
    <text>  
    {text}  
    </text>  

    Aqui está o resumo 0:  
    <0>  
    {summary_0}  
    </0>  

    Aqui está o resumo 1:  
    <1>  
    {summary_1}  
    </1>  

    Responda no seguinte formato (JSON):  
    
    {{
        "best_summary": (0 ou 1),
        "explanation": "uma breve explicação do porquê."
    }}
    
'''


tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

FastLanguageModel.for_inference(model)

out_dir = f"data/win_rates.pkl"

generated = []

# Check if output file exists and load existing data
if os.path.exists(out_dir):
    existing_data = pd.read_pickle(out_dir)
    processed_ids = set(existing_data['id'])
    generated.extend(existing_data.to_dict('records'))
    # out_dir = f"data/win_rates_2.pkl"
else:
    processed_ids = set()

i = 0
for idx, group in tqdm(df.groupby("id"), total = len(df)//4):
    if idx in processed_ids:
        continue  # Skip already processed IDs

    finetuned_summary = group.query("model_name == 'Finetuned Model'")

    text_finetune = finetuned_summary['generated_text'].item()

    for idx, row in group.iterrows():
        if row['model_name'] == 'Finetuned Model':
            continue
        text_model = row['generated_text']


        index_of_finetune = int(np.random.random() > 1/2)
    
        shuffling_dict = {
            index_of_finetune: text_finetune,
            1 - index_of_finetune: text_model
        }

        inverse_shuffling_map = {
            index_of_finetune: "Finetuned Model",
            1 - index_of_finetune: row['model_name']
        }

        prompt = base_prompt.format(
            text = finetuned_summary["text"].item(),
            summary_0 = shuffling_dict[0],
            summary_1 = shuffling_dict[1]
        )

        message = [{'role': 'user', 'content': prompt}]
        inputs = tokenizer.apply_chat_template(
            message,
            tokenize = True,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
        ).to("cuda")

        generated_ids = model.generate(inputs, max_new_tokens = 500)

        generated_text = tokenizer.decode(generated_ids[0, inputs.shape[1]:]).split("<|eot_id|>")[0]

        # print(index_of_finetune)
        # print(row['model_name'])
        # print(generated_text)

        try:
            generated_json = json.loads(generated_text)
            generated_json['best_summary'] = inverse_shuffling_map[generated_json['best_summary']]
            generated_json["index_of_modified"] = index_of_finetune

            new_row = {
                'id': row['id'],
                'model': row['model_name'],
                'winner': generated_json['best_summary'],
                'explanation': generated_json['explanation']
            }

            print(new_row)

            generated.append(new_row)

        except Exception as e:
            print(f"Error processing LLM: {e}")


    if i % 100 == 0:
        pd.DataFrame(generated).to_pickle(out_dir)  # Save progress every 100 iterations
    i+=1
    # break
    clear_output()
    # print(generated_text)

pd.DataFrame(generated).to_pickle(out_dir)