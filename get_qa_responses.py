import argparse
import dataclasses
import json
import logging
import os
import pathlib
import random
import re
import sys
import time
from copy import deepcopy
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils.decoding_utils import ParallelDecoding
from lostinmid import Document, get_qa_prompt
from ddp import get_rank, setup, cleanup, spawn_nproc

logger = logging.getLogger(__name__)
random.seed(0)


def format_chat_prompt(message: str):
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Please ensure that your responses are socially unbiased and positive in nature. "
        "If a question does not make any sense, or is not factually coherent, explain "
        "why instead of answering something not correct. If you don't know the answer "
        "to a question, please don't share false information."
    )
    lines = ["<s>[INST] <<SYS>>", DEFAULT_SYSTEM_PROMPT, "<</SYS>>", "", f"{message} [/INST]"]
    return "\n".join(lines)


def demo_fn(rank, args, cfg, dataset):
    setup(rank, world_size=torch.cuda.device_count(), args=args)
    torch.cuda.set_device(rank)  # <- This is crucial

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16  # Avoid bitsandbytes for DDP
    ).to(f"cuda:{rank}")

    model = DDP(model, device_ids=[rank], output_device=rank)

    prompt_template = "Write a high-quality answer for the given question using only the provided search results.\n\n{search_results}\n\nQuestion: {question}\nAnswer:"
    prompt_template_wo_results = "Write a high-quality answer for the given question.\n\nQuestion: {question}\nAnswer:"

    pattern = r'##(Passages\d+):([\s\S]+?)(?=##|\Z)'
    for data_idx in range(rank, len(dataset), torch.cuda.device_count()):
        i = dataset[data_idx]
        kldout = []
        for id2, j in enumerate(i["full_queries"]):
            prompts, all_model_documents_texts, questions = [], [], []
            question = i["problem_statements"][id2]
            matches = re.findall(pattern, j)
            ctxs = []

            for ii, match in enumerate(matches, 1):
                if ii not in [1, 4, 7, 10]:
                    continue
                title_map = {1: "Online Tutorials", 2: "Online Tutorials", 4: "Github Repos", 5: "Github Repos",
                             7: "Programming Solutions", 8: "Programming Solutions", 10: "Library Documentations", 11: "Library Documentations"}
                ttle = title_map[ii]
                if ii == 3:
                    m = match[1][::-1].replace(":sopeR buhtiG#", "", 1)[::-1]
                elif ii == 6:
                    m = match[1][::-1].replace(":snoituloS gnimmargorP#", "", 1)[::-1]
                elif ii == 9:
                    m = match[1][::-1].replace(":snoitatnemucoD yrarbiL#", "", 1)[::-1]
                else:
                    m = match[1]
                ctxs.append({'title': ttle, 'text': m, 'score': None})

            documents = [Document.from_dict(deepcopy(ctx)) for ctx in ctxs]
            prompt = get_qa_prompt(question, documents, mention_random_ordering=False, query_aware_contextualization=False)
            prompt = format_chat_prompt(prompt)
            prompts.append(prompt)
            documents_texts = [f"(Title: {document.title}) {document.text}" for document in documents]
            all_model_documents_texts.append(documents_texts)
            questions.append(question)

            pd_object = ParallelDecoding(model=model.module, tokenizer=tokenizer, device=torch.device(f"cuda:{rank}"), using_norm=False, using_entropy=True)
            generate_func = pd_object.clehe_using_logits

            for question, document_texts in zip(questions, all_model_documents_texts):
                _, _, response, _ = generate_func(
                    prompt_template=prompt_template,
                    prompt_template_wo_results=prompt_template_wo_results,
                    question=question,
                    document_texts=document_texts,
                    max_tokens=800,
                    beta=0.25,
                    temp_cpmi=0.1,
                    metric_criterion="weighted_entropy",
                    temperature=1e-4,
                    sampling_method="greedy",
                    alpha=0.1,
                    candidate_layers=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
                    reweight_logit=False)
                print(f"[RANK {rank}] Response: {response}")
                kldout.append(response)

        dataset[data_idx]["outputs"] = kldout

    if get_rank() == 0:
        with open("kldgen.json", "w") as f:
            json.dump(dataset, f, indent=4)

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=12355)
    args = parser.parse_args()

    with open('/home/users/ntu/mohor001/raginftemplate.json', 'r') as file:
        dataset = json.load(file)

    spawn_nproc(demo_fn, args, cfg=None, dataset=dataset)
    logger.info("Finished distributed inference.")
