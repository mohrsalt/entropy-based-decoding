#!/usr/bin/env python3
"""Given a data file with questions and retrieval results to use, run Llama-2 to get responses.

"""
##check main fn-> done, change title-> done, align lists-> done, check internals
import argparse
import dataclasses
import json
import os 
import logging
import pathlib
import random
import sys
from copy import deepcopy
from typing import List
from utils.decoding_utils import ParallelDecoding

import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from collections import defaultdict
import heapq

import time 

from lostinmid import (
    Document,
    get_qa_prompt,
)

logger = logging.getLogger(__name__)
random.seed(0)


def main(
):

    logger.info("Loading tokenizer")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",quantization_config=quantization_config, torch_dtype="auto")


    prompt_template = "Write a high-quality answer for the given question using only the provided search results.\n\n{search_results}\n\nQuestion: {question}\nAnswer:"
    prompt_template_wo_results = "Write a high-quality answer for the given question.\n\nQuestion: {question}\nAnswer:"

    with open('/home/users/ntu/mohor001/raginftemplate.json','r') as file:
        data=file.read()

    data=json.loads(data)
    pattern = r'##(Passages\d+):([\s\S]+?)(?=##|\Z)'
    
    for idx,i in enumerate(data):
        if(idx==2):
            break
        kldout=[]
        for id2,j in enumerate(i["full_queries"]):
            
            prompts = []
            all_model_documents = []

            all_model_documents_texts = []
            questions = []
            question=i["problem_statements"][id2]
            matches = re.findall(pattern, j)
            ctxs=[]
            ttle=None
            for ii, match in enumerate(matches, 1):
                if(ii!=1 and ii!=2 and ii!=4 and ii!=5 and ii!=7 and ii!=8 and ii!=10 and ii!=11):
                      continue
                if(ii<4):
                    ttle="Online Tutorials"
                elif(ii>3 and ii<7):
                    ttle="Github Repos"
                elif(ii>6 and ii<10):
                    ttle="Programming Solutions"
                else:
                    ttle="Library Documentations"
                if(ii==3):
                    m=match[1][::-1].replace(":sopeR buhtiG#", "", 1)[::-1]
                elif(ii==6):
                    m=match[1][::-1].replace(":snoituloS gnimmargorP#", "", 1)[::-1]
                elif(ii==9):
                    m=match[1][::-1].replace(":snoitatnemucoD yrarbiL#", "", 1)[::-1] ##Change here 
                else:
                    m=match[1]
                ctxs.append({'title': ttle, 'text': m, 'score': None})
            documents = []
            for ctx in deepcopy(ctxs):
                        documents.append(Document.from_dict(ctx))
            prompt = get_qa_prompt(
                        question,
                        documents,
                        mention_random_ordering=False,
                        query_aware_contextualization=False,
                    )
            prompt = format_chat_prompt(prompt)
            prompts.append(prompt)
            all_model_documents.append(documents)
            documents_texts = [f"(Title: {document.title}) {document.text}" for document in documents]
            all_model_documents_texts.append(documents_texts)
            questions.append(question)
            device = torch.device("cuda")
            pd_ojbect = ParallelDecoding(model=model, tokenizer=tokenizer, device=device, using_norm=False,using_entropy=True)
            
            generate_func = pd_ojbect.clehe_using_logits
            
            
            for question, documents_texts in tqdm(zip(questions, all_model_documents_texts)):

                    print("Response here: ")
                    _, _, response, _ = generate_func(prompt_template=prompt_template, prompt_template_wo_results=prompt_template_wo_results,
                                        question=question, document_texts=documents_texts, max_tokens=800,
                                        beta=0.25, temp_cpmi=0.1, metric_criterion="weighted_entropy", temperature=1e-4,
                                        sampling_method="greedy", alpha=0.1,
                                        # candidate_layers=[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
                                        candidate_layers=[2,10,24,30],
                                        reweight_logit=False)
                    print(response)
                    kldout.append(response)
            torch.cuda.empty_cache()
        i["outputs"]=kldout

    with open("kldgen.json", "w") as kld_file:
        json.dump(data, kld_file, indent=4)





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


if __name__ == "__main__":
    





    main(
    )
    logger.info("finished running %s", sys.argv[0])

