#!/usr/bin/env python3
"""Given a data file with questions and retrieval results to use, run Llama-2 to get responses.

"""
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
from lost_in_the_middle.metrics import best_subspan_em

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict
import heapq
import jsonlines
import time 

from xopen import xopen
from lost_in_the_middle.prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
)

logger = logging.getLogger(__name__)
random.seed(0)


def main(
    input_path,
    model_name,
    temperature,
    closedbook,
    prompt_mention_random_ordering,
    use_random_ordering,
    query_aware_contextualization,
    max_new_tokens,
    max_prompt_length,
    output_path,
    beta,
    metric_criterion,
    sampling_method,
    func_name,
    alpha,
    num_topk_docs,
    sample_num,
    start_index,
    sample_start_index,
    temp_cpmi,
    candidate_layers,
    reweight_logit,
    using_norm,
    using_entropy
):
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # for inference only
    did_format_warn = False
    examples = []
    prompts = []
    all_model_documents = []

    all_model_documents_texts = []
    questions = []

    prompt_template = "Write a high-quality answer for the given question using only the provided search results.\n\n{search_results}\n\nQuestion: {question}\nAnswer:"
    prompt_template_wo_results = "Write a high-quality answer for the given question.\n\nQuestion: {question}\nAnswer:"

    # make directory for outputpath
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    # Fetch all of the prompts
    if input_path.endswith(".jsonl"):
        with jsonlines.open(input_path, 'r') as f_reader: # For Main Experiment in Table 1
            for idx, input_example in enumerate(f_reader):
                # TODO for test only
                if idx < sample_start_index or idx >= sample_start_index + sample_num:
                    continue

                # Get the prediction for the input example
                question = input_example["question"]
                ctxs = []
                for doc_content, doc_score, doc_title in zip(input_example['doc_contents'], input_example['doc_scores'], input_example['doc_titles']):
                    ctxs.append({'title': doc_title, 'text': doc_content, 'score': doc_score})

                if closedbook:
                    documents = []
                else:
                    documents = []
                    for ctx in deepcopy(ctxs):
                        documents.append(Document.from_dict(ctx))
                    if not documents:
                        raise ValueError(f"Did not find any documents for example: {input_example}")
                    
                # consider the top-k documents
                documents= documents[start_index: num_topk_docs + start_index]

                if use_random_ordering:
                    raise NotImplementedError

                if closedbook:
                    prompt = get_closedbook_qa_prompt(question)
                else:
                    # answering with the search results
                    prompt = get_qa_prompt(
                        question,
                        documents,
                        mention_random_ordering=prompt_mention_random_ordering,
                        query_aware_contextualization=query_aware_contextualization,
                    )

                if "chat" in model_name:
                    if did_format_warn is False:
                        logger.warning(f"Model {model_name} appears to be an chat model, applying chat formatting")
                        did_format_warn = True
                    prompt = format_chat_prompt(prompt)

                prompt_length = len(tokenizer(prompt)["input_ids"])
                if max_prompt_length < prompt_length:
                    logger.info(
                        f"Skipping prompt {prompt[:100]}... with length {prompt_length}, which "
                        f"is greater than maximum prompt length {max_prompt_length}"
                    )
                    continue
                
                # `prompt` only serves to check if it exceeds the maximum length.
                prompts.append(prompt)
                examples.append(deepcopy(input_example))
                all_model_documents.append(documents)
                documents_texts = [f"(Title: {document.title}) {document.text}" for document in documents]
                all_model_documents_texts.append(documents_texts)
                questions.append(question)

    elif input_path.endswith(".jsonl.gz"): # for "Lost in the middle" experiment
        with xopen(input_path) as fin:
            for line in tqdm(fin):
                input_example = json.loads(line)
                # Get the prediction for the input example
                question = input_example["question"]
                if closedbook:
                    documents = []
                else:
                    documents = []
                    for ctx in deepcopy(input_example["ctxs"]):
                        documents.append(Document.from_dict(ctx))
                    if not documents:
                        raise ValueError(f"Did not find any documents for example: {input_example}")

                if closedbook:
                    prompt = get_closedbook_qa_prompt(question)
                else:
                    # document拿出来
                    prompt = get_qa_prompt(
                        question,
                        documents,
                        mention_random_ordering=prompt_mention_random_ordering,
                        query_aware_contextualization=query_aware_contextualization,
                    )

                if "chat" in model_name:
                    if did_format_warn is False:
                        logger.warning(f"Model {model_name} appears to be an chat model, applying chat formatting")
                        did_format_warn = True
                    prompt = format_chat_prompt(prompt)

                prompt_length = len(tokenizer(prompt)["input_ids"])
                if max_prompt_length < prompt_length:
                    logger.info(
                        f"Skipping prompt {prompt[:100]}... with length {prompt_length}, which "
                        f"is greater than maximum prompt length {max_prompt_length}"
                    )
                    continue

                prompts.append(prompt)
                examples.append(deepcopy(input_example))
                all_model_documents.append(documents)
                documents_texts = [f"(Title: {document.title}) {document.text}" for document in documents]
                all_model_documents_texts.append(documents_texts)
                questions.append(question)


    logger.info(f"Loaded {len(prompts)} prompts to process")

    # Get responses for all of the prompts
    if not torch.cuda.is_available():
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")

    logger.info("Loading model")

    device = torch.device("cuda")
    pd_ojbect = ParallelDecoding(model_path=model_name, tokenizer_path=model_name, device=device, using_norm=using_norm,using_entropy=using_entropy)

    responses = []
    token_document_idx_list = []
    if func_name == "logits":
        if args.beta == 0.: 
            print("using |D| forward process")
            generate_func = pd_ojbect.leens_using_logits
        else:
            generate_func = pd_ojbect.clehe_using_logits
    else:
        raise NotImplementedError
    
    all_top1_idx_consistent_ratio = []
    for question, documents_texts, example in tqdm(zip(questions, all_model_documents_texts, examples)):
        start = time.time()
        token_document_idx, _, response, _ = generate_func(prompt_template=prompt_template, prompt_template_wo_results=prompt_template_wo_results,
                                        question=question, document_texts=documents_texts, max_tokens=max_new_tokens,
                                        beta=beta, temp_cpmi=temp_cpmi, metric_criterion=metric_criterion, temperature=temperature,
                                        sampling_method=sampling_method, alpha=alpha,
                                        candidate_layers=candidate_layers,
                                        reweight_logit=reweight_logit)
        responses.append(response)

        token_document_idx_list.append(token_document_idx)



    with xopen(output_path, "w") as f:
        for example, model_documents, prompt, response, token_document_idx in zip(examples, all_model_documents, prompts, responses, token_document_idx_list):
            output_example = {"answers": example['answers']}
            output_example["model_answer"] = response
            # print(response)
            # output_example['token_doument_idx'] = token_document_idx
            f.write(json.dumps(output_example) + "\n")

def top_three_frequent_elements(lst):
    frequency_dict = defaultdict(int)

    for element in lst:
        frequency_dict[element] += 1

    # Sort by frequency in descending order
    sorted_elements = heapq.nlargest(3, frequency_dict, key=frequency_dict.get)

    return sorted_elements

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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
    
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)

    parser.add_argument("--model", help="Model to use in generating responses", required=True, type=str)
    parser.add_argument("--temperature", help="fixed small value ", type=float, default=1e-4)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument("--closedbook", action="store_true", help="Run the model in closed-book mode (i.e., don't use documents).")

    parser.add_argument("--prompt-mention-random-ordering", action="store_true", help="Mention that search results are ordered randomly in the prompt") # false
    parser.add_argument("--use-random-ordering",action="store_true", help="Randomize the ordering of the distractors, rather than sorting by relevance.") # false
    parser.add_argument("--query-aware-contextualization", action="store_true", help="Place the question both before and after the documents.") # false

    parser.add_argument("--max-new-tokens", help="Maximum number of new tokens to generate", type=int, default=100)
    parser.add_argument("--max-prompt-length", help="Maximum number of tokens in the prompt. Longer prompts will be skipped.", type=int,default=4096)
    
    # decoding-related parameters
    parser.add_argument("--metric-criterion", choices=["weighted_entropy"], type=str)
    parser.add_argument("--beta", help="beta value for balancing the two entropy terms", type=float, default=0)
    
    parser.add_argument("--sampling-method", choices=['greedy'], type=str)
    parser.add_argument("--func_name", choices=["logits"], type=str)
    parser.add_argument("--alpha", help="alpha value", type=float, default=0.1) # do not impact
    parser.add_argument("--temp-cpmi", type=float, default=0.1)
    parser.add_argument("--candidate-layers", type=str, default=[16])

    parser.add_argument("--num-topk-docs", type=int, default=10)

    parser.add_argument("--sample-num", type=int, default=1000000)

    parser.add_argument("--start-index", type=int, default=0)


    # To be continued
    parser.add_argument("--reweight-logit", action="store_true")
    parser.add_argument("--using-norm",action="store_true")
    parser.add_argument("--sample-start-index", type=int, default=0)

    parser.add_argument("--using-entropy", action="store_true")

    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    candidate_layers = eval(args.candidate_layers)
    if isinstance(candidate_layers, int):
        candidate_layers = [candidate_layers]

    main(
        args.input_path,
        args.model,
        args.temperature,
        args.closedbook,
        args.prompt_mention_random_ordering,
        args.use_random_ordering,
        args.query_aware_contextualization,
        args.max_new_tokens,
        args.max_prompt_length,
        args.output_path,
        args.beta,
        args.metric_criterion,
        args.sampling_method,
        args.func_name,
        args.alpha,
        args.num_topk_docs,
        args.sample_num,
        args.start_index,
        args.sample_start_index,
        args.temp_cpmi,
        candidate_layers,
        args.reweight_logit,
        args.using_norm,
        args.using_entropy

    )
    logger.info("finished running %s", sys.argv[0])
