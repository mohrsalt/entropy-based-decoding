
from utils.retrieval_utils import normalize_query
import csv
import faiss,pickle        
import numpy as np 
from tqdm import tqdm
from transformers import DPRQuestionEncoder,DPRQuestionEncoderTokenizer,BertModel,BertTokenizer
import torch
from utils.tokenizers import SimpleTokenizer
import unicodedata
import time
import transformers
transformers.logging.set_verbosity_error()
import jsonlines
import os 
import json

def load_natural_questions(file_name, encoding_batch_size):
    questions_list = [] 
    answers_list = []
    with jsonlines.open(file_name, "r") as f_reader:
        for cur_json_obj in f_reader:
            questions_list.append(normalize_query(cur_json_obj['question']))
            answers_list.append(cur_json_obj['answer'])
    questions_list = [questions_list[idx:idx+encoding_batch_size] for idx in range(0,len(questions_list),encoding_batch_size)]
    return [questions_list, answers_list]

def load_pop_qa(file_name, encoding_batch_size):
    questions_list = [] 
    answers_list = []
    with jsonlines.open(file_name, "r") as f_reader:
        for cur_json_obj in f_reader:
            questions_list.append(normalize_query(cur_json_obj['question']))
            answers_list.append(cur_json_obj['answer'])
    questions_list = [questions_list[idx:idx+encoding_batch_size] for idx in range(0,len(questions_list),encoding_batch_size)]
    return [questions_list, answers_list]


def load_web_qa(file_name, encoding_batch_size):
    questions_list = []
    answers_list = [] 
    from pandas import read_parquet
    data_df = read_parquet(file_name)
    for index, row in data_df.iterrows():
        questions_list.append(row['question'])
        # assert isinstance(row['answer']['aliases'], list)
        answers_list.append(row['answers'].tolist())
    questions_list = [questions_list[idx:idx+encoding_batch_size] for idx in range(0,len(questions_list),encoding_batch_size)]
    return [questions_list, answers_list]

def load_trivia_qa(file_name, encoding_batch_size):
    questions_list = []
    answers_list = [] 
    aliases_list = [] 
    from pandas import read_parquet
    data_df = read_parquet(file_name)
    for index, row in data_df.iterrows():
        questions_list.append(normalize_query(row['question']))
        assert isinstance(row['answer']['value'],str)
        # assert isinstance(row['answer']['aliases'], list)
        answers_list.append([row['answer']['value']])
        aliases_list.append(row['answer']['aliases'].tolist())
    questions_list = [questions_list[idx:idx+encoding_batch_size] for idx in range(0,len(questions_list),encoding_batch_size)]
    return [questions_list, answers_list, aliases_list]




def has_answer(answers,doc):
    tokenizer = SimpleTokenizer()
    doc = tokenizer.tokenize(normalize(doc)).words(uncased=True)
    for answer in answers:
        answer = tokenizer.tokenize(normalize(answer)).words(uncased=True)
        for i in range(0, len(doc) - len(answer) + 1):
                if answer == doc[i : i + len(answer)]:
                    return True
    return False


def normalize(text):
    return unicodedata.normalize("NFD", text)

def embed_queries(questions_list):
    query_embeddings = []
    for query in tqdm(questions_list, desc="encoding queries..."):
        with torch.no_grad():
            query_embedding = query_encoder(**tokenizer(query,max_length=256, truncation=True,
                                            padding="max_length", return_tensors="pt").to(device))
        if isinstance(query_encoder, DPRQuestionEncoder):
            query_embedding = query_embedding.pooler_output
        else:
            query_embedding = query_embedding.last_hidden_state[:,0,:]
        query_embeddings.append(query_embedding.cpu().detach().numpy())
    query_embeddings = np.concatenate(query_embeddings, axis=0) # (#query, 768)
    return query_embeddings

def search_topk(index, query_embeddings, top_k):
    start_time = time.time()
    D, I = index.search(query_embeddings, top_k)
    print(f"takes {time.time() - start_time} s")
    return D, I
    

def save_data(D, I, data, wiki_passages, dataset_name, save_filename):

    def print_topk_accuracy(D, I, answers_list, aliases_list, wiki_passages):
        if aliases_list is not None:
            for idx in range(len(answers_list)):
                answers_list[idx] = answers_list[idx] + aliases_list[idx]

        hit_lists = [] 
        for answer_list, id_list in tqdm(zip(answers_list,I), total=len(answers_list), desc="calculating metrics..."):
            # process single query
            hit_list = [] 
            for doc_id in id_list:
                doc = wiki_passages[str(doc_id)][0]
                hit_list.append(has_answer(answer_list, doc)) 
            hit_lists.append(hit_list) # sample of hit_list: [True, False, False, True, False

        top_k = 100
        top_k_hits = [0]*top_k
        for hit_list in hit_lists:
            best_hit = next((i for i, x in enumerate(hit_list) if x), None)
            if best_hit is not None:
                top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
        topk_ratio = [x/len(answers_list) for x in top_k_hits]
        for idx in range(top_k):
            if (idx+1) % 10 == 0:
                print(f"top-{idx+1} accuracy",topk_ratio[idx])
                print("-------------------------------------------\n")

    if dataset_name == "nq":
        json_list = []
        batch_questions_list = data[0]
        questions_list = []
        for batch_questions in batch_questions_list:
            questions_list = questions_list + batch_questions
        answers_list = data[1]
        print_topk_accuracy(D,I, answers_list, None, wiki_passages)

        assert len(questions_list) == len(answers_list) == len(D) == len(I)
        
        for question, answers, scores, doc_ids in tqdm(zip(questions_list, answers_list, D, I), total=len(answers_list), 
                                                        desc="saving data..."):
            docs = [wiki_passages[str(i)] for i in doc_ids]
            doc_contents = [ele[0] for ele in docs]
            doc_titles = [ele[1] for ele in docs]
            json_list.append({'question': question, 'answers': answers, "doc_scores": scores.tolist(), "doc_contents": doc_contents, "doc_titles": doc_titles})
        save_dir = "./data_with_passages/nq"
        os.makedirs(save_dir,exist_ok=True)
        with jsonlines.open(os.path.join(save_dir, save_filename), "w") as f_writer:
            for cur_json_obj in json_list:
                f_writer.write(cur_json_obj)

    elif dataset_name == "webqa":
        json_list = []
        batch_questions_list = data[0]
        questions_list = []
        for batch_questions in batch_questions_list:
            questions_list = questions_list + batch_questions
        answers_list = data[1]
        print_topk_accuracy(D,I, answers_list, None, wiki_passages)

        assert len(questions_list) == len(answers_list) == len(D) == len(I)
        
        for question, answers, scores, doc_ids in tqdm(zip(questions_list, answers_list, D, I), total=len(answers_list), 
                                                        desc="saving data..."):
            docs = [wiki_passages[str(i)] for i in doc_ids]
            doc_contents = [ele[0] for ele in docs]
            doc_titles = [ele[1] for ele in docs]
            json_list.append({'question': question, 'answers': answers, "doc_scores": scores.tolist(), "doc_contents": doc_contents, "doc_titles": doc_titles})
        save_dir = "./data_with_passages/webqa"
        os.makedirs(save_dir,exist_ok=True)
        with jsonlines.open(os.path.join(save_dir, save_filename), "w") as f_writer:
            for cur_json_obj in json_list:
                f_writer.write(cur_json_obj)

    elif dataset_name == "popqa":
        json_list = []
        batch_questions_list = data[0]
        questions_list = []
        for batch_questions in batch_questions_list:
            questions_list = questions_list + batch_questions
        answers_list = data[1]
        print_topk_accuracy(D,I, answers_list, None, wiki_passages)

        assert len(questions_list) == len(answers_list) == len(D) == len(I)
        
        for question, answers, scores, doc_ids in tqdm(zip(questions_list, answers_list, D, I), total=len(answers_list), 
                                                        desc="saving data..."):
            docs = [wiki_passages[str(i)] for i in doc_ids]
            doc_contents = [ele[0] for ele in docs]
            doc_titles = [ele[1] for ele in docs]
            json_list.append({'question': question, 'answers': answers, "doc_scores": scores.tolist(), "doc_contents": doc_contents, "doc_titles": doc_titles})
        save_dir = "./data_with_passages/popqa"
        os.makedirs(save_dir,exist_ok=True)
        with jsonlines.open(os.path.join(save_dir, save_filename), "w") as f_writer:
            for cur_json_obj in json_list:
                f_writer.write(cur_json_obj)

    elif dataset_name == 'tqa':
        json_list = []
        batch_questions_list = data[0]
        questions_list = []
        for batch_questions in batch_questions_list:
            questions_list = questions_list + batch_questions
        answers_list = data[1]
        aliases_list = data[2]
        print_topk_accuracy(D,I, answers_list, aliases_list, wiki_passages)
        assert len(questions_list) == len(answers_list) == len(aliases_list) == len(D) == len(I)
        for question, answers, aliases, scores, doc_ids in tqdm(zip(questions_list, answers_list, aliases_list, D, I), total=len(answers_list), 
                                                        desc="saving data..."):
            docs = [wiki_passages[str(i)] for i in doc_ids]
            doc_contents = [ele[0] for ele in docs]
            doc_titles = [ele[1] for ele in docs]
            json_list.append({'question': question, 'answers': answers, "aliases": aliases, "doc_scores": scores.tolist(), "doc_contents": doc_contents, "doc_titles": doc_titles})
        save_dir = "./data_with_passages/tqa"
        os.makedirs(save_dir,exist_ok=True)
        with jsonlines.open(os.path.join(save_dir, save_filename), "w") as f_writer:
            for cur_json_obj in json_list:
                f_writer.write(cur_json_obj)


if __name__ == "__main__":
     

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikipedia_path", default="./original_data/wikipedia_dump/psgs_w100.tsv", type=str)
    parser.add_argument("--dataset_name", default="nq", type=str)
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--encoding_batch_size",type=int,default=8)
    parser.add_argument("--num_docs",type=int,default=21015324) # 21M
    parser.add_argument("--pretrained_model_path",required=True)
    parser.add_argument("--topk", default=50, type=int, required=True )
    args = parser.parse_args()

    ## load QA dataset
    # qa_dir = DATASET_MAP[args.dataset_name]
    if args.dataset_name == "nq":
        train_data = load_natural_questions(os.path.join(args.data_dir, "NaturalQuestions.resplit.train.jsonl"), encoding_batch_size=args.encoding_batch_size)
        dev_data = load_natural_questions(os.path.join(args.data_dir, "NaturalQuestions.resplit.dev.jsonl"), encoding_batch_size=args.encoding_batch_size)
        test_data = load_natural_questions(os.path.join(args.data_dir, "NaturalQuestions.resplit.test.jsonl"),
                                           encoding_batch_size=args.encoding_batch_size)
    elif args.dataset_name == "tqa":
        train_data = load_trivia_qa(os.path.join(args.data_dir, "train-00000-of-00001.parquet"),
                                    encoding_batch_size=args.encoding_batch_size)
        dev_data = load_trivia_qa(os.path.join(args.data_dir, "validation-00000-of-00001.parquet"),
                                  encoding_batch_size=args.encoding_batch_size)
        test_data = None 
    elif args.dataset_name == "webqa":
        test_data = load_web_qa(os.path.join(args.data_dir, "test-00000-of-00001.parquet"),
                                  encoding_batch_size=args.encoding_batch_size) 
    elif args.dataset_name == "popqa":
        test_data = load_pop_qa(os.path.join(args.data_dir, "popqa_full_test.jsonl"),
                                    encoding_batch_size=args.encoding_batch_size)


    # load 21M wikipedia passages 
    id_col, text_col, title_col = 0,1,2
    wiki_passages = {}
    with open(args.wikipedia_path, 'r') as f:
    
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader, total=args.num_docs, desc="loading 21M wikipedia passages..."):
            if row[id_col] == "id": 
                continue
            else:
                # wiki_passages.append(row[text_col].strip('"'))
                wiki_passages[str(row[id_col])] = [row[text_col].strip('"'), row[title_col].strip('"')]
    # assert "284453" in wiki_passages.keys()

    embedding_dimension = 768
    orig_index = faiss.IndexFlatIP(embedding_dimension)
    index = faiss.IndexIDMap(orig_index)
    for idx in tqdm(range(50), desc="building index from embedding..."):
        embed_path = "./original_data/wikipedia_dump/embedding_files/wiki_passages_{}.pkl"
        embed_path = embed_path.format(idx)
        data = pickle.load(open(embed_path, 'rb'))
        embeddings = [i[1] for i in data]
        embeddings = np.stack(embeddings, axis=0)
        ids = np.array([int(i[0]) for i in data])
        index.add_with_ids(embeddings, ids)

    # load query encoder 
    if args.pretrained_model_path == "facebook/dpr-question_encoder-single-nq-base":
        query_encoder = DPRQuestionEncoder.from_pretrained(args.pretrained_model_path)
        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.pretrained_model_path)
    else:
        query_encoder = BertModel.from_pretrained(args.pretrained_model_path,add_pooling_layer=False)
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_encoder.to(device).eval()


    if args.dataset_name == "nq":

        test_query_embeddings = embed_queries(test_data[0])
        test_D, test_I = search_topk(index, test_query_embeddings, args.topk)
        save_data(test_D, test_I, test_data, wiki_passages, args.dataset_name, "nq_test_with_passages.jsonl")
    
        dev_query_embeddings = embed_queries(dev_data[0])
        dev_D, dev_I = search_topk(index, dev_query_embeddings, args.topk)
        save_data(dev_D, dev_I, dev_data, wiki_passages, args.dataset_name, "nq_dev_with_passages.jsonl")

        train_query_embeddings = embed_queries(train_data[0])
        train_D, train_I = search_topk(index, train_query_embeddings, args.topk)
        save_data(train_D, train_I, train_data, wiki_passages, args.dataset_name, "nq_train_with_passages.jsonl")

    

    elif args.dataset_name == "tqa":
        dev_query_embeddings = embed_queries(dev_data[0])
        dev_D, dev_I = search_topk(index, dev_query_embeddings, args.topk)

        # save
        save_data(dev_D, dev_I, dev_data, wiki_passages, args.dataset_name, "tqa_dev_with_passages.jsonl")
        train_query_embeddings = embed_queries(train_data[0])
        train_D, train_I = search_topk(index, train_query_embeddings, args.topk)
        save_data(train_D, train_I, train_data, wiki_passages, args.dataset_name, "tqa_train_with_passages.jsonl")

    elif args.dataset_name == "webqa":
        test_query_embeddings = embed_queries(test_data[0])
        test_D, test_I = search_topk(index, test_query_embeddings, args.topk)

        # save
        save_data(test_D, test_I, test_data, wiki_passages, args.dataset_name, "webqa_test_with_passages.jsonl")
    
    elif args.dataset_name == "popqa":
        test_query_embeddings = embed_queries(test_data[0])
        test_D, test_I = search_topk(index, test_query_embeddings, args.topk)

        # save
        save_data(test_D, test_I, test_data, wiki_passages, args.dataset_name, "popqa_full_test_with_passages.jsonl")

    else:
        raise NotImplementedError
                
            
         

    
