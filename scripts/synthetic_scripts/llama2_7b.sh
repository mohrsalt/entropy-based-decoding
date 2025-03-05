# top_10_documents
## First, generate the responses
# the proposed decoding is document-position-invariant, gold_at_x (varying x) does not have influence on the decoding
CUDA_VISIBLE_DEVICES=0 python -u get_qa_responses.py \
    --input-path ./processed_data/synthetic_data/10_total_documents/nq-open-10_total_documents_gold_at_2.jsonl.gz \
    --max-new-tokens 100 \
    --num-gpus 1 \
    --model "meta-llama/Llama-2-7b-hf" \
    --max-prompt-len 4096 \
    --beta 5.0 \
    --metric-criterion weighted_entropy \
    --sampling-method greedy \
    --func_name logits \
    --temp-cpmi 0.1 \
    --candidate-layers "[18,20,22,24,26,28,30,32]" \
    --output-path ./output/synthetic/llama2-7b/nq-top10.jsonl.gz

## Second, evaluate the responses
python -u evaluate_qa_responses.py \
    --input-path ./output/synthetic/llama2-7b/nq-top10.jsonl.gz


# top_20_documents
CUDA_VISIBLE_DEVICES=0 python -u get_qa_responses.py \
    --input-path ./processed_data/synthetic_data/20_total_documents/nq-open-20_total_documents_gold_at_2.jsonl.gz \
    --max-new-tokens 100 \
    --num-gpus 1 \
    --model "meta-llama/Llama-2-7b-hf" \
    --max-prompt-len 4096 \
    --beta 5.0 \
    --metric-criterion weighted_entropy \
    --sampling-method greedy \
    --func_name logits \
    --temp-cpmi 0.1 \
    --using-entropy \
    --candidate-layers "[18,20,22,24,26,28,30,32]" \
    --output-path ./output/synthetic/llama2-7b/nq-top20.jsonl.gz

## Second, evaluate the responses
python -u evaluate_qa_responses.py \
    --input-path ./output/synthetic/llama2-7b/nq-top20.jsonl.gz