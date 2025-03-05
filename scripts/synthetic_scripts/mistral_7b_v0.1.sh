# Take "NQ, --num-topk-docs=5" as an example, for other datasets and topk setting, just change the input-path and --num-topk-docs accordingly.
## First, generate the responses
CUDA_VISIBLE_DEVICES=0 python -u get_qa_responses.py \
    --input-path ./processed_data/synthetic_data/10_total_documents/nq-open-10_total_documents_gold_at_2.jsonl.gz \
    --max-new-tokens 100 \
    --num-gpus 1 \
    --model "mistralai/Mistral-7B-v0.1" \
    --max-prompt-len 4096 \
    --beta 0.25 \
    --metric-criterion weighted_entropy \
    --sampling-method greedy \
    --func_name logits \
    --temp-cpmi 0.1 \
    --using-entropy \
    --candidate-layers "[18,20,22,24,26,28,30,32]" \
    --output-path ./output/synthetic/mistral-7b-v0.1/nq-top5.jsonl.gz

## Second, evaluate the responses
python -u evaluate_qa_responses.py \
    --input-path ./output/synthetic/mistral-7b-v0.1/nq-top5.jsonl.gz
