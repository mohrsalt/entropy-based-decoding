# retrive top-k documents for each query
cd .. 
CUDA_VISIBLE_DEVICES=0 python generate_topk_passages.py \
    --pretrained_model_path facebook/dpr-question_encoder-single-nq-base \
    --dataset_name "nq" \
    --data_dir ./original_data/nq \
    --topk 100

CUDA_VISIBLE_DEVICES=0 python generate_topk_passages.py \
    --pretrained_model_path facebook/dpr-question_encoder-single-nq-base \
    --dataset_name "tqa" \
    --data_dir ./original_data/tqa \
    --topk 100

CUDA_VISIBLE_DEVICES=0 python generate_topk_passages.py \
    --pretrained_model_path facebook/dpr-question_encoder-single-nq-base \
    --dataset_name "webqa" \
    --data_dir ./original_data/webqa \
    --topk 100

CUDA_VISIBLE_DEVICES=0 python generate_topk_passages.py \
    --pretrained_model_path facebook/dpr-question_encoder-single-nq-base \
    --dataset_name "popqa" \
    --data_dir ./original_data/popqa \
    --topk 100


