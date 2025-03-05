## Entropy-Based Decoding for Retrieval-Augmented Large Language Models

This repo provides the source code and data of our paper.

### Requirements

Please install the main requirements by running `pip install -r requirements.txt`.

### Response Generation and Evaluation

1. Download all the processed data from the [link](https://drive.google.com/drive/folders/1zsrCbw8T7Q2ZsW0MGt3Kwvg4N3z9rqmn?usp=sharing) under `processed_data` and place it in the `./processed_data/` folder.
2. To reproduce our method's experimental results in open-domain QA (Table 1 of the paper), please refer to `./scripts/odqa_scripts/`.
3. To reproduce our method's experimental results on synthetic data (Figure 2 of the paper), please refer to `./scripts/synthetic_scripts/`.

If you want to construct retrieval data from scratch, you first need to download four original QA datasets from this [link](https://drive.google.com/drive/folders/1zsrCbw8T7Q2ZsW0MGt3Kwvg4N3z9rqmn?usp=sharing) and place them in the `original_data` folder. Then refer to `./original_data/download_data.sh` to download Wikipedia passages, and then refer to `./scripts/retrieval_scripts/` to construct retrieval-augmented queries.



### Acknowledgements

Our code primarily refers to [NBCE](https://github.com/bojone/NBCE) and [lost-in-the-middle](https://github.com/nelson-liu/lost-in-the-middle). Thanks for their awesome implementations.

