# download text
python download_data.py --resource data.wikipedia_split.psgs_w100 --output_dir ./wikipedia_dump/

# donwload embeddings
python download_data.py --resource data.retriever_results.nq.single.wikipedia_passages --output_dir ./wikipedia_dump/embedding_files/