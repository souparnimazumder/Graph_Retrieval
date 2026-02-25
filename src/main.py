# run_pipeline

import os
from graphs import export_funsd
from base_embeddings import build_base_embeddings_all
from doc2graph_embeddings_embeddings import build_doc2graph_embeddings_all
from graphformer_embeddings import build_graphformer_embeddings_all
from query_generator import generate_queries
#from gold_builder import build_gold_targets
from evaluation import run_full_evaluation

DATA_ROOT = "data/FUNSD"
EXPORTED = "data/exported_graphs"

def main():

    print("Exporting FUNSD graphs...")
    export_funsd(DATA_ROOT, EXPORTED)

    print("Building Base Embeddings...")
    base_embeddings_all(EXPORTED)

    print("Building Doc2graph Embeddings...")
    doc2graph_embeddings_all(EXPORTED)

    print("Building GraphFormer Embeddings...")
    graphformer_embeddings_all(EXPORTED)

    print("Generating 100 queries...")
    generate_queries(EXPORTED, num_queries=100)

    #print("Building gold targets...")
    #build_gold_targets(EXPORTED)

    print("Running retrieval evaluation...")
    run_full_evaluation()

if __name__ == "__main__":
    main()
