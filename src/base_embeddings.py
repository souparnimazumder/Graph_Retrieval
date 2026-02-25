from sentence_transformers import SentenceTransformer
import numpy as np
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_base_embeddings(page_json):

    embeddings = {}

    for node in page_json["nodes"]:
        vec = model.encode(node["text"])
        embeddings[node["node_id"]] = vec

    return embeddings
