import json
import torch
from src.retrieval import retrieve_nodes

def compute_metrics(rankings, target_id):

    recall1 = int(rankings[0]["node_id"] == target_id)

    recall5 = int(any(r["node_id"] == target_id for r in rankings[:5]))

    mrr = 0
    for i, r in enumerate(rankings):
        if r["node_id"] == target_id:
            mrr = 1 / (i + 1)
            break

    return recall1, recall5, mrr
