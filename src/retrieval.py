import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def retrieve_nodes(query_vec, node_embeddings, nodes, mode="cosine",
                   k=5, adjacency=None, type_boost=False):

    results = []

    if mode == "cosine":

        for node in nodes:
            node_id = node["node_id"]
            score = cosine_similarity(query_vec, node_embeddings[node_id])

            if type_boost and node["type"] == "answer":
                score *= 1.1

            results.append((node, score))

        results.sort(key=lambda x: x[1], reverse=True)

    elif mode == "knn":

        matrix = torch.stack([node_embeddings[n["node_id"]] for n in nodes])
        nbrs = NearestNeighbors(n_neighbors=k, metric="cosine")
        nbrs.fit(matrix.numpy())
        distances, indices = nbrs.kneighbors(query_vec.unsqueeze(0).numpy())

        for idx in indices[0]:
            results.append((nodes[idx], 1 - distances[0][0]))

    elif mode == "expansion":

        base = retrieve_nodes(query_vec, node_embeddings, nodes,
                              mode="cosine", k=k)

        expanded = []

        for node, score in base[:k]:

            expanded.append((node, score))

            for neigh in adjacency[node["node_id"]]:
                neigh_score = cosine_similarity(query_vec,
                                                node_embeddings[neigh]) * 0.9
                expanded.append(
                    (next(n for n in nodes if n["node_id"] == neigh),
                     neigh_score)
                )

        expanded.sort(key=lambda x: x[1], reverse=True)
        results = expanded

    return [
        {
            "node_id": node["node_id"],
            "text": node["text"],
            "type": node["type"],
            "score": float(score)
        }
        for node, score in results[:k]
    ]
