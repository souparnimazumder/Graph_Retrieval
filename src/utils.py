import os
import json
import torch
import numpy as np


# -----------------------------
# Device handling
# -----------------------------
def get_device(device_id=-1):
    if device_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


# -----------------------------
# JSON helpers
# -----------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------
# Cosine similarity
# -----------------------------
def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()


# -----------------------------
# IoU (for gold matching)
# -----------------------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)


# -----------------------------
# Build adjacency dict
# -----------------------------
def build_adjacency(page_json):
    adjacency = {n["node_id"]: [] for n in page_json["nodes"]}

    for edge in page_json["edges"]:
        adjacency[edge["src"]].append(edge["dst"])
        adjacency[edge["dst"]].append(edge["src"])

    return adjacency
