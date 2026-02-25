import os
import torch
import json
from tqdm import tqdm

from src.data.graph_builder import GraphBuilder
from src.data.feature_builder import FeatureBuilder
from src.models.doc2graph import SetModel
from src.training.utils import get_device

EMBED_SAVE_DIR = "data/embeddings/doc2graph"
WEIGHTS_PATH = "checkpoints/e2e-funsd.pt"  


def build_doc2graph_embeddings_all(image_paths, device_id=-1):

    os.makedirs(EMBED_SAVE_DIR, exist_ok=True)
    device = get_device(device_id)

    print("Creating graphs...")
    gb = GraphBuilder()
    graphs, _, _, features = gb.get_graph(image_paths, "CUSTOM")

    print("Creating features...")
    fb = FeatureBuilder(d=device)
    chunks, _ = fb.add_features(graphs, features)
    input_dim = sum(chunks)

    print("Loading Doc2Graph model...")
    model_wrapper = SetModel(name="e2e", device=device)
    model = model_wrapper.get_model(
        nodes=4,  # FUNSD
        edges=2,
        chunks=chunks,
        verbatim=False
    )

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    print("Extracting embeddings...")

    with torch.no_grad():
        for idx, graph in tqdm(enumerate(graphs), total=len(graphs)):

            graph = graph.to(device)
            h = graph.ndata["feat"].to(device)

            # --- Forward manually until message passing ---
            h = model.projector(h)

            # For E2E (single layer)
            h = model.message_passing(graph, h)

            node_embeddings = h.detach().cpu()

            doc_name = os.path.basename(image_paths[idx]).split(".")[0]

            torch.save(
                node_embeddings,
                os.path.join(EMBED_SAVE_DIR, f"{doc_name}.pt")
            )

    print("Doc2Graph embeddings saved.")