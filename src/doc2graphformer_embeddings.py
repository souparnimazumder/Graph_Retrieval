import os
import torch
from tqdm import tqdm

from src.data.graph_builder import GraphBuilder
from src.data.feature_builder import FeatureBuilder
from src.models.graphformer import GraphformerPEneo
from src.training.utils import get_device

EMBED_SAVE_DIR = "data/embeddings/doc2graphformer"
WEIGHTS_PATH = "checkpoints/graphformer-funsd.pt"


def build_doc2graphformer_embeddings_all(image_paths, device_id=-1):

    os.makedirs(EMBED_SAVE_DIR, exist_ok=True)
    device = get_device(device_id)

    print("Creating graphs...")
    gb = GraphBuilder()
    graphs, _, _, features = gb.get_graph(image_paths, "CUSTOM")

    print("Creating features...")
    fb = FeatureBuilder(d=device)
    chunks, _ = fb.add_features(graphs, features)
    input_dim = sum(chunks)

    print("Loading GraphformerPEneo model...")

    model = GraphformerPEneo(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        num_node_classes=4,
        num_edge_classes=2,
        num_grouping_classes=2,
        dropout=0.1
    ).to(device)

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    print("Extracting GraphFormer embeddings...")

    with torch.no_grad():
        for idx, graph in tqdm(enumerate(graphs), total=len(graphs)):

            graph = graph.to(device)
            node_feat = graph.ndata["feat"].to(device)

            # Extract backbone only
            node_repr = model.backbone(node_feat, attn_mask=None)

            node_embeddings = node_repr.detach().cpu()

            doc_name = os.path.basename(image_paths[idx]).split(".")[0]

            torch.save(
                node_embeddings,
                os.path.join(EMBED_SAVE_DIR, f"{doc_name}.pt")
            )

    print(" GraphFormer embeddings saved.")