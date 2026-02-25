import os
import json

def export_funsd(funsd_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    ann_dir = os.path.join(funsd_path, "adjusted_annotations")

    for idx, file in enumerate(sorted(os.listdir(ann_dir))):

        with open(os.path.join(ann_dir, file)) as f:
            data = json.load(f)

        nodes = []
        edges = []
        form = data["form"]

        id_map = {}

        for i, item in enumerate(form):
            nid = f"n{i}"
            id_map[item["id"]] = nid

            nodes.append({
                "node_id": nid,
                "text": item["text"],
                "type": item["label"],
                "bbox": item["box"]
            })

        for item in form:
            src = id_map[item["id"]]
            for link in item["linking"]:
                dst = id_map[link[1]]
                edges.append({
                    "src": src,
                    "dst": dst,
                    "type": "kv_link"
                })

        page = {
            "doc_id": f"funsd_{idx:03d}",
            "page_id": "0",
            "nodes": nodes,
            "edges": edges
        }

        with open(os.path.join(output_dir, f"funsd_{idx:03d}.json"), "w") as f:
            json.dump(page, f, indent=2)
