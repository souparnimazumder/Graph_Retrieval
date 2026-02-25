import json
import random
import os

def generate_queries(exported_dir, num_queries=100):

    files = os.listdir(exported_dir)
    queries = []

    for _ in range(num_queries):

        file = random.choice(files)
        with open(os.path.join(exported_dir, file)) as f:
            page = json.load(f)

        answer_nodes = [n for n in page["nodes"] if n["type"] == "answer"]
        if not answer_nodes:
            continue

        target = random.choice(answer_nodes)

        queries.append({
            "query": f"What is the value for {target['text']}?",
            "doc_id": page["doc_id"],
            "page_id": page["page_id"],
            "target_node_id": target["node_id"]
        })

    os.makedirs("data/queries", exist_ok=True)
    with open("data/queries/queries.json", "w") as f:
        json.dump(queries, f, indent=2)
