def load_metadata(path):
    import json
    with open(path, "r") as f:
        return json.load(f)

def format_metrics(metrics_dict):
    return "\n".join([f"- **{k}**: {v}" for k, v in metrics_dict.items()])
