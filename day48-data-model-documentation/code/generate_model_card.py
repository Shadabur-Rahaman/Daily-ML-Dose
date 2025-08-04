import json

def generate_model_card(metadata):
    with open("model_card.md", "w") as f:
        f.write(f"# Model Card: {metadata['name']}\n\n")
        f.write(f"## Overview\n{metadata['overview']}\n\n")
        f.write("## Performance\n")
        for metric, value in metadata["metrics"].items():
            f.write(f"- **{metric}**: {value}\n")
        f.write("\n## Limitations\n")
        f.write(metadata["limitations"])

# Sample usage
if __name__ == "__main__":
    with open("model_metadata.json") as meta_file:
        metadata = json.load(meta_file)
    generate_model_card(metadata)
