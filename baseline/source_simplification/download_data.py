import argparse
import sys
from huggingface_hub import snapshot_download
from datasets import load_dataset

DATASETS = [
    ("G4KMU/t2-ragbench", "ConvFinQA"),
    ("G4KMU/t2-ragbench", "FinQA"),
    ("G4KMU/t2-ragbench", "TAT-DQA")
]
MODEL = "intfloat/multilingual-e5-large-instruct"

def download_dataset(repo_id, config_name, token):
    print(f"Downloading dataset: {repo_id} ({config_name})")
    try:
        load_dataset(repo_id, config_name, token=token)
    except Exception as e:
        print(f"Lỗi khi tải {repo_id} [{config_name}]: {str(e)}")
        sys.exit(1)

def download_model(repo_id, token):
    print(f"Downloading model: {repo_id}")
    try:
        snapshot_download(repo_id=repo_id, repo_type="model", token=token)
    except Exception as e:
        print(f"Lỗi khi tải {repo_id}: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="T2-RAGBench Data Download")
    parser.add_argument("-m", "--mode", type=int, choices=[1, 2, 3], required=True,
                        help="1: All, 2: Only Datasets, 3: Only Model")
    parser.add_argument("-t", "--token", type=str, required=False,
                        help="Hugging Face User Access Token (optional if public)")
    
    args = parser.parse_args()

    if args.mode == 1:
        for repo_id, config_name in DATASETS:
            download_dataset(repo_id, config_name, args.token)
        download_model(MODEL, args.token)
    elif args.mode == 2:
        for repo_id, config_name in DATASETS:
            download_dataset(repo_id, config_name, args.token)
    elif args.mode == 3:
        download_model(MODEL, args.token)

if __name__ == "__main__":
    main()
