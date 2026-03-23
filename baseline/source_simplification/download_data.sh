#!/bin/bash

# Configuration
DATASETS=("G4KMU/finqa-german" "G4KMU/convfinqa" "G4KMU/tat-dqa")
MODEL="intfloat/multilingual-e5-large-instruct"

# Alias để gọi CLI qua python nếu lệnh trực tiếp fail
HF_COMMAND="python3 -m huggingface_hub.commands.huggingface_cli"

echo "Starting T2-RAGBench Data Download Script"

download_dataset() {
    echo "Downloading dataset: $1..."
    $HF_COMMAND download --repo-type dataset "$1"
}

download_model() {
    echo "Downloading embedding model: $1..."
    $HF_COMMAND download "$1"
}

# Lấy lựa chọn từ tham số dòng lệnh ($1), nếu không có thì mặc định là 1
choice=${1:-1}

case $choice in
    1)
        for ds in "${DATASETS[@]}"; do download_dataset "$ds"; done
        download_model "$MODEL"
        ;;
    2)
        for ds in "${DATASETS[@]}"; do download_dataset "$ds"; done
        ;;
    3)
        download_model "$MODEL"
        ;;
    *)
        echo "Invalid choice or usage: !bash download_data.sh [1-3]"
        exit 1
        ;;
esac

echo "Finished!"