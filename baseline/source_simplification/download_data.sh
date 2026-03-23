#!/bin/bash

# Configuration
DATASETS=("G4KMU/finqa-german" "G4KMU/convfinqa" "G4KMU/tat-dqa")
MODEL="intfloat/multilingual-e5-large-instruct"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}Error: huggingface-cli could not be found.${NC}"
    echo "Please install it using: pip install huggingface-hub"
    exit 1
fi

echo -e "${GREEN}Starting T2-RAGBench Data Download Script${NC}"
echo "----------------------------------------"

# Function to download dataset
download_dataset() {
    local ds=$1
    echo -e "${GREEN}Downloading dataset: $ds...${NC}"
    huggingface-cli download --repo-type dataset "$ds"
}

# Function to download model
download_model() {
    local mod=$1
    echo -e "${GREEN}Downloading embedding model: $mod...${NC}"
    huggingface-cli download "$mod"
}

# Menu
echo "Choose what to download:"
echo "1) All (Datasets + Embedding Model)"
echo "2) Only Datasets"
echo "3) Only Embedding Model"
echo "4) Specific Dataset"
echo "q) Quit"

read -p "Enter choice [1-4, q]: " choice

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
    4)
        echo "Choose dataset:"
        select ds in "${DATASETS[@]}"; do
            if [ -n "$ds" ]; then
                download_dataset "$ds"
                break
            else
                echo "Invalid selection"
            fi
        done
        ;;
    q)
        exit 0
        ;;
    *)
        echo "Invalid choice"
        ;;
esac

echo -e "${GREEN}Finished!${NC}"
