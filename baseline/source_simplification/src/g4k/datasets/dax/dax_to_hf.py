"""Tranform the DAX dataset to a Huggingface dataset."""

from g4k.file_manager import FileManager
from g4k.huggingface import HuggingfaceDatasetHandler


def main() -> None:
    """Main function for dataset creation.

    Note: The dataset is available to members of the Huggingface organization G4KMU.
    You can access it here: https://huggingface.co/datasets/g4kmu/dax_german
    """
    print("Transforming dataset to HF and push it to hf...")
    dataset = FileManager("../data/dax/dax_raw.jsonl").load_jsonlines()
    dataset = transform_dax(dataset)

    handler = HuggingfaceDatasetHandler("g4kmu/dax_german", dataset, "train")
    handler.push_to_hf(private=True)
    print("Done.")


def transform_dax(dataset: list[dict]) -> list[dict]:
    """Transform the DAX dataset to a Huggingface dataset."""
    return [
        {"metric": line.get("Kennzahlen"), "company": line.get("Unternehmen"), "raw": str(line)}
        for line in dataset
    ]


if __name__ == "__main__":
    main()
