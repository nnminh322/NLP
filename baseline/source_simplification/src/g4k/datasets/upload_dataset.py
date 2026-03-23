"""Upload a dataset to Huggingface Datasets Hub."""

import argparse

import pandas as pd

from g4k.huggingface import HuggingfaceDatasetHandler


def main(args: argparse.Namespace) -> None:
    """Uploads the dataset to Huggingface Datasets Hub."""
    df = pd.read_csv(args.path)
    handler = HuggingfaceDatasetHandler(args.dataset_name, df, dataset_split=args.split)
    handler.push_to_hf(private=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="G4KMU/finqa-german",
        help="The name of the dataset to upload.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="~/genial4kmu/data/pixiu/finqa_de_train.csv",
        help="The path to the dataset file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="The name of the dataset split.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
