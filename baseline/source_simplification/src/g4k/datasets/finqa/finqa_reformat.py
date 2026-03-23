"""Translates the Flare-FinQA dataset to German."""

import argparse
import random
import time

import pandas as pd
from googletrans import Translator
from tqdm import tqdm

from g4k.file_manager import FileManager


def transform_finqa(json_data: str, df_company: pd.DataFrame) -> pd.DataFrame:
    """Transforms the FinQA dataset to the required format."""
    df = pd.json_normalize(json_data)
    df_final = pd.DataFrame(
        columns=[
            "id",
            "question",
            "answer",
            "program_solution",
            "pre_text",
            "post_text",
            "table",
            "context",
            "report_year",
            "page_number",
        ]
    )
    df_final["question"] = df["qa.question"]
    df_final["answer"] = df["qa.answer"]
    df_final["program_solution"] = df["qa.program_re"]
    df_final["pre_text"] = df["pre_text"].apply(lambda x: "\n".join(x))
    df_final["post_text"] = df["post_text"].apply(lambda x: "\n".join(x))
    df_final["table"] = df["table"].apply(
        lambda row: pd.DataFrame(row[1:], columns=row[0]).to_markdown()
    )
    df_final["context"] = df_final[["pre_text", "table", "post_text"]].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1
    )
    df_final["report_year"] = df["filename"].apply(lambda x: x.split("/")[1])
    df_final["page_number"] = df["filename"].apply(
        lambda x: int(x.split("/page_")[1].split(".")[0])
    )
    df_final["company_symbol"] = df["filename"].apply(lambda x: x.split("/")[0])
    df_final = pd.merge(df_final, df_company, on="company_symbol")
    df_final["id"] = "finqa" + df_final.index.astype(str)
    return df_final


def prepare_dataset(
    data_path: str,
    company_data_path: str,
) -> pd.DataFrame:
    """Loads the dataset and prepares it for processing."""
    json_data = FileManager(data_path).load_json()
    company_df = pd.read_csv(company_data_path)
    company_df.columns = [
        "company_symbol",
        "company_name",
        "company_sector",
        "company_industry",
        "company_headquarters",
        "company_date_added",
        "company_cik",
        "company_founded",
    ]
    return transform_finqa(json_data, company_df)


def load_dataset(df: pd.DataFrame, output_file_path: str, translation_cols: list) -> pd.DataFrame:
    """Loads the existing dataset and merges it with the current dataset."""
    # If CSV exists, merge it with current dataset
    try:
        df_existing = pd.read_csv(output_file_path)
        df = df.merge(df_existing, on="id", how="left", suffixes=("", "_existing"))
    except FileNotFoundError:
        for col in translation_cols:
            df[col] = None

    return df


def translate_text(
    text: str,
    translator: Translator,
    src_lang: str = "en",
    dest_lang: str = "de",
    chunk_size: int = 4900,
) -> str | None:
    """Translates the given text into the destination language."""
    if len(text) > chunk_size:
        translated_parts = []
        for i in range(0, len(text), chunk_size):
            part = text[i : i + chunk_size]
            translated_part = translator.translate(part, src=src_lang, dest=dest_lang).text
            translated_parts.append(translated_part)
        return "".join(translated_parts)
    else:
        return str(translator.translate(text, src=src_lang, dest=dest_lang).text)


def translate_row(
    index: int, row: pd.Series, df: pd.DataFrame, translator: Translator, translation_cols: list
) -> None:
    """Translates the question and context of a single row."""
    # Translate question if not already translated
    for col in translation_cols:
        if pd.isna(row[col]):
            try:
                df.loc[index, col] = translate_text(row[col.split("_")[0]], translator)
            except Exception as e:
                print(f"Error translating {col} at index {index}: {e}")
                df.loc[index, col] = None


def save_to_csv(df: pd.DataFrame, csv_file: str) -> None:
    """Saves the dataframe to a CSV file."""
    df.to_csv(csv_file, index=False)


def parse_args() -> argparse.Namespace:
    """Parses the command-line arguments."""
    parser = argparse.ArgumentParser(description="Translates the FinQA dataset to German.")
    parser.add_argument(
        "--json_file",
        type=str,
        default="dev.json",
        help="The path to the json file to save the translated dataset.",
    )
    parser.add_argument(
        "--company_data_path",
        type=str,
        default="company_data.csv",
        help="The path to the company data CSV file.",
    )
    parser.add_argument(
        "--output_data_path",
        type=str,
        default="finqa_de.csv",
        help="The path to save the translated dataset.",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Translates the dataset to German."""
    translation_cols = ["question_de", "context_de"]
    df = prepare_dataset(args.json_file, args.company_data_path)
    df = load_dataset(df, args.output_data_path, translation_cols)
    translator = Translator()

    for index, row in tqdm(df.iterrows(), total=len(df)):
        time.sleep(random.randint(200, 500) / 1000)  # Sleep for a random duration
        translate_row(index, row, df, translator, translation_cols)
        save_to_csv(df, args.output_data_path)  # Save after each row


if __name__ == "__main__":
    args = parse_args()
    main(args)
