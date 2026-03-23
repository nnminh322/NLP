"""A script for downloading DAX dataset."""

import argparse
import concurrent.futures
import logging
import os
import random
import re
import time
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

from g4k.file_manager import FileManager

logger = logging.getLogger(__name__)


def get_links(url: str) -> pd.DataFrame:
    """Fetches a table of data from the specified URL and extracts links."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find(class_="ath_table")
    if table and isinstance(table, Tag):
        list_table = table_data_text(table)
        df = pd.DataFrame(list_table[1:], columns=list_table[0])
        # Only get href from elements that are Tags (not NavigableString or other PageElement types)
        df["links"] = [
            a.get("href")
            for a in table.find_all("a", href=re.compile("^https?://www.boe"))
            if isinstance(a, Tag)
        ]
        return df
    return pd.DataFrame()


def table_data_text(table: Tag) -> list[list[str]]:
    """Parses an HTML table segment and extracts data from it.

    The function processes a <table> HTML element, extracting text from <tr> (table rows)
    and <td> (table data) tags. It returns a list of rows, where each row is a list of
    column values. The function can handle a single <th> (table header) in the first row.
    """

    def row_get_data_text(tr: Tag, col_tag: str = "td") -> list[str]:  # td (data) or th (header)
        return [td.get_text(strip=True) for td in tr.find_all(col_tag) if isinstance(td, Tag)]

    rows = []
    trs = table.find_all("tr")

    # Check for header row
    if trs and isinstance(trs[0], Tag) and (header_row := row_get_data_text(trs[0], "th")):
        rows.append(header_row)
        trs = trs[1:]  # type: ignore

    # Process data rows and ensure we're only using Tag objects
    for tr in trs:
        if isinstance(tr, Tag):
            rows.append(row_get_data_text(tr, "td"))

    return rows


def fetch_page_content(link: str) -> Any:
    """Fetches the content of the page for a given link."""
    try:
        page = requests.get(link)
        page.raise_for_status()  # Raise an exception if the request was not successful
        return page.content
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching page content for link: %s", link)
        logger.error(str(e))
        return None


def add_footnotes(row_names: list) -> list:
    """Appends specific footnotes to each element in a list of row names."""
    row_names[0] += " (in Mio. Euro)"
    row_names[1] += " (EBITDA = Ergebnis vor Zinsen, Steuern und Abschreibungen, in Mio. Euro)"
    row_names[2] += " (EBITDA in Relation zu Umsatz)"
    row_names[3] += " (EBIT = Ergebnis vor Zinsen und Steuern, in Mio. Euro)"
    row_names[4] += " (EBIT in Relation zum Umsatz)"
    row_names[5] += " (in Mio. Euro)"
    row_names[6] += " (Jahresüberschuss (-fehlbetrag) in Relation zum Umsatz)"
    row_names[7] += " (Cashflow aus der gewöhnlichen Geschäftstätigkeit, in Mio. Euro)"
    row_names[8] += " (in Euro)"
    row_names[9] += " (in Euro)"

    return row_names


def process_page_content(content: str) -> Any:
    """Processes the HTML content to extract company data."""
    soup = BeautifulSoup(content, "html.parser")
    tables = soup.find_all("table")

    if len(tables) > 10 and isinstance(tables[10], Tag):
        list_table = table_data_text(tables[10])
        indexes = [
            re.sub(r"[0-9]", "", x.replace(",", ""))
            for x in [sublist[0] for sublist in list_table[2:]]
        ]
        indexes = add_footnotes(indexes)
        data = list(list_table[2:])
        td_element = soup.find("td", style="border:0px solid #CCC; padding:10px;")
        if td_element and isinstance(td_element, Tag):
            profil = td_element.get_text(strip=True)
            return data, indexes, list_table[1], profil

    return [], []


def extract_company_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts detailed company data from the links provided in the DataFrame."""
    all_data = []
    all_indices = []
    all_columns = None

    links = df["links"]
    companies = df["Unternehmen"]
    contents = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_company = {}
        for link, company_name in zip(links, companies):
            future = executor.submit(fetch_page_content, link)
            future_to_company[future] = company_name
            time.sleep(random.uniform(0.5, 2.0))

        for future in concurrent.futures.as_completed(future_to_company):
            company_name = future_to_company[future]
            try:
                content = future.result()
                if content:
                    contents.append((company_name, content))
            except Exception as e:
                logger.error(f"Error processing link for {company_name}: {e}")

    # Process pages and gather data
    for company_name, content in contents:
        data, indexes, columns, profil = process_page_content(content)
        if data:
            if all_columns is None:
                all_columns = columns + ["Unternehmen"] + ["Profil"]
            for row_data in data:
                if len(row_data) == len(columns):
                    all_data.append(row_data + [company_name] + [profil])
                else:
                    logger.warning(f"Data length mismatch for {company_name}: {row_data}")
            all_indices.extend(indexes)

    return pd.DataFrame(all_data, columns=all_columns, index=all_indices)


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    """Formats the DataFrame to have the correct data types and column names."""
    df = df.rename(columns={df.columns[0]: "Kennzahlen"})
    df["Kennzahlen"] = df.index
    df["Kennzahlen"] = df["Kennzahlen"].replace("[0-9,]", "", regex=True)
    df["Kennzahlen"] = df["Kennzahlen"].replace("ö", "oe", regex=True)
    df["Kennzahlen"] = df["Kennzahlen"].replace("ü", "ue", regex=True)
    df["Kennzahlen"] = df["Kennzahlen"].replace("ä", "ae", regex=True)

    df["Profil"] = df["Profil"].replace("ö", "oe", regex=True)
    df["Profil"] = df["Profil"].replace("ü", "ue", regex=True)
    df["Profil"] = df["Profil"].replace("ä", "ae", regex=True)
    df["Profil"] = df["Profil"].replace("ß", "ss", regex=True)
    df["Profil"] = df["Profil"].replace("\n", " ", regex=True)

    for col in df.columns[1 : (len(df.columns) - 2)]:
        df[col] = df[col].str.replace(".", "")
        df[col] = df[col].str.replace(",", ".")
        df[col] = df[col].astype(float)
    return df


def save_df(
    df: pd.DataFrame,
    folder_name: str = "/media/NAS/projects/g4k/datasets/DAX_tables",
    file_name: str = "dax_raw.csv",
) -> None:
    """Saves data frames to CSV files in the specified folder."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    df.to_csv(os.path.join(folder_name, file_name), index=False)


def save_json_lines(df: pd.DataFrame, file_name: str) -> None:
    """Saves a DataFrame to a JSON Lines file."""
    reformatted_json = df.to_dict(orient="records")
    file_manager = FileManager(file_name)
    file_manager.dump_jsonlines(reformatted_json)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--links_url",
        type=str,
        default="https://www.boersengefluester.de/wirtschaftspruefer-boersengelistete-unternehmen/",
    )
    argparser.add_argument("--output_folder", type=str, default="data")
    argparser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="Log levels: 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL",
    )
    argparser.add_argument(
        "--file_name", type=str, default="dax_raw", help="Name of the output file"
    )
    args = argparser.parse_args()

    logger.info("Fetching links from %s", args.links_url)
    df_links = get_links(args.links_url)
    logger.info("Extracting company data from links")
    company_data = extract_company_data(df_links)
    logger.info("Formatting data")
    company_data = format_df(company_data)
    logger.info("Saving data to CSV and JSON Lines files")
    save_df(company_data, args.output_folder, args.file_name + ".csv")
    save_json_lines(company_data, os.path.join(args.output_folder, args.file_name + ".jsonl"))
