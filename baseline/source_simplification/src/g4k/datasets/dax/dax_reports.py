"""Download PDFs from HTML.

Usage
python -m g4k.datasets.dax.dax_reports ~/genial4kmu/data/dax_website.html ~/genial4kmu/data/dax_pdf.
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag


def load_html_content(file_path: str) -> str:
    """Load HTML content from a specified file path."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def parse_pdf_links(html_content: str) -> Tuple[List[str], List[str]]:
    """Parse PDF links and titles from the HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    pdf_links = soup.find_all("a", href=lambda href: href and ".pdf" in href)
    hrefs: List[str] = []
    report_titles: List[str] = []

    for a in pdf_links:
        if isinstance(a, Tag):
            href = a.get("href")
            title = a.get("title")
            if href is not None and isinstance(href, str):
                hrefs.append(href)
            if title is not None and isinstance(title, str):
                report_titles.append(title)

    return hrefs, report_titles


def extract_report_data(report_titles: List[str]) -> List[Tuple[str, str]]:
    """Extract company names and years from report titles."""
    data = []
    for title in report_titles:
        match = re.search(r"Geschäftsbericht (.+?) (\d{4})", title)
        if match:
            company = match.group(1)
            year = match.group(2)
            data.append((company.lower(), year))
    return data


def create_dataframe(hrefs: List[str], data: List[Tuple[str, str]]) -> pd.DataFrame:
    """Create a DataFrame from extracted PDF links and report data."""
    df = pd.DataFrame(data, columns=["Company", "Year"])
    if len(hrefs) != len(df):
        raise ValueError("Number of PDF links and report data do not match.")
    df.insert(0, "Link", hrefs)
    df["Company"] = df["Company"].apply(lambda x: x.split()[0])  # Keep only the first word
    return df


def download_pdf(pdf_url: str, file_name: str, folder: str) -> None:
    """Download a PDF file from a URL and save it to a specified folder."""
    response = requests.get(pdf_url)
    if response.status_code == 200:
        pdf_path = Path(folder, file_name)
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(response.content)
        print(f"Downloaded: {pdf_path}")
    else:
        print(f"Failed to download: {pdf_url}")


def main(input_file: str, output_folder: str) -> None:
    """Main function to extract PDF links and download the files."""
    # Load the HTML content
    html_content = load_html_content(input_file)

    # Parse PDF links and titles
    hrefs, report_titles = parse_pdf_links(html_content)

    # Extract report data
    data = extract_report_data(report_titles)

    # Create a DataFrame
    df = create_dataframe(hrefs, data)

    # Convert output_folder to Path and create it if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download each PDF
    for _, row in df.iterrows():
        download_pdf(row["Link"], f"{row['Company']}_{row['Year']}.pdf", output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PDFs from HTML.")
    parser.add_argument("input_file", type=str, help="Path to the input HTML file.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder for PDFs.")
    args = parser.parse_args()

    main(args.input_file, args.output_folder)
