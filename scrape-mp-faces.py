#!/usr/bin/env python3
"""
Scrape MP face photos from the Singapore Parliament website.
Downloads all MP photos and creates an index mapping file.
"""

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.parliament.gov.sg"
LIST_URL = f"{BASE_URL}/mps/list-of-current-mps"
OUTPUT_DIR = Path(__file__).resolve().parent / "mp_faces"

# Titles/honorifics to strip from names
TITLE_PATTERN = re.compile(
    r"^(Mr|Mrs|Ms|Miss|Mdm|Dr|Prof|Assoc\s+Prof\s+Dr|Assoc\s+Prof|Associate\s+Professor)\s+",
    re.IGNORECASE,
)


def clean_name(raw_name: str) -> str:
    """Strip titles/honorifics and extra whitespace from an MP name."""
    name = raw_name.strip()
    name = TITLE_PATTERN.sub("", name)
    return name.strip()


def name_to_filename(name: str, ext: str) -> str:
    """Convert a name like 'Lawrence Wong' to 'lawrence-wong.jpg'."""
    slug = name.lower()
    # Replace non-alphanumeric characters (except spaces) with nothing
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    # Replace whitespace with hyphens
    slug = re.sub(r"\s+", "-", slug.strip())
    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug)
    return f"{slug}{ext}"


def get_extension_from_url(url: str) -> str:
    """Extract file extension from a URL, stripping query params."""
    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    return ext.lower() if ext else ".jpg"


def fetch_mp_list(session: requests.Session) -> list[dict]:
    """Fetch the MP listing page and extract names + photo URLs."""
    print(f"Fetching MP list from {LIST_URL} ...")
    resp = session.get(LIST_URL, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Each MP is in a <li> containing <div class="row list-of-mps-wrap">
    mp_wraps = soup.select("div.list-of-mps-wrap")
    print(f"Found {len(mp_wraps)} MP entries on the page.")

    mps = []
    for wrap in mp_wraps:
        # Extract image URL
        img_tag = wrap.select_one("img.img-responsive")
        if not img_tag or not img_tag.get("src"):
            print("  WARNING: No image found for an entry, skipping.")
            continue

        img_src = img_tag["src"]
        # Build full URL
        photo_url = urljoin(BASE_URL, img_src)

        # Extract MP name from the <a> tag inside mp-sort-name div
        name_div = wrap.select_one("div.mp-sort-name")
        if name_div:
            a_tag = name_div.select_one("a")
            if a_tag:
                raw_name = a_tag.get_text(strip=True)
            else:
                raw_name = name_div.get_text(strip=True)
        else:
            # Fallback: try the mp-last-name div
            last_name_div = wrap.select_one("div.mp-last-name")
            if last_name_div:
                raw_name = last_name_div.get_text(strip=True)
            else:
                print("  WARNING: No name found for an entry, skipping.")
                continue

        name = clean_name(raw_name)
        ext = get_extension_from_url(photo_url)
        filename = name_to_filename(name, ext)

        mps.append(
            {
                "name": name,
                "raw_name": raw_name,
                "photo_url": photo_url,
                "filename": filename,
            }
        )

    return mps


def download_photos(session: requests.Session, mps: list[dict]) -> dict:
    """Download all MP photos and return the filename->name mapping."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mapping = {}
    total = len(mps)

    for i, mp in enumerate(mps, 1):
        filename = mp["filename"]
        filepath = OUTPUT_DIR / filename
        name = mp["name"]
        url = mp["photo_url"]

        # Handle duplicate filenames by appending a number
        if filename in mapping:
            base, ext = os.path.splitext(filename)
            counter = 2
            while f"{base}-{counter}{ext}" in mapping:
                counter += 1
            filename = f"{base}-{counter}{ext}"
            filepath = OUTPUT_DIR / filename
            mp["filename"] = filename

        print(f"  [{i}/{total}] Downloading {name} -> {filename}")

        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            filepath.write_bytes(resp.content)
            mapping[filename] = name
        except requests.RequestException as e:
            print(f"    ERROR downloading {url}: {e}")
            continue

        # Small delay to be polite to the server
        if i % 10 == 0:
            time.sleep(0.5)

    return mapping


def save_index(mapping: dict) -> None:
    """Save the filename->name mapping as index.json."""
    index_path = OUTPUT_DIR / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"\nSaved index with {len(mapping)} entries to {index_path}")


def main():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )

    # Step 1: Fetch and parse MP list
    mps = fetch_mp_list(session)
    if not mps:
        print("No MPs found. The page structure may have changed.")
        return

    print(f"\nExtracted {len(mps)} MPs with photos.")
    print(f"Downloading photos to {OUTPUT_DIR} ...\n")

    # Step 2: Download all photos
    mapping = download_photos(session, mps)

    # Step 3: Save index.json
    save_index(mapping)

    print(f"\nDone! Downloaded {len(mapping)} photos to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
