import undetected_chromedriver as uc
import time
from bs4 import BeautifulSoup
import re
import csv
import os
import math
import json

# --- Configuration ---

# Mapping of filter names to their base URLs
FILTERS = {
    # Tipus
    "tegla_epitesu": "https://ingatlan.com/lista/elado+lakas+tegla-epitesu-lakas+budapest",
    "panel": "https://ingatlan.com/lista/elado+lakas+panel-lakas+budapest",
    "csuszozsalus": "https://ingatlan.com/lista/elado+lakas+csuszozsalus-lakas+budapest",
    # Allapot
    "uj_epitesu": "https://ingatlan.com/lista/elado+lakas+uj-epitesu+budapest",
    "ujszeru": "https://ingatlan.com/lista/elado+lakas+ujszeru+budapest",
    "felujitott": "https://ingatlan.com/lista/elado+lakas+felujitott+budapest",
    "jo_allapotu": "https://ingatlan.com/lista/elado+lakas+jo-allapotu+budapest",
    "kozepes_allapotu": "https://ingatlan.com/lista/elado+lakas+kozepes-allapotu+budapest",
    "felujitando": "https://ingatlan.com/lista/elado+lakas+felujitando+budapest",
    # Epites eve
    "epitve_1950_elott": "https://ingatlan.com/lista/elado+lakas+epitve-1950-elott+budapest",
    "epitve_1950_1980": "https://ingatlan.com/lista/elado+lakas+epitve-1950-1980-kozott+budapest",
    "epitve_2001_2010": "https://ingatlan.com/lista/elado+lakas+epitve-2001-2010-kozott+budapest",
    "epitve_2011_utan": "https://ingatlan.com/lista/elado+lakas+epitve-2011-utan+budapest",
    # Parkolas
    "udvari_beallo": "https://ingatlan.com/lista/elado+lakas+udvari-beallo+budapest",
    "teremgarazs": "https://ingatlan.com/lista/elado+lakas+teremgarazs-hely+budapest",
    "onallo_garazs": "https://ingatlan.com/lista/elado+lakas+onallo-garazs+budapest",
    "utcan_parkolas": "https://ingatlan.com/lista/elado+lakas+utcan-parkolas+budapest"
}

# Files to store our CSV database and progress state
CSV_FILENAME = "listings_database.csv"
PROGRESS_FILENAME = "progress.json"

# How many listings per page
PAGE_LIMIT = 1000

# --- Utility Functions ---

def load_progress():
    """Load progress from a JSON file to know which page we processed per filter."""
    if os.path.exists(PROGRESS_FILENAME):
        with open(PROGRESS_FILENAME, "r") as f:
            progress = json.load(f)
            print(f"[INFO] Loaded progress: {progress}")
            return progress
    else:
        print("[INFO] No progress file found. Starting fresh.")
        return {}

def save_progress(progress):
    """Save current progress to a JSON file."""
    with open(PROGRESS_FILENAME, "w") as f:
        json.dump(progress, f)
    print(f"[INFO] Progress saved: {progress}")

def load_existing_data():
    """Load existing listings (keyed by listing_id) from CSV so we do not duplicate base data."""
    data = {}
    if os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                lid = row.get("listing_id")
                if lid:
                    data[lid] = row
        print(f"[INFO] Loaded {len(data)} records from {CSV_FILENAME}.")
    else:
        print(f"[INFO] No existing CSV database found. Starting fresh.")
    return data

def save_data(data):
    """Write all listings data to CSV."""
    fieldnames = ["listing_id", "price", "address", "sqm", "rooms"] + list(FILTERS.keys())
    with open(CSV_FILENAME, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data.values():
            writer.writerow(row)
    print(f"[INFO] Saved {len(data)} records to {CSV_FILENAME}.")

def scrape_page(url, debug=True):
    """
    Open the page with undetected_chromedriver.
    After opening the page, you'll have time (sleep) to manually handle any captcha or bot check.
    """
    options = uc.ChromeOptions()
    # Uncomment the next line to run headless if desired:
    # options.headless = True
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")
    
    driver = uc.Chrome(options=options)
    driver.get(url)
    if debug:
        print(f"[DEBUG] Navigated to {url}")
    # Wait for content (and for you to solve any captcha if needed)
    time.sleep(16)
    html = driver.page_source
    driver.quit()
    return html

def parse_total_listings(html):
    """Extract the total number of listings from the element that contains the word 'találat'."""
    soup = BeautifulSoup(html, "html.parser")
    span = soup.find("span", string=lambda t: t and "találat" in t)
    if span:
        text = span.get_text(strip=True)
        print(f"[DEBUG] Found total listings text: '{text}'")
        # Remove all non-digit characters to get the full number
        number_str = re.sub(r"\D", "", text)
        try:
            total = int(number_str)
            return total
        except Exception as e:
            print(f"[ERROR] Error parsing total listings from '{number_str}': {e}")
            return None
    print("[WARN] No total listings element found.")
    return None


def parse_listings(html, debug=False):
    """
    Extract listing data from the page.
    For each listing, we extract:
      - listing_id (from the <a> tag's data-listing-id)
      - price, address, sqm, rooms from within the listing's content.
    """
    soup = BeautifulSoup(html, "html.parser")
    listings_data = []
    for a in soup.find_all("a", attrs={"data-listing-id": True}):
        listing_id = a.get("data-listing-id")
        content = a.find("div", class_="listing-card-content")
        if content is None:
            if debug:
                print(f"[DEBUG] Listing {listing_id}: No content container found, skipping.")
            continue

        price_tag = content.find("span", class_="fw-bold fs-5 text-onyx me-3 font-family-secondary")
        price = price_tag.get_text(strip=True) if price_tag else None

        address_tag = content.find("span", class_="d-block fw-500 fs-7 text-onyx font-family-secondary")
        address = address_tag.get_text(strip=True) if address_tag else None

        sqm = None
        rooms = None
        detail_container = content.select_one("div.d-flex.justify-content-start")
        if detail_container:
            detail_divs = detail_container.select("div.d-flex.flex-column")
            if debug:
                print(f"[DEBUG] Listing {listing_id}: Found {len(detail_divs)} detail div(s).")
            for detail in detail_divs:
                label_tag = detail.select_one("span.fs-8.text-nickel")
                value_tag = detail.select_one("span.fs-7")
                if label_tag and value_tag:
                    label = label_tag.get_text(strip=True).lower()
                    value = value_tag.get_text(" ", strip=True)
                    if debug:
                        print(f"[DEBUG] Listing {listing_id}: Detail - Label: {label}, Value: {value}")
                    if "alapterület" in label:
                        match = re.search(r'(\d+)', value)
                        sqm = match.group(1) if match else value
                    elif "szobák" in label:
                        match = re.search(r'(\d+)', value)
                        rooms = match.group(1) if match else value
        else:
            if debug:
                print(f"[DEBUG] Listing {listing_id}: No detail container found.")

        listing_dict = {
            "listing_id": listing_id,
            "price": price,
            "address": address,
            "sqm": sqm,
            "rooms": rooms
        }
        listings_data.append(listing_dict)
    if debug:
        print(f"[DEBUG] Parsed {len(listings_data)} listings from page.")
    return listings_data

# --- Main Scraper Routine ---

def main():
    print("[INFO] Starting scraper.")
    progress = load_progress()          # progress tracking per filter
    existing_data = load_existing_data()  # dictionary keyed by listing_id
    total_existing = len(existing_data)
    print(f"[INFO] Currently, the CSV database has {total_existing} records.")

    for filter_name, base_url in FILTERS.items():
        if progress.get(filter_name) == "done":
            print(f"[INFO] Filter '{filter_name}' is marked as complete. Skipping.")
            continue

        print(f"\n=== Processing filter: {filter_name} ===")
        start_page = progress.get(filter_name, 0) + 1
        print(f"[INFO] Starting at page {start_page} for filter '{filter_name}'.")
        
        url_first = f"{base_url}?limit={20}&page=1"
        html_first = scrape_page(url_first)
        total_listings = parse_total_listings(html_first)
        if total_listings is None:
            print(f"[WARN] Could not determine total listings for filter '{filter_name}'. Skipping.")
            continue

        total_pages = math.ceil(total_listings / PAGE_LIMIT)
        print(f"[INFO] Filter '{filter_name}': Found {total_listings} listings over {total_pages} page(s).")
        
        for page in range(start_page, total_pages + 1):
            url = f"{base_url}?limit={PAGE_LIMIT}&page={page}"
            print(f"[{filter_name}] Scraping page {page}/{total_pages} ...")
            html = scrape_page(url)
            listings = parse_listings(html, debug=True)
            print(f"[{filter_name}] Found {len(listings)} listings on page {page}.")
            
            for listing in listings:
                lid = listing["listing_id"]
                if lid in existing_data:
                    # Update the filter flag
                    existing_data[lid][filter_name] = True
                else:
                    new_entry = {
                        "listing_id": lid,
                        "price": listing["price"],
                        "address": listing["address"],
                        "sqm": listing["sqm"],
                        "rooms": listing["rooms"],
                    }
                    for key in FILTERS.keys():
                        new_entry[key] = False
                    new_entry[filter_name] = True
                    existing_data[lid] = new_entry

            progress[filter_name] = page
            save_progress(progress)
            save_data(existing_data)
            print(f"[{filter_name}] Completed page {page}. Total records now: {len(existing_data)}")

        # After finishing all pages for the filter, mark it as complete
        progress[filter_name] = "done"
        save_progress(progress)
        print(f"[INFO] Filter '{filter_name}' processing complete.")
    
    print("Scraping and updating complete.")

if __name__ == "__main__":
    main()
