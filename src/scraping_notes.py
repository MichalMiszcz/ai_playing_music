import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from time import sleep

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

BASE_URL = "https://www.free-scores.com"
START_URL = "https://www.free-scores.com/download-sheet-music.php?pdf=1508"

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

def download_file(file_url, folder):
    local_filename = os.path.join(folder, file_url.split("/")[-1])
    r = requests.get(file_url, headers=headers)
    with open(local_filename, "wb") as f:
        f.write(r.content)
    print(f"Saved: {local_filename}")

def scrape():
    response = requests.get(START_URL, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    score_links = soup.select("a.tl")
    print(f"Found {len(score_links)} scores")

    for a in score_links:
        href = a.get("href")
        score_url = urljoin(BASE_URL, href)
        score_page = requests.get(score_url, headers=headers)
        score_soup = BeautifulSoup(score_page.text, "html.parser")

        title = score_soup.select_one("h1[itemprop='name']")
        if not title:
            continue
        title_text = sanitize_filename(title.get_text(strip=True))
        folder = os.path.join("downloads", title_text)
        os.makedirs(folder, exist_ok=True)

        for a_tag in score_soup.find_all("a", href=True):
            link = a_tag["href"]
            if link.endswith(".pdf") or link.endswith(".mp3"):
                full_url = urljoin(BASE_URL, link)
                try:
                    download_file(full_url, folder)
                    sleep(1)  # polite delay
                except Exception as e:
                    print(f"Failed to download {full_url}: {e}")

if __name__ == "__main__":
    os.makedirs("downloads", exist_ok=True)
    scrape()
