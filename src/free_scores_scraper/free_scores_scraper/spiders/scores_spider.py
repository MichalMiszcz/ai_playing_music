import scrapy
from urllib.parse import urljoin
import os
import re


class ScoresSpider(scrapy.Spider):
    name = "scores"
    allowed_domains = ["free-scores.com"]
    start_urls = [
        "https://www.free-scores.com/free-sheet-music.php?search=&CATEGORIE=70&divers=mp3"
    ]

    def parse(self, response):
        # Go to each music score's detail page
        for link in response.css('a.tl::attr(href)').getall():
            score_page = urljoin(response.url, link)
            yield scrapy.Request(score_page, callback=self.parse_score_page)

    def parse_score_page(self, response):
        # Get the title of the music score
        title = response.css('h1[itemprop="name"]::text').get()
        if not title:
            title = "unknown_title"

        # Clean title to make it a valid folder name
        folder_name = re.sub(r'[\\/*?:"<>|]', "", title).strip()
        folder_path = os.path.join("../../../downloads", folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Find MP3 and PDF links
        for href in response.css('a::attr(href)').getall():
            if href.endswith(".pdf") or href.endswith(".mp3"):
                file_url = urljoin(response.url, href)
                yield scrapy.Request(
                    file_url,
                    callback=self.save_file,
                    cb_kwargs={"folder": folder_path}
                )

    def save_file(self, response, folder):
        filename = response.url.split("/")[-1]
        path = os.path.join(folder, filename)
        with open(path, "wb") as f:
            f.write(response.body)
        self.log(f"Saved file {path}")
