import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

def scrape_django_docs():
    base_url = "https://docs.djangoproject.com/en/4.2/"
    os.makedirs("./data/django", exist_ok=True)

    # Scrape the table of contents
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract all documentation links
    for link in soup.select("a.reference.internal, a.link"):
        doc_url = urljoin(base_url, link["href"])
        if doc_url.endswith("/") or "#" in doc_url:
            continue  # Skip directories and anchors

        try:
            print(f"📄 Scraping: {doc_url}")
            doc_response = requests.get(doc_url)
            doc_soup = BeautifulSoup(doc_response.text, "html.parser")

            # Save clean text (remove nav/headers/footers)
            content = doc_soup.find("div", {"class": "document"}).get_text(separator="\n", strip=True)
            filename = doc_url.split("/")[-1].replace(".html", ".md")
            with open(f"./data/django/{filename}", "w") as f:
                f.write(content)
        except Exception as e:
            print(f"⚠️ Failed to scrape {doc_url}: {e}")

if __name__ == "__main__":
    scrape_django_docs()

