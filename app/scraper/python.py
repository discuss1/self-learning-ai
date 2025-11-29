import requests
from bs4 import BeautifulSoup
import os

def scrape_python_docs():
    os.makedirs("./data/python", exist_ok=True)
    base_url = "https://docs.python.org/3/"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.select("a.reference.internal"):
        doc_url = link["href"] if link["href"].startswith("http") else base_url + link["href"]
        try:
            doc_response = requests.get(doc_url)
            doc_soup = BeautifulSoup(doc_response.text, "html.parser")
            content = doc_soup.get_text(separator="\n", strip=True)
            with open(f"./data/python/{link.text.replace('/', '_')}.md", "w") as f:
                f.write(content)
        except Exception as e:
            print(f"Failed to scrape {doc_url}: {e}")

if __name__ == "__main__":
    scrape_python_docs()

