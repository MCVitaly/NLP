import requests
from bs4 import BeautifulSoup
import csv
import time
import re
from urllib.parse import urljoin, urlparse


start_urls = [
    "https://www.ikea.com",
    "https://www.wayfair.com",
    "https://www.overstock.com",
    "https://www.westelm.com",
    "https://www.hermanmiller.com"
]

# Header of the request to simulate a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36"
}


stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now"
])

# filtering words, similar to URL
def is_valid_word(word):
    invalid_patterns = ["https", "www", "com", "net", "org", "html", "php", "asp"]
    return not any(pattern in word.lower() for pattern in invalid_patterns)

#getting text from the url
def fetch_page(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_all_links(url, html):
    soup = BeautifulSoup(html, 'html.parser')
    base_url = "{0.scheme}://{0.netloc}".format(urlparse(url))
    links = set()

    for link in soup.find_all("a", href=True):
        full_url = urljoin(base_url, link["href"])
        if urlparse(full_url).netloc == urlparse(url).netloc:  # Internal links only
            links.add(full_url)

    return links

def fetch_headings_text(html):

    soup = BeautifulSoup(html, 'html.parser')
    headings = []

    # Extracting the text of the h1 - h6 headers
    for i in range(1, 7):
        for heading in soup.find_all(f"h{i}"):
            headings.append(heading.get_text(strip=True))

    # Header text processing
    all_headings_text = ' '.join(headings)
    return filter_text(all_headings_text)

def filter_text(text):

    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation marks
    words = text.split()
    filtered_words = set(
        word for word in words
        if word.lower() not in stop_words and word.isalpha() and is_valid_word(word)
    )
    return filtered_words

def collect_data_to_csv(urls, filename="furniture_data.csv"):

    visited_urls = set()

    with open(filename, mode="w", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["text"])  #write a header

        for start_url in urls:
            to_visit = {start_url}

            while to_visit:
                url = to_visit.pop()
                if url in visited_urls:
                    continue

                html = fetch_page(url)
                visited_urls.add(url)

                if html:
                    all_headings_text = fetch_headings_text(html)
                    if all_headings_text:
                        for word in all_headings_text:
                            writer.writerow([word])
                        print(f"Filtered headings from {url} saved successfully.")
                    else:
                        print(f"No headings found for {url}.")

                    # Adding internal links to pages that have not yet been visited
                    to_visit.update(extract_all_links(url, html) - visited_urls)

                time.sleep(5)

collect_data_to_csv(start_urls, filename="furniture_data.csv")
