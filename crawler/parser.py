from bs4 import BeautifulSoup
from urllib.parse import urljoin

def parse_links(base_url, html): # all href links 
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for a in soup.find_all('a', href=True):
        link = a['href']
        # Convert relative links to absolute links
        full_link = urljoin(base_url, link)
        links.append(full_link)
    return links


