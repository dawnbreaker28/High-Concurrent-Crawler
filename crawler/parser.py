from bs4 import BeautifulSoup

def parse_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    return [a.get('href') for a in soup.find_all('a', href=True)]
