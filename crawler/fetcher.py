import requests
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from utils import logger

def requests_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def fetch_content(url, timeout=1):
    session = requests_session()
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return url, response.text
    except requests.exceptions.HTTPError as errh:
        logger.log(f"HTTP Error fetching {url}: {errh}")
    except requests.exceptions.ConnectionError as errc:
        logger.log(f"Error Connecting to {url}: {errc}")
    except requests.exceptions.Timeout as errt:
        logger.log(f"Timeout Error fetching {url}: {errt}")
    except requests.exceptions.RequestException as err:
        logger.log(f"Error fetching {url}: {err}")
    return url, ""

def fetch_all(urls, max_workers=10):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_content, url): url for url in urls}
        for future in futures:
            url = futures[future]
            try:
                url, content = future.result()
                results[url] = content
            except Exception as e:
                logger.log(f"Error fetching {url}: {e}")
                results[url] = ""
    return results
