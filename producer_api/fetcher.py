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

def fetch_content(url, timeout=5):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Connection": "keep-alive",
        "Cookie": "ckns_policy=111; _cb=CNRmQmCK23LICahvti; dnsDisplayed=undefined; ccpaApplies=true; signedLspa=undefined; _sp_su=false; ccpaUUID=d0cbd181-6210-4ce6-a1f3-b27f4e9c60b3; permutive-id=249df450-8682-4948-9b4f-6673d204d7fd; cX_P=ls282vla8xoqrl73; cX_G=cx%3A2fsvn30pk8bbu39m5rxi3rstyw%3Akhcwp7g92dre; _pcid=%7B%22browserId%22%3A%22ls282vla8xoqrl73%22%2C%22_t%22%3A%22m7qn0ge5%7Cls282z25%22%7D; _pcus=eyJ1c2VyU2VnbWVudHMiOnsiQ09NUE9TRVIxWCI6eyJzZWdtZW50cyI6WyJMVHJldHVybjplN2Y1N2MxNTRhZWZmYWYxYjM0YWY2ZDZiYjcxMjg0ZjQ0NDUxMzllOm5vX3Njb3JlIl19fSwiX3QiOiJtN3FuMGdlNXxsczI4MnoyNSJ9; _pctx=%7Bu%7DN4IgrgzgpgThIC4B2YA2qA05owMoBcBDfSREQpAeyRCwgEt8oBJAEzIGYA2ABi644AmQQFYALDwDsYgIwiR3AJy0QAfXxkAtpICOSHgHMoIgD6oIggByCAXqJABfIA; _pctx=%7Bu%7DN4IgrgzgpgThIC4B2YA2qA05owMoBcBDfSREQpAeyRCwgEt8oBJAEzIGYA2ABi644AmQQFYALDwDsYgIwiR3AJy0QAfXxkAtpICOSHgHMoIgD6oIggByCAXqJABfIA; pa_privacy=%22optin%22; atuserid={%22val%22:%22b7b3939e-e47d-4eff-8d43-e9f760b66c8b%22}; ckns_privacy=july2019; ckns_explicit=2; optimizelyEndUserId=oeu1720685113655r0.87581775464375; blaize_session=be16b443-d17e-4640-9746-8dbb664426fe; blaize_tracking_id=3a87e7b7-bc11-4727-b30f-ceb3f0217537; ckpf_ppid=4368ac21135149f28e933938d31d3125; pa_vid=%22b7b3939e-e47d-4eff-8d43-e9f760b66c8b%22; DM_SitId1778=1; DM_SitId1778SecId13934=1; __pat=3600000; ckns_mvt=e03ba7b5-9fa0-4fea-8391-470114a30dd3; DM_SitId1778SecId14803=1; _pbjs_userid_consent_data=3524755945110770; __tbc=%7Bkpex%7DAmvACVQxl2oczh7qTW6YZOAuFe65_K6Ne1LjsNUzecR89RiZ7-Ljx0E6SNx6G2U_4KsDPkAcyg-anYlUere9oQ; xbc=%7Bkpex%7DdJQzOUSE4elZDJhTQiagqA; _tracking_consent=%7B%22con%22%3A%7B%22CMP%22%3A%7B%22a%22%3A%22%22%2C%22m%22%3A%22%22%2C%22p%22%3A%22%22%2C%22s%22%3A%22%22%7D%7D%2C%22v%22%3A%222.1%22%2C%22region%22%3A%22JP13%22%2C%22reg%22%3A%22%22%7D; _shopify_y=079d6bbc-85a9-4965-9846-3c408287b4ab; _ga=GA1.2.700593736.1720855942; _ga_2KNJ69MBJ6=GS1.1.1720855941.1.1.1720855947.58.0.0; _pcus=eyJ1c2VyU2VnbWVudHMiOnsiQ09NUE9TRVIxWCI6eyJzZWdtZW50cyI6WyJMVHJldHVybjplN2Y1N2MxNTRhZWZmYWYxYjM0YWY2ZDZiYjcxMjg0ZjQ0NDUxMzllOm5vX3Njb3JlIl19fSwiX3QiOiJtN3FuMGdlNXxsczI4MnoyNSJ9; atuserid=%7B%22name%22%3A%22atuserid%22%2C%22val%22%3A%22b7b3939e-e47d-4eff-8d43-e9f760b66c8b%22%2C%22options%22%3A%7B%22end%22%3A%222025-08-15T08%3A35%3A48.142Z%22%2C%22path%22%3A%22%2F%22%7D%7D; _chartbeat2=.1706731853916.1722096708215.1110100000000001.DTABCFBycwcQDNVQl0Cvk9FqMq_vi.1; _cb_svref=https%3A%2F%2Fwww.bbc.com%2Fnews; _pcid=%7B%22browserId%22%3A%22b7b3939e-e47d-4eff-8d43-e9f760b66c8b%22%2C%22_t%22%3A%22m7qn0ge5%7Cls282z25%22%7D; __gads=ID=569652f5b01b3af6:T=1720685117:RT=1722096708:S=ALNI_MZ4UEKpOk3ufo93IPSZisQZi0IdOA; __gpi=UID=00000e8c4ca6f4a6:T=1720685117:RT=1722096708:S=ALNI_MZ_Di3jsf0ZBdjFtQZjQVquY2pozA; __eoi=ID=3f07b242cd14a7ff:T=1720685117:RT=1722096708:S=AA-AfjYrwQAv9ELuCj31d_cxBktl; ecos.dt=1722096716273"
    }
    session = requests_session()
    try:
        response = session.get(url, timeout=timeout, headers=headers)
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

def fetch_all(urls, max_workers=100):
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
