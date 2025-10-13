import csv
import requests
import time
import os
from tqdm import tqdm
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


INPUT_FILE  = 'path/to/you/input.csv'  # your input CSV (first column = species)
BASE_NAME = os.path.splitext(os.path.basename(INPUT_FILE))[0]
OUTPUT_DIR  = f'wiki_{BASE_NAME}'       # <— all outputs go here


DESCRIPTION_NAME = f'description_{BASE_NAME}.csv'
ERRORS_NAME      = f'errors_{BASE_NAME}.csv'
CHECKPOINT_NAME  = f'scraper_{BASE_NAME}.chk'
LOG_NAME         = f'scraper_{BASE_NAME}.log'


os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, DESCRIPTION_NAME)
ERROR_FILE      = os.path.join(OUTPUT_DIR, ERRORS_NAME)
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, CHECKPOINT_NAME)
LOG_FILE        = os.path.join(OUTPUT_DIR, LOG_NAME)

# ──────────────────────────────────────────────────────────────────────────────
# Keyword list
# ──────────────────────────────────────────────────────────────────────────────

KEYWORDS = [
    "description", "morphology", "appearance",
    "identification", "feature", "characteristics",
    "physical", "structure", "explanation of names"
]
HEADING_TAGS = ['h2', 'h3', 'h4', 'h5', 'h6']

# ──────────────────────────────────────────────────────────────────────────────
# Parameters like thread number
# ──────────────────────────────────────────────────────────────────────────────
THREAD_COUNT = 30          # worker threads
FLUSH_INTERVAL = 1000      # write every N rows
MAX_RETRIES = 5            # HTTP retries
RETRY_DELAY = 5            # sleep seconds between retries in safe_get()


session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 "
        "BioClipScraper/1.0 (+contact@example.com)"
    ),
    "Accept-Language": "en;q=0.9,en-US;q=0.8"
})
retry = Retry(
    total=MAX_RETRIES,
    backoff_factor=1.2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry, pool_connections=128, pool_maxsize=max(128, THREAD_COUNT*8))
session.mount("https://", adapter)
session.mount("http://", adapter)

result_queue = queue.Queue()
write_lock = threading.Lock()
stop_event = threading.Event()

def log(message: str):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    with open(LOG_FILE, 'a', encoding='utf-8') as logf:
        logf.write(f"[{timestamp}] {message}\n")


def safe_get(url: str, **kwargs) -> requests.Response | None:
    """GET with manual retry sleep (on top of adapter retry)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=30, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            log(f"Request error ({attempt}/{MAX_RETRIES}) for URL {url}: {e}")
            time.sleep(RETRY_DELAY)
    log(f"Max retries reached for {url}, skipping.")
    return None


def _clean_query(q: str) -> str:
    """Remove trailing author/year or bracket annotations to improve enwiki hit rate."""
    q = q.strip()
    q = re.sub(r"\s*\(.*?\)\s*$", "", q)
    q = re.sub(r"\s{2,}", " ", q)
    q = q.replace("_", " ")
    return q


def search_wiki_for_species(name: str) -> str | None:
    """
    Use the Wikipedia Search API to find the best page for the species.
    Prefer exact title match (case-insensitive); otherwise pick the first result.
    Returns a stable curid link when possible.
    Falls back to direct /wiki/Title fetch if the API yields nothing.
    """
    q = _clean_query(name)
    params = {
        "action": "query",
        "list": "search",
        "srsearch": q,
        "srlimit": 5,
        "format": "json",
    }
    r = safe_get("https://en.wikipedia.org/w/api.php", params=params)
    if r is not None:
        try:
            data = r.json()
            hits = data.get("query", {}).get("search", [])
            if hits:
                exact = next((h for h in hits if h.get("title", "").lower() == q.lower()), None)
                pick = exact or hits[0]
                pageid = pick.get("pageid")
                if pageid:
                    return f"https://en.wikipedia.org/?curid={pageid}"
        except Exception as e:
            log(f"Wikipedia API parse error for '{name}': {e}")

    from requests.utils import quote
    title_url = "https://en.wikipedia.org/wiki/" + quote(q.replace(" ", "_"))
    r2 = safe_get(title_url, allow_redirects=True)
    if r2:
        return r2.url
    return None


def extract_section_content(start) -> list[str]:
    """Collect paragraphs and lists until the next heading of similar level."""
    content = []
    nxt = start.find_next()
    while nxt and nxt.name not in HEADING_TAGS:
        if nxt.name == 'p':
            content.append(nxt.get_text(separator=' ', strip=True))
        elif nxt.name in ['ul', 'ol']:
            for li in nxt.find_all('li'):
                content.append(li.get_text(separator=' ', strip=True))
        nxt = nxt.find_next()
    return content


def _is_disambiguation(soup: BeautifulSoup) -> bool:
    """
    Light-weight disambiguation detection.
    We avoid API category queries; check common HTML markers instead.
    """
    if soup.find('table', id='disambigbox'):
        return True
    hatnotes = soup.select('.hatnote')
    for h in hatnotes:
        txt = h.get_text(" ", strip=True).lower()
        if "may refer to:" in txt or "may refer to" in txt:
            return True
    return False


def scrape_description_from_wiki(url: str) -> list[str]:
    r = safe_get(url)
    if not r:
        return []
    soup = BeautifulSoup(r.text, 'html.parser')

    if _is_disambiguation(soup):
        return []

    for kw in KEYWORDS:
        for tag in soup.find_all(HEADING_TAGS):
            if kw.lower() in tag.get_text(strip=True).lower():
                sec = extract_section_content(tag)
                if sec:
                    return sec

    main = soup.find('div', id='mw-content-text')
    if main:
        first = main.find(['p', 'ul', 'ol'])
        if first:
            return extract_section_content(first)

    return []


def worker_task(idx: int, species: str):
    try:
        url = search_wiki_for_species(species)
        if not url:
            raise ValueError('No search results')
        data = scrape_description_from_wiki(url)
        if not data:
            raise ValueError('No content extracted')
        result_queue.put((idx, species, '\n'.join(data), None))
    except Exception as e:
        result_queue.put((idx, species, None, str(e)))


def checkpoint_worker(total_count: int):
    """
    Background writer thread:
    - Drains result_queue
    - Appends successes to OUTPUT_FILE
    - Appends failures to ERROR_FILE
    - Every FLUSH_INTERVAL processed rows, writes checkpoint
    """
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['species', 'content'])
    if not os.path.exists(ERROR_FILE):
        with open(ERROR_FILE, 'w', newline='', encoding='utf-8') as ef:
            csv.writer(ef).writerow(['species', 'reason'])

    error_writer_fp = open(ERROR_FILE, 'a', newline='', encoding='utf-8')
    err_csv = csv.writer(error_writer_fp)

    buffer_success: list[tuple[str, str]] = [] 
    processed = load_checkpoint()

    while not stop_event.is_set() or not result_queue.empty():
        try:
            idx, species, content, err = result_queue.get(timeout=1)
            processed += 1

            if err:
                err_csv.writerow([species, err])
            else:
                buffer_success.append((species, content))

            if processed % FLUSH_INTERVAL == 0:
                with write_lock:
                    if buffer_success:
                        with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerows(buffer_success)
                        buffer_success.clear()
                    with open(CHECKPOINT_FILE, 'w') as chk:
                        chk.write(str(processed))
                log(f"Checkpoint at {processed}")
        except queue.Empty:
            continue

    with write_lock:
        if buffer_success:
            with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(buffer_success)
        with open(CHECKPOINT_FILE, 'w') as chk:
            chk.write(str(processed))
    log(f"Final checkpoint at {processed}")
    error_writer_fp.close()


def load_checkpoint() -> int:
    if os.path.exists(CHECKPOINT_FILE):
        try:
            return int(open(CHECKPOINT_FILE, 'r', encoding='utf-8').read().strip())
        except Exception:
            return 0
    return 0


def main():
    open(LOG_FILE, 'w', encoding='utf-8').close()
    log(f"Starting multithreaded scraper for {INPUT_FILE}")

    with open(INPUT_FILE, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        all_species = [r[0] for r in reader]

    start_idx = load_checkpoint()
    total = len(all_species)

    writer_thread = threading.Thread(target=checkpoint_worker, args=(total,), daemon=True)
    writer_thread.start()

    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        futures = []
        for idx, species in enumerate(all_species[start_idx:], start=start_idx + 1):
            futures.append(executor.submit(worker_task, idx, species))
        for _ in tqdm(as_completed(futures), total=len(futures), desc='Tasks'):
            pass

    stop_event.set()
    writer_thread.join()

    log(f"Completed all {total} species")
    print("Scraping complete.")


if __name__ == '__main__':
    main()
