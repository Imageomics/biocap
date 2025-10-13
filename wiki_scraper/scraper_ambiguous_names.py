import csv, os, time, random, threading, queue, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from tqdm import tqdm

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8"}
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
RETRY_DELAY = 3
JITTER = (0.15, 0.45)

# Heuristic section keywords for description-like content
KEYWORDS = [
    "description", "morphology", "appearance", "identification",
    "feature", "characteristics", "physical", "structure", "explanation of names"
]
HEADING_TAGS = ['h2','h3','h4','h5','h6']

# Rank aliases (plants may use "division" in place of "phylum")
RANK_ALIASES = {
    "kingdom": ["kingdom"],
    "phylum":  ["phylum", "division"],
    "class":   ["class"],
    "order":   ["order"],
    "family":  ["family"],
    "genus":   ["genus"],
    "species": ["species"],
}

_thread_local = threading.local()

def get_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        _thread_local.session = s
    return s

def safe_get(url: str):
    s = get_session()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = s.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            time.sleep(random.uniform(*JITTER))
            return r
        except Exception:
            time.sleep(RETRY_DELAY * (1.5 ** (attempt - 1)))
    return None

def search_wiki_candidates(query: str) -> List[str]:
    q = quote_plus(query)
    url = f"https://en.wikipedia.org/w/index.php?search={q}"
    r = safe_get(url)
    if not r:
        return []
    if r.url.startswith("https://en.wikipedia.org/wiki/"):
        return [r.url]
    soup = BeautifulSoup(r.text, 'html.parser')
    return [
        f"https://en.wikipedia.org{a['href']}"
        for a in soup.select('.mw-search-result-heading a')
        if a.has_attr('href')
    ]

def _norm(txt: str) -> str:
    return ' '.join(txt.split()).strip().lower()

def _extract_taxobox_rows(soup: BeautifulSoup) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    tables = soup.select('table.infobox') or soup.find_all('table')
    for tbl in tables:
        for tr in tbl.find_all('tr'):
            cells = tr.find_all(['th', 'td'])
            if not cells:
                continue
            label_raw = cells[0].get_text(' ', strip=True).lower()
            label_norm: Optional[str] = None
            for std, aliases in RANK_ALIASES.items():
                if any((alias + ':') in label_raw for alias in aliases):
                    label_norm = std
                    break
            if label_norm is None:
                continue
            if len(cells) > 1:
                val = cells[1].get_text(' ', strip=True)
            else:
                row_text = tr.get_text(' ', strip=True)
                val = row_text.replace(cells[0].get_text(' ', strip=True), '').strip()
            val_norm = _norm(val)
            if val_norm:
                rows.append((label_norm, val_norm))
    bn = soup.select_one('.binomial')
    if bn:
        parts = _norm(bn.get_text(' ', strip=True)).replace('\u00a0', ' ').split()
        if len(parts) >= 2:
            if not any(lab == 'genus' for lab, _ in rows):
                rows.append(('genus', parts[0]))
            if not any(lab == 'species' for lab, _ in rows):
                rows.append(('species', parts[1]))
    return rows

def _soft_norm(x: str) -> str:
    """Looser normalization: e.g., Tracheophyta ~ Tracheophytes."""
    t = ''.join(ch for ch in x.lower().strip() if ch.isalpha())
    for suf in ('phyta','phyte','phytes','opsida','idae','inae','oidea','aceae','ales','ina','eae','ae','es','us','a','e','s'):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            t = t[: -len(suf)]
            break
    return t

def _approx_equal(a: str, b: str) -> bool:
    return _soft_norm(a) == _soft_norm(b)

def _taxonomy_matches_overlap(expected: Dict[str, str], rows: List[Tuple[str, str]]) -> bool:
    """Compare ONLY overlapping standard ranks between wiki (rows) and CSV.
       Consider ranks: kingdom, phylum, class, order, family, genus (species ignored).
       If a rank exists in BOTH, it must match (approx or exact). If multiple rows for
       a rank exist on wiki, the first matching wins.
    """
    ranks = ["kingdom", "phylum", "class", "order", "family", "genus"]
    wiki_by_rank: Dict[str, List[str]] = {rk: [] for rk in ranks}
    for lab, val in rows:
        if lab in wiki_by_rank:
            wiki_by_rank[lab].append(val)

    for rk in ranks:
        exp_val = expected.get(rk)
        if not exp_val:
            continue
        exp_norm = _norm(str(exp_val))
        candidates = wiki_by_rank.get(rk, [])
        if not candidates:
            continue
        matched = False
        for cand in candidates:
            if _approx_equal(cand, exp_norm) or cand == exp_norm:
                matched = True
                break
        if not matched:
            return False
    return True

def resolve_binomial(binomial: str, expected: Dict[str, str]) -> Tuple[Optional[str], Dict[str, str]]:
    for url in search_wiki_candidates(binomial):
        r = safe_get(url)
        if not r:
            continue
        soup = BeautifulSoup(r.text, 'html.parser')
        rows = _extract_taxobox_rows(soup)
        if _taxonomy_matches_overlap(expected, rows):
            found: Dict[str, str] = {}
            for lab, val in rows:
                if lab not in found:
                    found[lab] = val
            return url, found
    return None, {}

def extract_section_content(start) -> List[str]:
    out: List[str] = []
    nxt = start.find_next()
    while nxt and nxt.name not in HEADING_TAGS:
        if nxt.name == 'p':
            t = nxt.get_text(' ', strip=True)
            if t:
                out.append(t)
        elif nxt.name in ['ul', 'ol']:
            for li in nxt.find_all('li'):
                t = li.get_text(' ', strip=True)
                if t:
                    out.append(t)
        nxt = nxt.find_next()
    return out

def scrape_description_from_wiki(url: str) -> List[str]:
    r = safe_get(url)
    if not r:
        return []
    soup = BeautifulSoup(r.text, 'html.parser')
    for kw in KEYWORDS:
        for tag in soup.find_all(HEADING_TAGS):
            if kw in tag.get_text(strip=True).lower():
                sec = extract_section_content(tag)
                if sec:
                    return sec
    main = soup.find('div', id='mw-content-text')
    if main:
        first = main.find(['p', 'ul', 'ol'])
        if first:
            return extract_section_content(first)
    return []

def load_checkpoint(path: str) -> int:
    if os.path.exists(path):
        try:
            return int(open(path).read().strip())
        except Exception:
            return 0
    return 0

def ensure_headers(success_path: str, error_path: str):
    os.makedirs(os.path.dirname(success_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(error_path) or '.', exist_ok=True)
    if not os.path.exists(success_path):
        with open(success_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'binomial','kingdom','phylum','class','order','family','genus','species','url','content'
            ])
    if not os.path.exists(error_path):
        with open(error_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['binomial','reason'])

def run_ambiguous(input_csv: str, out_csv: str, err_csv: str, chk: str, threads: int, flush_every: int):
    import pandas as pd
    ensure_headers(out_csv, err_csv)

    df = pd.read_csv(input_csv)
    required = ["binomial","kingdom","phylum","class","order","family","genus","species"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Ambiguous CSV missing columns: {missing}")

    rows = df[required].values.tolist()
    start_idx = load_checkpoint(chk)

    # Queue holds: (idx, success_row_or_None, error_row_or_None)
    q: "queue.Queue[tuple]" = queue.Queue()
    stop_event = threading.Event()
    wlock = threading.Lock()

    def worker(idx: int, rec: List[str]):
        binomial, kingdom, phylum, klass, order, family, genus, species = rec
        expected = {
            "kingdom": kingdom, "phylum": phylum, "class": klass,
            "order": order, "family": family, "genus": genus, "species": species
        }
        try:
            url, _found = resolve_binomial(binomial, expected)
            if not url:
                raise ValueError('No candidate matched expected taxonomy')
            data = scrape_description_from_wiki(url)
            if not data:
                raise ValueError('No content extracted')
            content = "\n".join(data)
            q.put((
                idx,
                (binomial, kingdom, phylum, klass, order, family, genus, species, url, content),
                None
            ))
        except Exception as e:
            q.put((idx, None, (binomial, str(e))))

    def writer(total: int):
        processed = start_idx
        buffer: List[tuple] = []
        with open(err_csv, 'a', newline='', encoding='utf-8') as ef:
            errw = csv.writer(ef)
            while not stop_event.is_set() or not q.empty():
                try:
                    idx, success_row, error_row = q.get(timeout=1)
                except queue.Empty:
                    continue
                processed += 1
                if error_row:
                    errw.writerow(error_row)
                else:
                    buffer.append(success_row)
                if processed % flush_every == 0:
                    with wlock:
                        if buffer:
                            with open(out_csv, 'a', newline='', encoding='utf-8') as sf:
                                csv.writer(sf).writerows(buffer)
                            buffer.clear()
                        with open(chk, 'w') as cf:
                            cf.write(str(processed))
        # final flush
        with wlock:
            if buffer:
                with open(out_csv, 'a', newline='', encoding='utf-8') as sf:
                    csv.writer(sf).writerows(buffer)
            with open(chk, 'w') as cf:
                cf.write(str(processed))

    writer_thread = threading.Thread(target=writer, args=(len(rows),), daemon=True)
    writer_thread.start()

    futures = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for i, rec in enumerate(rows[start_idx:], start=start_idx + 1):
            futures.append(ex.submit(worker, i, rec))
        for _ in tqdm(as_completed(futures), total=len(futures), desc='Ambiguous tasks'):
            pass

    stop_event.set()
    writer_thread.join()
    print("Ambiguous scraping complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Wikipedia descriptions for ambiguous species names")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file with species names and taxonomy")
    parser.add_argument("--output", type=str, default="descriptions_ambiguous.csv", help="Output CSV file for descriptions")
    parser.add_argument("--errors", type=str, default="errors_ambiguous.csv", help="Output CSV file for errors")
    parser.add_argument("--checkpoint", type=str, default="ambiguous.chk", help="Checkpoint file")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads")
    parser.add_argument("--flush-every", type=int, default=1000, help="Flush frequency")

    args = parser.parse_args()

    run_ambiguous(args.input, args.output, args.errors, args.checkpoint, args.threads, args.flush_every)
