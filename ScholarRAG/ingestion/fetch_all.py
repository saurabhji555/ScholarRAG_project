import json
import time
import threading
import requests
import urllib3
import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_base = Path(__file__).parent
for _env in (_base / ".env", _base.parent / "backend" / ".env"):
    if _env.exists():
        load_dotenv(dotenv_path=_env)
        break
load_dotenv(dotenv_path=Path(__file__).parent.parent / "backend" / ".env")

PWC_FILE       = Path("ingestion/pwc_papers.json")
OA_FILE        = Path("ingestion/oa_papers.json")
OA_PROGRESS    = Path("ingestion/oa_progress.json")
CIT_CACHE      = Path("ingestion/citations_cache.json")
ENRICHED_FILE  = Path("ingestion/papers_enriched.json")
PROGRESS_FILE  = Path("ingestion/fetch_progress.json")
PWC_LINKS_FILE = Path("ingestion/pwc_links.json")

PWC_ABSTRACTS_DS = "pwc-archive/papers-with-abstracts"
OA_BASE    = "https://api.openalex.org/works"
OA_HEADERS = {"User-Agent": "ScholarRAG/1.0 (mailto:24f3003029@ds.study.iitm.ac.in)"}

OA_FIELDS = [
    "Medicine", "Biology", "Astronomy", "Chemistry",
    "Environmental Science", "Psychology", "Economics",
    "Physics", "Political Science", "History",
]
OA_YEARS    = list(range(2014, 2026))
OA_PER_YEAR = 100

_print_lock = threading.Lock()


def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


_AREA_KEYWORDS: list[tuple[str, list[str]]] = [
    ("vision",   ["image classification", "image generation", "image segmentation",
                  "object detection", "depth estimation", "optical flow",
                  "pose estimation", "medical imaging", "3d generation",
                  "image synthesis", "text-to-image", "nerf", "neural radiance",
                  "inpainting", "style transfer", "stereo matching"]),
    ("video",    ["video classification", "video generation", "video segmentation",
                  "video understanding", "object tracking", "action recognition",
                  "video captioning", "activity recognition", "video question"]),
    ("language", ["machine translation", "named entity recognition", "question answering",
                  "relation extraction", "summarization", "text classification",
                  "text-to-sql", "sentiment", "coreference", "parsing",
                  "reading comprehension", "dialogue", "natural language inference",
                  "information extraction"]),
    ("audio",    ["automatic speech recognition", "text-to-speech", "voice cloning",
                  "audio classification", "audio generation", "speech recognition",
                  "speech synthesis", "speaker verification", "music generation",
                  "speech enhancement", "asr"]),
    ("other",    ["tabular", "time-series", "forecasting", "anomaly detection",
                  "drug discovery", "protein", "genomics", "bioinformatics",
                  "gradient boosting", "xgboost", "random forest"]),
    ("general",  ["reinforcement learning", "language model", "reasoning",
                  "autonomous driving", "robotics", "embedding", "retrieval",
                  "recommendation", "code generation", "federated",
                  "knowledge distillation", "neural architecture search",
                  "meta-learning", "few-shot", "transfer learning",
                  "self-supervised", "contrastive", "graph neural",
                  "knowledge graph", "multimodal", "vision-language",
                  "diffusion", "generative", "adversarial", "pruning",
                  "quantization", "classification", "regression", "optimization"]),
]


def _keyword_area(text: str) -> str:
    t = text.lower()
    for area, kws in _AREA_KEYWORDS:
        if any(k in t for k in kws):
            return area
    return "miscellaneous"


def paper_area(item: dict) -> str:
    tasks = item.get("tasks") or []
    for t in tasks:
        name = (t if isinstance(t, str) else t.get("name", "")).strip()
        area = _keyword_area(name)
        if area != "miscellaneous":
            return area
    text = (item.get("title", "") + " " + (item.get("abstract", "") or "")[:300])
    return _keyword_area(text)


def reconstruct_abstract(inv: dict) -> str:
    if not inv:
        return ""
    pairs = []
    for word, positions in inv.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort()
    return " ".join(w for _, w in pairs)


def load_progress() -> set:
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                return set(json.load(f).get("done", []))
        except Exception:
            pass
    return set()


def save_progress(done: set):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"done": list(done)}, f)


# ── Phase 1 ────────────────────────────────────────────────────────────────────

def fetch_pwc(hf_token: str, test: bool) -> list:
    papers: list = []
    seen:   set  = set()

    if PWC_FILE.exists():
        try:
            with open(PWC_FILE, encoding="utf-8") as f:
                papers = json.load(f)
            for p in papers:
                uid = p.get("arxiv_id") or p.get("id")
                if uid:
                    seen.add(uid)
            tprint(f"[PWC] Resuming from {len(papers):,} existing papers")
        except Exception:
            papers, seen = [], set()

    try:
        from datasets import load_dataset
    except ImportError:
        tprint("[PWC] ERROR: install `datasets` package")
        return papers

    tprint("[PWC] Streaming papers-with-abstracts ...")
    collected = len(papers)
    skipped   = 0
    SAVE_EVERY = 50_000

    for item in load_dataset(PWC_ABSTRACTS_DS, split="train", streaming=True, token=hf_token):
        abstract = (item.get("abstract") or "").strip()
        if not abstract:
            skipped += 1
            continue

        arxiv_id = (item.get("arxiv_id") or "").strip()
        uid      = arxiv_id or item.get("paper_url", "").split("/")[-1]

        if not uid or uid in seen:
            skipped += 1
            continue

        lang = (item.get("language") or "en").lower()
        if lang and lang not in ("en", "english", ""):
            skipped += 1
            continue

        seen.add(uid)
        if arxiv_id:
            seen.add(arxiv_id)

        raw_date  = item.get("date")
        published = str(raw_date.date()) if hasattr(raw_date, "date") else str(raw_date or "")
        year      = published[:4] if published else ""

        papers.append({
            "id":              uid,
            "arxiv_id":        arxiv_id,
            "title":           item.get("title", ""),
            "authors":         item.get("authors") or [],
            "summary":         abstract,
            "published":       published,
            "year":            year,
            "pdf_url":         f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
            "category":        paper_area(item),
            "journal":         "",
            "doi":             (item.get("doi") or "").strip(),
            "citations":       0,
            "fields_of_study": [t if isinstance(t, str) else t.get("name", "")
                                 for t in (item.get("tasks") or [])],
            "source":          "pwc",
            "github_repos":    [],
            "has_code":        False,
        })
        collected += 1

        if collected % SAVE_EVERY == 0:
            with open(PWC_FILE, "w", encoding="utf-8") as f:
                json.dump(papers, f, ensure_ascii=False)
            tprint(f"[PWC] checkpoint {collected:,} | skipped {skipped:,}", flush=True)

        if test and collected >= 500:
            break

    with open(PWC_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False)
    tprint(f"[PWC] done — {collected:,} collected | {skipped:,} skipped")
    return papers


# ── Phase 2 ────────────────────────────────────────────────────────────────────

def fetch_openalex(test: bool) -> list:
    papers:   list = []
    seen:     set  = set()
    done_set: set  = set()

    if OA_FILE.exists():
        try:
            with open(OA_FILE, encoding="utf-8") as f:
                papers = json.load(f)
            for p in papers:
                uid = p.get("doi") or p.get("id")
                if uid:
                    seen.add(uid)
            tprint(f"[OA] Resuming from {len(papers):,} existing papers")
        except Exception:
            papers, seen = [], set()

    if OA_PROGRESS.exists():
        try:
            with open(OA_PROGRESS, encoding="utf-8") as f:
                done_set = set(json.load(f).get("done_pairs", []))
        except Exception:
            done_set = set()

    years  = OA_YEARS[:2] if test else OA_YEARS
    per_yr = 5 if test else OA_PER_YEAR
    new    = 0
    SAVE_EVERY = 1_000

    for field in OA_FIELDS:
        field_count = 0
        for year in years:
            pair = f"{field}_{year}"
            if pair in done_set:
                continue

            try:
                r = requests.get(
                    OA_BASE,
                    headers=OA_HEADERS,
                    params={
                        "filter":   f"topics.display_name:{field},publication_year:{year},has_abstract:true",
                        "sort":     "cited_by_count:desc",
                        "per_page": per_yr,
                        "select":   "id,doi,display_name,authorships,publication_year,"
                                    "primary_location,abstract_inverted_index,"
                                    "cited_by_count,topics,ids",
                    },
                    timeout=30,
                )
                if r.status_code == 429:
                    tprint("[OA] rate limited — sleeping 30s")
                    time.sleep(30)
                    continue
                if r.status_code != 200:
                    continue

                for w in r.json().get("results", []):
                    abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
                    if not abstract:
                        continue

                    doi   = (w.get("doi") or "").replace("https://doi.org/", "").strip()
                    oa_id = w.get("id", "").split("/")[-1]
                    uid   = doi or oa_id

                    if not uid or uid in seen:
                        continue
                    seen.add(uid)
                    if doi:
                        seen.add(doi)

                    authors = [
                        (a.get("author") or {}).get("display_name", "")
                        for a in (w.get("authorships") or [])
                        if (a.get("author") or {}).get("display_name")
                    ]

                    year_str = str(w.get("publication_year") or "")
                    loc      = w.get("primary_location") or {}
                    src      = loc.get("source") or {}
                    pdf_url  = loc.get("pdf_url") or loc.get("landing_page_url") or ""

                    concepts = [
                        t.get("display_name", "")
                        for t in (w.get("topics") or [])
                        if t.get("score", 0) >= 0.3
                    ]

                    ext_ids  = w.get("ids") or {}
                    arxiv_id = (ext_ids.get("arxiv") or "").replace("https://arxiv.org/abs/", "").strip()

                    papers.append({
                        "id":              uid,
                        "arxiv_id":        arxiv_id,
                        "title":           w.get("display_name", ""),
                        "authors":         authors,
                        "summary":         abstract,
                        "published":       f"{year_str}-01-01",
                        "year":            year_str,
                        "pdf_url":         pdf_url,
                        "category":        field.lower().replace(" ", "_"),
                        "journal":         src.get("display_name", ""),
                        "doi":             doi,
                        "citations":       w.get("cited_by_count", 0),
                        "fields_of_study": concepts,
                        "source":          "openalex",
                        "github_repos":    [],
                        "has_code":        False,
                    })
                    new += 1
                    field_count += 1

                done_set.add(pair)
                with open(OA_PROGRESS, "w", encoding="utf-8") as f:
                    json.dump({"done_pairs": list(done_set)}, f)

                if new % SAVE_EVERY == 0 and new > 0:
                    with open(OA_FILE, "w", encoding="utf-8") as f:
                        json.dump(papers, f, ensure_ascii=False)
                    tprint(f"[OA] checkpoint {new:,} | {field} {year}", flush=True)

                time.sleep(0.15)

            except Exception as e:
                tprint(f"[OA] error {field} {year}: {e}")

        tprint(f"[OA] {field:<25} {field_count:>5} papers")

    with open(OA_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False)
    tprint(f"[OA] done — {new:,} new papers")
    return papers


# ── Merge ──────────────────────────────────────────────────────────────────────

def merge_papers(pwc: list, oa: list) -> list:
    seen:   set  = set()
    merged: list = []
    for p in pwc + oa:
        uid = p.get("arxiv_id") or p.get("doi") or p.get("id")
        if uid and uid in seen:
            continue
        if uid:
            seen.add(uid)
        merged.append(p)
    tprint(f"[Merge] {len(pwc):,} PWC + {len(oa):,} OA → {len(merged):,} unique")
    return merged


# ── Phase 3a ───────────────────────────────────────────────────────────────────

def enrich_github(papers: list, test: bool):
    tprint("[GitHub] Loading pwc_links.json ...")
    if not PWC_LINKS_FILE.exists():
        tprint(f"[GitHub] WARNING: {PWC_LINKS_FILE} not found — skipping")
        return

    with open(PWC_LINKS_FILE, encoding="utf-8") as f:
        pwc_links: dict = json.load(f)

    arxiv_index: dict[str, int] = {
        p["arxiv_id"]: i
        for i, p in enumerate(papers)
        if p.get("arxiv_id")
    }

    matched = 0
    for arxiv_id, repos in pwc_links.items():
        if arxiv_id not in arxiv_index:
            continue
        p           = papers[arxiv_index[arxiv_id]]
        max_stars   = 0
        is_official = False
        for repo in repos:
            if not isinstance(repo, dict):
                continue
            url = repo.get("url", "").strip()
            if url and url not in p["github_repos"]:
                p["github_repos"].append(url)
            stars = repo.get("stars") or 0
            if stars > max_stars:
                max_stars = stars
            if repo.get("is_official"):
                is_official = True
        if p["github_repos"]:
            p["has_code"]    = True
            p["max_stars"]   = max_stars
            p["num_repos"]   = len(p["github_repos"])
            p["is_official"] = is_official
            matched += 1

    tprint(f"[GitHub] matched {matched:,} papers")


# ── Phase 3b ───────────────────────────────────────────────────────────────────

def enrich_citations(papers: list, test: bool):
    tprint("[Citations] Loading cache ...")
    cache: dict = {}
    if CIT_CACHE.exists():
        try:
            with open(CIT_CACHE, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    doi_index:   dict[str, dict] = {}
    arxiv_index: dict[str, dict] = {}

    for p in papers:
        if p.get("source") == "openalex":
            continue
        doi      = (p.get("doi") or "").strip()
        arxiv_id = (p.get("arxiv_id") or "").strip()

        doi_key   = f"doi:{doi}"     if doi      else None
        arxiv_key = f"arxiv:{arxiv_id}" if arxiv_id else None

        if doi_key and doi_key in cache:
            p["citations"] = cache[doi_key]
        elif arxiv_key and arxiv_key in cache:
            p["citations"] = cache[arxiv_key]
        elif doi:
            doi_index[doi] = p
        elif arxiv_id:
            arxiv_index[arxiv_id] = p

    tprint(
        f"[Citations] {len(cache):,} from cache | "
        f"{len(doi_index):,} via DOI | {len(arxiv_index):,} via arxiv"
    )

    enriched   = 0
    batch_size = 100
    SAVE_EVERY = 5_000

    doi_list = list(doi_index.keys())
    if test:
        doi_list = doi_list[:200]

    for i in range(0, len(doi_list), batch_size):
        batch = doi_list[i:i + batch_size]
        try:
            r = requests.get(
                OA_BASE,
                headers=OA_HEADERS,
                params={
                    "filter":   "doi:" + "|".join(batch),
                    "select":   "doi,cited_by_count",
                    "per_page": batch_size,
                },
                timeout=30,
            )
            if r.status_code == 429:
                time.sleep(30)
                continue
            if r.status_code == 200:
                for w in r.json().get("results", []):
                    doi = (w.get("doi") or "").replace("https://doi.org/", "").strip()
                    p   = doi_index.get(doi)
                    if p:
                        count = w.get("cited_by_count", 0)
                        p["citations"]     = count
                        cache[f"doi:{doi}"] = count
                        enriched += 1
        except Exception as e:
            tprint(f"[Citations] DOI batch error: {e}")

        time.sleep(0.15)

        if enriched > 0 and enriched % SAVE_EVERY == 0:
            with open(CIT_CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            tprint(f"[Citations] checkpoint {enriched:,} enriched", flush=True)

    arxiv_list = list(arxiv_index.keys())
    if test:
        arxiv_list = arxiv_list[:200]

    for i in range(0, len(arxiv_list), batch_size):
        batch = arxiv_list[i:i + batch_size]
        try:
            r = requests.get(
                OA_BASE,
                headers=OA_HEADERS,
                params={
                    "filter":   "ids.arxiv:" + "|".join(
                        f"https://arxiv.org/abs/{aid}" for aid in batch
                    ),
                    "select":   "ids,cited_by_count",
                    "per_page": batch_size,
                },
                timeout=30,
            )
            if r.status_code == 429:
                time.sleep(30)
                continue
            if r.status_code == 200:
                for w in r.json().get("results", []):
                    ext = w.get("ids") or {}
                    aid = (ext.get("arxiv") or "").replace("https://arxiv.org/abs/", "").strip()
                    p   = arxiv_index.get(aid)
                    if p:
                        count = w.get("cited_by_count", 0)
                        p["citations"]        = count
                        cache[f"arxiv:{aid}"] = count
                        enriched += 1
        except Exception as e:
            tprint(f"[Citations] ArXiv batch error: {e}")

        time.sleep(0.15)

        if enriched > 0 and enriched % SAVE_EVERY == 0:
            with open(CIT_CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            tprint(f"[Citations] checkpoint {enriched:,} enriched", flush=True)

    with open(CIT_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    total = len(doi_index) + len(arxiv_index)
    tprint(f"[Citations] done — {enriched:,}/{total:,} enriched")


# ── main ───────────────────────────────────────────────────────────────────────

def main(test: bool):
    hf_token = os.getenv("HF_TOKEN", "")
    done     = load_progress()

    pwc_papers: list = []
    oa_papers:  list = []

    pwc_skip = not test and "pwc_done" in done
    oa_skip  = not test and "oa_done"  in done

    if pwc_skip:
        print("[Phase 1] PWC — skipping (done)")
        with open(PWC_FILE, encoding="utf-8") as f:
            pwc_papers = json.load(f)
        print(f"  Loaded {len(pwc_papers):,} PWC papers")

    if oa_skip:
        print("[Phase 2] OpenAlex — skipping (done)")
        with open(OA_FILE, encoding="utf-8") as f:
            oa_papers = json.load(f)
        print(f"  Loaded {len(oa_papers):,} OA papers")

    fetch_needed = (not pwc_skip) or (not oa_skip)
    if fetch_needed:
        print("[Phase 1+2] Fetching PWC and OpenAlex in parallel ...")
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {}
            if not pwc_skip:
                futures["pwc"] = ex.submit(fetch_pwc, hf_token, test)
            if not oa_skip:
                futures["oa"] = ex.submit(fetch_openalex, test)

            if "pwc" in futures:
                pwc_papers = futures["pwc"].result()
                if not test:
                    done.add("pwc_done")
                    save_progress(done)

            if "oa" in futures:
                oa_papers = futures["oa"].result()
                if not test:
                    done.add("oa_done")
                    save_progress(done)

    all_papers = merge_papers(pwc_papers, oa_papers)

    github_skip = not test and "github_done" in done
    cit_skip    = not test and "citations_done" in done

    if github_skip:
        print("[Phase 3a] GitHub — skipping (done)")
    if cit_skip:
        print("[Phase 3b] Citations — skipping (done)")

    enrich_needed = (not github_skip) or (not cit_skip)
    if enrich_needed:
        print("[Phase 3a+3b] GitHub and Citations in parallel ...")
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {}
            if not github_skip:
                futures["github"] = ex.submit(enrich_github, all_papers, test)
            if not cit_skip:
                futures["citations"] = ex.submit(enrich_citations, all_papers, test)

            if "github" in futures:
                futures["github"].result()
                if not test:
                    done.add("github_done")
                    save_progress(done)

            if "citations" in futures:
                futures["citations"].result()
                if not test:
                    done.add("citations_done")
                    save_progress(done)

    with open(ENRICHED_FILE, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    pwc_c  = sum(1 for p in all_papers if p.get("source") == "pwc")
    oa_c   = sum(1 for p in all_papers if p.get("source") == "openalex")
    code_c = sum(1 for p in all_papers if p.get("has_code"))
    cit_c  = sum(1 for p in all_papers if p.get("citations", 0) > 0)

    print(f"\n{'='*60}")
    print(f"DONE — {len(all_papers):,} papers → {ENRICHED_FILE}")
    print(f"  PWC             : {pwc_c:,}")
    print(f"  OpenAlex        : {oa_c:,}")
    print(f"  With GitHub     : {code_c:,}")
    print(f"  With citations  : {cit_c:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(args.test)
