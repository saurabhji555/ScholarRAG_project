import json
import time
import math
import random
import heapq
import argparse
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=Path(__file__).parent.parent / "backend" / ".env")

# ── paths ──────────────────────────────────────────────────────────────────────
FETCH_FILE     = Path("ingestion/fetch_papers.json")       # all 565k raw papers
ENRICHED_FILE  = Path("ingestion/papers_enriched.json")    # final 20k sampled
PROGRESS_FILE  = Path("ingestion/fetch_progress.json")
PWC_LINKS_FILE = Path("ingestion/pwc_links.json")          # local pre-downloaded
SEEN_IDS_FILE  = Path("ingestion/seen_ids.json")           # lightweight dedup cache

# ── PWC ────────────────────────────────────────────────────────────────────────
PWC_ABSTRACTS_DS = "pwc-archive/papers-with-abstracts"
PWC_LINKS_DS     = "pwc-archive/links-between-paper-and-code"
PWC_TARGET       = 30_000

PWC_AREA_QUOTAS = {
    "general":        9200,
    "vision":         8000,
    "language":       5800,
    "video":          2800,
    "audio":          2100,
    "other":          1400,
    "miscellaneous":   700,
}

# ── OpenAlex (citation enrichment only) ───────────────────────────────────────
OA_BASE    = "https://api.openalex.org/works"
OA_HEADERS = {
    "User-Agent": "ScholarRAG/1.0 (mailto:24f3003029@ds.study.iitm.ac.in)",
    "Accept":     "application/json",
}
OA_SELECT = "doi,cited_by_count,concepts,primary_location"

CURRENT_YEAR      = 2025
GITHUB_MULTIPLIER = 1.5


# ── helpers ────────────────────────────────────────────────────────────────────

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


def recency_factor(year: str) -> float:
    try:
        age = max(0, CURRENT_YEAR - int(str(year)[:4]))
    except (ValueError, TypeError):
        age = 6
    return 1.0 + 1.5 * math.exp(-0.25 * age)


def star_score(stars: int) -> float:
    return math.log(max(stars, 0) + 2)


def pwc_weight(year: str, stars: int, num_repos: int, is_official: bool) -> float:
    recency  = recency_factor(year)
    repos_s  = math.log(num_repos + 1) + 1.0
    official = 2.0 if is_official else 1.0
    has_code = GITHUB_MULTIPLIER if num_repos > 0 else 1.0
    return recency * repos_s * official * has_code


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


def load_fetch_papers() -> tuple[list, set]:
    if FETCH_FILE.exists():
        import ijson
        papers = []
        seen = set()
        with open(FETCH_FILE, "rb") as f:
            for p in ijson.items(f, "item"):
                papers.append(p)
                for k in ("arxiv_id", "doi", "id"):
                    v = p.get(k, "")
                    if v:
                        seen.add(v)
                        break
        return papers, seen
    return [], set()


def load_seen_ids_only() -> set:
    if SEEN_IDS_FILE.exists():
        with open(SEEN_IDS_FILE, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_seen_ids(seen_ids: set):
    with open(SEEN_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(seen_ids), f)


def save_fetch_papers(papers: list):
    with open(FETCH_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)




# ── Phase 1: stream PWC abstracts ─────────────────────────────────────────────

def fetch_pwc(seen_ids: set, hf_token: str, test: bool) -> list:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: install `datasets` package")
        return []

    print("\n[Phase 1] Streaming PWC papers-with-abstracts ...")
    papers   = []
    skipped  = 0
    total    = 0

    for item in load_dataset(PWC_ABSTRACTS_DS, split="train", streaming=True, token=hf_token):
        abstract = (item.get("abstract") or "").strip()
        if not abstract:
            skipped += 1
            continue

        arxiv_id = (item.get("arxiv_id") or "").strip()
        uid      = arxiv_id or item.get("paper_url", "").split("/")[-1]

        if not uid or uid in seen_ids:
            skipped += 1
            continue

        # english only — PWC abstracts are mostly english but filter just in case
        lang = (item.get("language") or "en").lower()
        if lang and lang not in ("en", "english", ""):
            skipped += 1
            continue

        seen_ids.add(uid)
        if arxiv_id:
            seen_ids.add(arxiv_id)

        raw_date  = item.get("date")
        published = str(raw_date.date()) if hasattr(raw_date, "date") else str(raw_date or "")
        year      = published[:4] if published else ""

        papers.append({
            "id":               uid,
            "arxiv_id":         arxiv_id,
            "title":            item.get("title", ""),
            "authors":          item.get("authors") or [],
            "summary":          abstract,
            "published":        published,
            "year":             year,
            "pdf_url":          f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
            "category":         paper_area(item),
            "journal":          "",
            "doi":              (item.get("doi") or "").strip(),
            "citations":        0,
            "fields_of_study":  [t if isinstance(t, str) else t.get("name", "")
                                  for t in (item.get("tasks") or [])],
            "source":           "pwc",
            "github_repos":     [],
            "has_code":         False,
            "language":         "en",
        })

        total += 1
        if total % 50_000 == 0:
            print(f"  ... {total:,} PWC papers collected", flush=True)

        if test and total >= 1_000:
            break

    print(f"  PWC done: {total:,} collected | {skipped:,} skipped\n")
    return papers


# ── Phase 3: enrich with GitHub links from PWC links dataset ──────────────────

def enrich_github(papers: list, test: bool):
    print("[Phase 3a] Loading GitHub links from local pwc_links.json ...")

    if not PWC_LINKS_FILE.exists():
        print(f"  WARNING: {PWC_LINKS_FILE} not found — skipping GitHub enrichment")
        return

    with open(PWC_LINKS_FILE, encoding="utf-8") as f:
        pwc_links: dict = json.load(f)

    print(f"  Loaded {len(pwc_links):,} arxiv_id entries from pwc_links.json")

    # build arxiv_id → paper index
    arxiv_index: dict[str, int] = {}
    for i, p in enumerate(papers):
        aid = p.get("arxiv_id", "")
        if aid:
            arxiv_index[aid] = i

    matched = 0
    for arxiv_id, repos in pwc_links.items():
        if arxiv_id not in arxiv_index:
            continue
        idx = arxiv_index[arxiv_id]
        p   = papers[idx]
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
        p["has_code"]    = len(p["github_repos"]) > 0
        p["max_stars"]   = max_stars
        p["num_repos"]   = len(p["github_repos"])
        p["is_official"] = is_official
        matched += 1

    print(f"  GitHub links matched: {matched:,} papers\n")


# ── Phase 3b: enrich PWC papers with citations from OpenAlex ──────────────────

def enrich_citations_s2(papers: list, test: bool):
    print("[Phase 3b] Enriching PWC paper citations via Semantic Scholar (bulk) ...")

    s2_key     = os.getenv("S2_API_KEY", "")
    headers    = {"x-api-key": s2_key} if s2_key else {}
    s2_url     = "https://api.semanticscholar.org/graph/v1/paper/batch"
    batch_size = 500

    pwc_papers = [p for p in papers if p.get("source") == "pwc" and p.get("arxiv_id")]
    if test:
        pwc_papers = pwc_papers[:200]

    arxiv_index = {p["arxiv_id"]: p for p in pwc_papers}
    total       = len(pwc_papers)
    enriched    = 0

    print(f"  {total:,} PWC papers with arxiv_id to enrich ...")

    for i in range(0, total, batch_size):
        batch = pwc_papers[i:i + batch_size]
        ids   = [f"ARXIV:{p['arxiv_id']}" for p in batch]
        try:
            r = requests.post(
                s2_url,
                headers={**headers, "Content-Type": "application/json"},
                params={"fields": "externalIds,citationCount,influentialCitationCount,journal,fieldsOfStudy"},
                json={"ids": ids},
                timeout=30,
            )

            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                print(f"    Rate limited — sleeping {wait}s")
                time.sleep(wait)
                continue

            if r.status_code == 200:
                for w in r.json():
                    if not w:
                        continue
                    ext      = w.get("externalIds") or {}
                    arxiv_id = (ext.get("ArXiv") or "").strip()
                    p        = arxiv_index.get(arxiv_id)
                    if not p:
                        continue
                    p["citations"] = w.get("citationCount", 0)
                    if not p.get("journal") and w.get("journal"):
                        p["journal"] = w["journal"].get("name", "")
                    if not p.get("fields_of_study") and w.get("fieldsOfStudy"):
                        p["fields_of_study"] = w["fieldsOfStudy"]
                    enriched += 1
            else:
                print(f"  S2 error {r.status_code} — skipping batch")

        except Exception as e:
            print(f"  Batch error: {e}")

        if (i // batch_size) % 10 == 0:
            print(f"  ... {min(i + batch_size, total):,}/{total:,} processed | {enriched:,} enriched", flush=True)

        time.sleep(1.1)

    print(f"  Citations enriched: {enriched:,}/{total:,}\n")


# ── Phase 4: weighted sampling ─────────────────────────────────────────────────

def sample_papers(all_papers: list, test: bool) -> list:
    print("[Phase 4] Sampling final 20k PWC papers ...")

    pwc_papers = [p for p in all_papers if p.get("source") == "pwc"]
    print(f"  Pool: {len(pwc_papers):,} PWC papers")

    NO_CODE_RATIO = 0.30
    quotas = {area: max(1, q // 100) for area, q in PWC_AREA_QUOTAS.items()} if test else PWC_AREA_QUOTAS
    heaps_code:    dict[str, list] = defaultdict(list)
    heaps_nocode:  dict[str, list] = defaultdict(list)

    for p in pwc_papers:
        w    = pwc_weight(p.get("year", ""), p.get("max_stars", 0), p.get("num_repos", 0), p.get("is_official", False))
        area = p.get("category", "miscellaneous")
        key  = p.get("id", "")
        if p.get("has_code"):
            quota = max(1, int(quotas.get(area, quotas["miscellaneous"]) * (1 - NO_CODE_RATIO)))
            heap  = heaps_code[area]
        else:
            quota = max(1, int(quotas.get(area, quotas["miscellaneous"]) * NO_CODE_RATIO))
            heap  = heaps_nocode[area]
        if len(heap) < quota:
            heapq.heappush(heap, (w, key, p))
        elif w > heap[0][0]:
            heapq.heapreplace(heap, (w, key, p))

    final = (
        [p for heap in heaps_code.values()   for _, _, p in heap] +
        [p for heap in heaps_nocode.values() for _, _, p in heap]
    )
    heaps = {area: heaps_code.get(area, []) + heaps_nocode.get(area, []) for area in quotas}
    print(f"  Selected: {len(final):,}")
    for area in sorted(heaps):
        print(f"    {area:<16} {len(heaps[area]):>5}")
    return final


# ── main ───────────────────────────────────────────────────────────────────────

def main(test: bool):
    hf_token = os.getenv("HF_TOKEN", "")
    done     = load_progress()

    pwc_done = not test and "pwc_abstracts" in done

    all_papers, seen_ids = load_fetch_papers()
    print(f"Loaded {len(all_papers):,} existing papers from {FETCH_FILE}\n")

    # Phase 1 — PWC abstracts
    if pwc_done:
        print("[Phase 1] PWC abstracts — skipping (done)\n")
    else:
        pwc_papers = fetch_pwc(seen_ids, hf_token, test)
        all_papers.extend(pwc_papers)
        save_fetch_papers(all_papers)
        save_seen_ids(seen_ids)
        print(f"  Saved {len(all_papers):,} papers after PWC\n")
        if not test:
            done.add("pwc_abstracts")
            save_progress(done)

    # Phase 3a — GitHub links from PWC links dataset
    if not test and "github_enriched" in done:
        print("[Phase 3a] GitHub links — skipping (done)\n")
    else:
        enrich_github(all_papers, test)
        save_fetch_papers(all_papers)
        if not test:
            done.add("github_enriched")
            save_progress(done)

    # Phase 3b — Citations for PWC papers via OpenAlex
    if not test and "citations_enriched" in done:
        print("[Phase 3b] Citations — skipping (done)\n")
    else:
        enrich_citations_s2(all_papers, test)
        save_fetch_papers(all_papers)
        if not test:
            done.add("citations_enriched")
            save_progress(done)

    # Phase 4 — Sample final 20k
    final = sample_papers(all_papers, test)

    code_c = sum(1 for p in final if p.get("has_code"))

    with open(ENRICHED_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"DONE — {len(final):,} papers saved to {ENRICHED_FILE}")
    print(f"  With GitHub     : {code_c:,}")
    print(f"  With citations  : {sum(1 for p in final if p.get('citations', 0) > 0):,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(args.test)
