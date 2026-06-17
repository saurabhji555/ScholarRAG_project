import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=Path(__file__).parent.parent / "backend" / ".env")

GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN", "")
PWC_LINKS_FILE = Path("ingestion/pwc_links.json")
STARS_CACHE    = Path("ingestion/github_stars.json")
GRAPHQL_URL    = "https://api.github.com/graphql"
BATCH_SIZE     = 100


def parse_repo(url: str):
    url = url.rstrip("/")
    parts = url.replace("https://github.com/", "").split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None


def fetch_stars_batch(repos: list[tuple[str, str, str]]) -> dict[str, int]:
    aliases = []
    for i, (owner, name, _) in enumerate(repos):
        aliases.append(
            f'r{i}: repository(owner: "{owner}", name: "{name}") {{ stargazerCount }}'
        )
    query = "{ " + " ".join(aliases) + " }"
    headers = {
        "Authorization": f"bearer {GITHUB_TOKEN}",
        "Content-Type":  "application/json",
    }
    try:
        r = requests.post(GRAPHQL_URL, json={"query": query}, headers=headers, timeout=30)
        if r.status_code == 200:
            data = r.json().get("data") or {}
            result = {}
            for i, (_, _, url) in enumerate(repos):
                node = data.get(f"r{i}")
                if node:
                    result[url] = node.get("stargazerCount", 0)
            return result
        elif r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 60))
            print(f"  Rate limited — sleeping {wait}s")
            time.sleep(wait)
            return {}
    except Exception as e:
        print(f"  Error: {e}")
    return {}


def main():
    print("Loading pwc_links.json ...")
    with open(PWC_LINKS_FILE, encoding="utf-8") as f:
        pwc_links: dict = json.load(f)

    stars_cache: dict = {}
    if STARS_CACHE.exists():
        with open(STARS_CACHE, encoding="utf-8") as f:
            stars_cache = json.load(f)
        print(f"  Loaded {len(stars_cache):,} cached star counts")

    unique_repos = {}
    for repos in pwc_links.values():
        for repo in repos:
            if not isinstance(repo, dict):
                continue
            url = repo.get("url", "").strip()
            if url and "github.com" in url and url not in stars_cache:
                owner, name = parse_repo(url)
                if owner and name:
                    unique_repos[url] = (owner, name, url)

    print(f"  {len(unique_repos):,} repos to fetch (uncached)")

    repo_list = list(unique_repos.values())
    total     = len(repo_list)
    fetched   = 0

    for i in range(0, total, BATCH_SIZE):
        batch = repo_list[i:i + BATCH_SIZE]
        result = fetch_stars_batch(batch)
        stars_cache.update(result)
        fetched += len(result)

        if (i // BATCH_SIZE) % 50 == 0:
            print(f"  ... {min(i + BATCH_SIZE, total):,}/{total:,} | {fetched:,} fetched", flush=True)
            with open(STARS_CACHE, "w", encoding="utf-8") as f:
                json.dump(stars_cache, f)

        time.sleep(0.72)

    with open(STARS_CACHE, "w", encoding="utf-8") as f:
        json.dump(stars_cache, f)
    print(f"\n  Stars fetched: {fetched:,} repos saved to {STARS_CACHE}")

    print("\nUpdating pwc_links.json with real star counts ...")
    updated = 0
    for repos in pwc_links.values():
        for repo in repos:
            if not isinstance(repo, dict):
                continue
            url = repo.get("url", "").strip()
            if url in stars_cache:
                repo["stars"] = stars_cache[url]
                updated += 1

    with open(PWC_LINKS_FILE, "w", encoding="utf-8") as f:
        json.dump(pwc_links, f)
    print(f"  Updated {updated:,} repo entries with star counts")
    print("Done.")


if __name__ == "__main__":
    main()
