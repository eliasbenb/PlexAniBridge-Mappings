import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep
from typing import Any

import requests

ANILIST_API_URL = "https://graphql.anilist.co"
SKYHOOK_API_URL = "http://skyhook.sonarr.tv/v1/tvdb/shows/en"


def create_batch_queries_anilist(
    ids: list[str | int], batch_size: int = 250
) -> list[str]:
    batches = []
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        query = f"""
query {{
{"\n".join([f"    a{k}: Media(id: {k}) {{episodes}}" for k in batch])}
}}
            """
        batches.append(query)
    return batches


def make_request_anilist(query: str, variables: dict | str | None = None) -> dict:
    response = requests.post(
        ANILIST_API_URL,
        headers={
            # "Authorization": f"Bearer {ANILIST_API_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={"query": query, "variables": variables or {}},
    )
    if response.status_code == 429:  # Handle rate limit retries
        retry_after = int(response.headers.get("Retry-After", 60))
        print(f"Rate limit exceeded, waiting {retry_after} seconds")
        sleep(retry_after + 1)
        return make_request_anilist(query, variables)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Error: {response.text}")
        raise e
    return response.json()


def make_request_tvdb(tvdb_id: str | int) -> dict:
    """Make a request to TVDB API for a specific series ID"""
    response = requests.get(
        f"{SKYHOOK_API_URL}/{tvdb_id}",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    response.raise_for_status()
    return response.json()


def process_tvdb_id(tvdb_id: int | str) -> tuple[str, dict]:
    """Process a single TVDB ID and return its episode counts"""
    try:
        series_data = make_request_tvdb(tvdb_id)
        seasons = series_data["seasons"]
        episode_counts = {
            season["seasonNumber"]: sum(
                1
                for e in series_data["episodes"]
                if e["seasonNumber"] == season["seasonNumber"]
            )
            for season in seasons
        }
        return str(tvdb_id), episode_counts
    except Exception as e:
        print(f"Error processing TVDB ID {tvdb_id}: {e}")
        return str(tvdb_id), {}


def update_anilist_counts(wanted_anilist: list[int | str]):
    """Update AniList episode counts"""
    print("Updating AniList episode counts...")
    episode_counts_anilist: dict[str, int] = {}
    batch_queries_anilist = create_batch_queries_anilist(wanted_anilist)
    for i, query in enumerate(batch_queries_anilist):
        print(f"Executing AniList batch {i + 1}/{len(batch_queries_anilist)}")
        response = make_request_anilist(query)
        data: dict[str, dict[str, dict[str, int]]] = response.get("data", {})
        for k, v in data.items():
            episode_counts_anilist[k.lstrip("a")] = v.get("episodes")

    with Path("data/anilist_episode_counts.json").open("w", newline="\n") as f:
        json.dump(episode_counts_anilist, f, indent=2)

    print("AniList episode counts updated successfully!")
    return episode_counts_anilist


def update_tvdb_counts(wanted_tvdb):
    """Update TVDB episode counts"""
    print("Updating TVDB episode counts...")
    episode_counts_tvdb: dict[str, dict] = {}

    total_ids = len(wanted_tvdb)
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_id = {
            executor.submit(process_tvdb_id, tvdb_id): tvdb_id
            for tvdb_id in wanted_tvdb
        }

        for i, future in enumerate(as_completed(future_to_id), 1):
            tvdb_id = future_to_id[future]
            print(f"Processing TVDB ID {tvdb_id} ({i}/{total_ids})")
            tvdb_id, counts = future.result()
            episode_counts_tvdb[tvdb_id] = counts

    with Path("data/tvdb_episode_counts.json").open("w", newline="\n") as f:
        json.dump(episode_counts_tvdb, f, indent=2)

    print("TVDB episode counts updated successfully!")
    return episode_counts_tvdb


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Update episode counts from AniList and/or TVDB"
    )
    parser.add_argument(
        "--source",
        choices=["anilist", "tvdb", "both"],
        default="both",
        help="Specify which source to update: anilist, tvdb, or both (default: both)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)

    args = parse_arguments()

    mappings_path = Path("mappings.json")
    with mappings_path.open("r") as f:
        mappings: dict[str, dict[str, Any]] = json.load(f)

    wanted_anilist = []
    wanted_tvdb = []
    for anilist_id_str, entry in mappings.items():
        tvdb_id = entry.get("tvdb_id")
        if tvdb_id:
            wanted_anilist.append(anilist_id_str)
            wanted_tvdb.append(tvdb_id)

    if args.source in ["anilist", "both"]:
        update_anilist_counts(wanted_anilist)

    if args.source in ["tvdb", "both"]:
        update_tvdb_counts(wanted_tvdb)

    print(f"Update completed for source(s): {args.source}")
