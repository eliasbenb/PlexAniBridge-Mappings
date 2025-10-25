"""Script to update episode counts from AniList, TVDB, and TMDB APIs."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from time import sleep
from typing import Any

import requests

ANILIST_API_URL = "https://graphql.anilist.co"
TMDB_API_URL = "https://api.themoviedb.org/3"
SKYHOOK_API_URL = "http://skyhook.sonarr.tv/v1/tvdb/shows/en"


def make_request_anilist(query: str, variables: dict | str | None = None) -> dict:
    """Make a request to AniList API with rate limit handling."""
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


def fetch_all_anilist_episode_counts(
    *, per_page: int = 50, pages_per_request: int = 83
) -> dict[str, int | None]:
    """Fetch all AniList anime IDs and episode counts via paginated batching."""
    per_page = max(1, min(int(per_page), 50))
    pages_per_request = max(1, int(pages_per_request))

    all_counts: dict[str, int | None] = {}
    seen: set[int] = set()
    current_page = 1

    while True:
        batch_var_defs = ["$perPage: Int!"]
        request_vars: dict[str, Any] = {"perPage": per_page}
        page_aliases: list[tuple[str, str, int]] = []

        for idx in range(pages_per_request):
            alias = f"batch{idx + 1}"
            page_var = f"page_{idx + 1}"
            page_number = current_page + idx
            batch_var_defs.append(f"${page_var}: Int!")
            request_vars[page_var] = page_number
            page_aliases.append((alias, page_var, page_number))

        query_sections = [
            f"""
            {alias}: Page(page: ${page_var}, perPage: $perPage) {{
                pageInfo {{ hasNextPage }}
                media(type: ANIME, sort: ID) {{
                    id
                    episodes
                }}
            }}
            """
            for alias, page_var, _page_number in page_aliases
        ]

        query = f"""
        query ({", ".join(batch_var_defs)}) {{
            {" ".join(query_sections)}
        }}
        """

        print(
            "Requesting AniList pages "
            f"{current_page}..{current_page + pages_per_request - 1}"
        )

        response = make_request_anilist(query, request_vars)
        data = response.get("data", {}) or {}

        stop = False
        for alias, _page_var, _page_number in page_aliases:
            page_data = data.get(alias) or {}
            media_list = page_data.get("media") or []

            for media in media_list:
                try:
                    anime_id = int(media.get("id"))
                except (TypeError, ValueError):
                    continue
                if anime_id in seen:
                    continue
                seen.add(anime_id)
                all_counts[str(anime_id)] = media.get("episodes")

            if not media_list:
                stop = True
                break

            page_info = page_data.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                stop = True
                break

        current_page += pages_per_request
        if stop:
            break

    return all_counts


def make_request_tvdb(tvdb_show_id: int | str) -> dict:
    """Make a request to TVDB API for a specific series ID."""
    response = requests.get(
        f"{SKYHOOK_API_URL}/{tvdb_show_id}",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=1)
def get_tmdb_api_key() -> str:
    """Retrieve and cache the TMDB API key from the environment."""
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TMDB_API_KEY environment variable is not set. "
            "Please provide a valid TMDB API key."
        )
    return api_key


def make_request_tmdb(tmdb_show_id: int | str) -> dict:
    """Make a request to the TMDB API with rate limit handling."""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {get_tmdb_api_key()}",
    }

    while True:
        response = requests.get(f"{TMDB_API_URL}/tv/{tmdb_show_id}", headers=headers)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 10))
            print(f"TMDB rate limit exceeded, waiting {retry_after} seconds")
            sleep(retry_after + 1)
            continue

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 10))
            print(f"TMDB rate limit exceeded, waiting {retry_after} seconds")
            sleep(retry_after + 1)
            continue

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"Error fetching TMDB endpoint {tmdb_show_id}: {response.text}")
            raise e

        return response.json()


def process_tmdb_show_id(tmdb_id: int | str) -> tuple[str, dict[str, int | None]]:
    """Process a single TMDB show ID and return its episode counts by season."""
    try:
        show_data = make_request_tmdb(tmdb_id)
        seasons = show_data.get("seasons", [])

        episode_counts = {}
        for season in seasons:
            season_number = season.get("season_number")
            if season_number is None:
                continue
            episode_count = season.get("episode_count")
            episode_counts[str(season_number)] = episode_count

        return str(tmdb_id), episode_counts
    except Exception as e:
        print(f"Error processing TMDB show ID {tmdb_id}: {e}")
        return str(tmdb_id), {}


def process_tvdb_show_id(tvdb_id: int | str) -> tuple[str, dict]:
    """Process a single TVDB show ID and return its episode counts."""
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
    """Update AniList episode counts using improved query method."""
    print("Updating AniList episode counts...")
    all_episode_counts = fetch_all_anilist_episode_counts()

    episode_counts_anilist: dict[str, int | None] = {}
    missing_ids: list[str] = []

    for anime_id in wanted_anilist:
        anime_id_str = str(anime_id)
        if anime_id_str in all_episode_counts:
            episode_counts_anilist[anime_id_str] = all_episode_counts[anime_id_str]
        else:
            missing_ids.append(anime_id_str)

    if missing_ids:
        print(
            "Warning: Missing episode counts for AniList IDs: "
            + ", ".join(sorted(missing_ids, key=int))
        )

    sorted_episode_counts_anilist = {
        k: episode_counts_anilist[k]
        for k in sorted(episode_counts_anilist.keys(), key=lambda x: int(x))
    }

    with Path("data/anilist_episode_counts.json").open("w", newline="\n") as f:
        json.dump(sorted_episode_counts_anilist, f, indent=2)

    print("AniList episode counts updated successfully!")
    return episode_counts_anilist


def update_tmdb_counts(wanted_tmdb: list[int | str]):
    """Update TMDB show episode counts."""
    if not wanted_tmdb:
        print("No TMDB show IDs found to process.")
        return {}

    print("Updating TMDB show episode counts...")

    unique_ids = sorted({int(str(tmdb_id)) for tmdb_id in wanted_tmdb})
    episode_counts_tmdb: dict[str, dict[str, int | None]] = {}

    total_ids = len(unique_ids)
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {
            executor.submit(process_tmdb_show_id, tmdb_id): tmdb_id
            for tmdb_id in unique_ids
        }

        for i, future in enumerate(as_completed(future_to_id), 1):
            tmdb_id = future_to_id[future]
            print(f"Processing TMDB show ID {tmdb_id} ({i}/{total_ids})")
            tmdb_id_str, counts = future.result()
            episode_counts_tmdb[tmdb_id_str] = counts

    sorted_episode_counts_tmdb = {
        tmdb_id: {
            season: counts[season]
            for season in sorted(counts.keys(), key=lambda x: int(x))
        }
        for tmdb_id, counts in sorted(
            episode_counts_tmdb.items(), key=lambda x: int(x[0])
        )
    }

    with Path("data/tmdb_episode_counts.json").open("w", newline="\n") as f:
        json.dump(sorted_episode_counts_tmdb, f, indent=2)

    print("TMDB show episode counts updated successfully!")
    return episode_counts_tmdb


def update_tvdb_counts(wanted_tvdb):
    """Update TVDB episode counts."""
    print("Updating TVDB episode counts...")
    episode_counts_tvdb: dict[str, dict] = {}

    total_ids = len(wanted_tvdb)
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_id = {
            executor.submit(process_tvdb_show_id, tvdb_id): tvdb_id
            for tvdb_id in wanted_tvdb
        }

        for i, future in enumerate(as_completed(future_to_id), 1):
            tvdb_id = future_to_id[future]
            print(f"Processing TVDB ID {tvdb_id} ({i}/{total_ids})")
            tvdb_id, counts = future.result()
            episode_counts_tvdb[tvdb_id] = counts

    sorted_episode_counts_tvdb = {
        tvdb_id: {
            int(season): count
            for season, count in sorted(
                episode_counts_tvdb[tvdb_id].items(), key=lambda x: int(x[0])
            )
        }
        for tvdb_id in sorted(episode_counts_tvdb.keys(), key=lambda x: int(x))
    }

    with Path("data/tvdb_episode_counts.json").open("w", newline="\n") as f:
        json.dump(sorted_episode_counts_tvdb, f, indent=2)

    print("TVDB episode counts updated successfully!")
    return episode_counts_tvdb


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update episode counts from AniList and/or TVDB"
    )
    parser.add_argument(
        "--source",
        choices=["anilist", "tmdb", "tvdb", "all"],
        default="all",
        help=(
            "Specify which source to update: anilist, tmdb, tvdb, all (default: all)"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)

    args = parse_arguments()

    with Path("mappings.json").open("r") as f:
        mappings: dict[str, dict[str, Any]] = json.load(f)

    wanted_anilist = []
    wanted_tmdb = []
    wanted_tvdb = []
    for anilist_id_str, entry in mappings.items():
        tmdb_show_id = entry.get("tmdb_show_id")
        if tmdb_show_id:
            wanted_tmdb.append(tmdb_show_id)

        tvdb_id = entry.get("tvdb_id")
        if tvdb_id:
            wanted_tvdb.append(tvdb_id)

        if tmdb_show_id or tvdb_id:
            wanted_anilist.append(anilist_id_str)

    if args.source in ("anilist", "all"):
        update_anilist_counts(wanted_anilist)

    if args.source in ("tvdb", "all"):
        update_tvdb_counts(wanted_tvdb)

    if args.source in ("tmdb", "all"):
        update_tmdb_counts(wanted_tmdb)

    completed_sources = "anilist, tvdb, tmdb" if args.source == "all" else args.source

    print(f"Update completed for source(s): {completed_sources}")
