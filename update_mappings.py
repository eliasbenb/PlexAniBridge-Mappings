import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, field_validator, model_validator

if sys.version_info < (3, 11):
    print(
        f"Version Error: Version: {sys.version_info.major}.{sys.version_info.minor}.{
            sys.version_info.micro
        } incompatible please use Python 3.11+"
    )
    sys.exit(1)

try:
    import requests
    from git import Repo
    from lxml import html
except ImportError:
    print("Requirements Error: Requirements are not installed")
    sys.exit(1)


class EpisodePattern(BaseModel):
    """Represents a parsed episode pattern with all its components."""

    season: int
    start_episode: int | None
    end_episode: int | None
    ratio: int | None

    @property
    def length(self) -> int | None:
        """Return the number of episodes in the pattern."""
        if self.end_episode is None:
            return None
        return self.end_episode - (self.start_episode or 1) + 1

    @staticmethod
    def parse(pattern: str) -> Self | None:
        """
        Parse an episode pattern string into its components.

        Args:
            pattern: A string like "s1:e1-e12|3" or "+s2:e1" or "-s1:e5|2"

        Returns:
            EpisodePattern object with parsed components or None if invalid
        """
        pattern_regex = r"""
            ^                           # Start of string
            s(?P<season>\d+):           # Season number (required)
            (?:                         # Non-capturing group for episode part
                (?:e(?P<start>\d+))?    # Optional start episode
                (?:                     # Non-capturing group for end part
                    -(?:e(?P<end>\d+))?     # Optional end episode with optional number
                )?                          # End part is optional
                |                       # OR
                -e(?P<before>\d+)           # Single episode with leading dash
            )?                          # Entire episode part is optional
            (?:\|(?P<ratio>-?\d+))?     # Optional ratio with pipe
            $                           # End of string
        """

        match = re.match(pattern_regex, pattern, re.VERBOSE)
        if not match:
            return None

        groups = match.groupdict()

        season = int(groups["season"])

        # Handle the case where we have a "before" episode number
        if groups["before"]:
            end_episode = int(groups["before"])
            start_episode = None
        else:
            start_episode = (
                int(groups["start"]) if groups["start"] is not None else None
            )
            end_episode = int(groups["end"]) if groups["end"] is not None else None

        ratio = int(groups["ratio"]) if groups["ratio"] is not None else None

        return EpisodePattern(
            season=season,
            start_episode=start_episode if start_episode is not None else 1,
            end_episode=end_episode,
            ratio=ratio if ratio is not None else 1,
        )

    @model_validator(mode="after")
    def validate_bounds(self):
        """Validate that the episode bounds are correct."""
        if self.start_episode is not None and self.end_episode is not None:
            if self.start_episode > self.end_episode:
                raise ValueError(
                    "Start episode must be less than or equal to end episode"
                )

        if self.season < 0:
            raise ValueError("Season number must be positive")

        return self

    def __repr__(self):
        start = f"e{self.start_episode}" if self.start_episode is not None else ""
        end = f"-e{self.end_episode}" if self.end_episode is not None else ""
        ratio = f"|{self.ratio}" if self.ratio is not None else ""
        return f"s{self.season}:{start}{end}{ratio}"

    def __str__(self):
        return f"{self.__class__.__name__}(season={self.season}, start_episode={self.start_episode}, end_episode={self.end_episode}, ratio={self.ratio}, length={self.length})"


class AniMap(BaseModel):
    """
    Model representing an anime mapping.

    Attributes:
        anidb_id: Optional AniDB ID
        anilist_id: Optional AniList ID
        imdb_id: Optional IMDB ID(s)
        mal_id: Optional MyAnimeList ID(s)
        tmdb_movie_id: Optional TMDB movie ID
        tmdb_show_id: Optional TMDB show ID
        tvdb_id: Optional TVDB ID
        tvdb_mappings: Optional list of TVDB mapping patterns
    """

    anidb_id: int | None = None
    anilist_id: int | None = None
    imdb_id: str | list[str] | None = None
    mal_id: int | list[int] | None = None
    tmdb_movie_id: int | list[int] | None = None
    tmdb_show_id: int | list[int] | None = None
    tvdb_id: int | None = None
    tvdb_mappings: list[str] | None = None

    @model_validator(mode="after")
    def id_validator(self):
        """Validate that at least one ID field is provided."""
        if not any(
            getattr(self, field) is not None
            for field in self.model_fields
            if field.endswith("_id")
        ):
            raise ValueError("At least one ID field must be provided")
        return self

    @field_validator("tvdb_mappings")
    def validate_tvdb_mappings(cls, v: list[str] | None) -> list[str] | None:
        """
        Validate TVDB mapping patterns.

        Args:
            v: List of mapping patterns or None

        Returns:
            Validated list of mapping patterns or None

        Raises:
            ValueError: If any mapping pattern is invalid
        """
        if v is None:
            return None

        for pattern in v:
            res = EpisodePattern.parse(pattern)
            if res is None:
                raise ValueError(f"Invalid TVDB mapping pattern: {pattern}")

        return v


class AnimeIDCollector:
    """
    A class to collect and aggregate anime IDs from various sources.

    This class handles the collection and processing of anime IDs from multiple sources
    including Anime-Lists, Manami-Project, and AnimeAggregations. It consolidates the data
    and saves it to a JSON file.
    """

    def __init__(self) -> None:
        """Initialize the AnimeIDCollector with necessary attributes and setup."""
        self.base_dir: Path = Path(__file__).parent.resolve()
        self.logger: logging.Logger = self._setup_logger()
        self.session: requests.Session = requests.Session()
        self.anime_entries: dict[int, AniMap] = {}
        self.temp_entries: dict[int, AniMap] = {}

    def _setup_logger(self) -> logging.Logger:
        """
        Set up and configure the logger.

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger("anime_ids")
        log_file = self.base_dir / "logs" / "anime_ids.log"
        log_file.parent.mkdir(exist_ok=True)

        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )
        return logger

    def _fetch_url(self, url: str, as_bytes: bool = False) -> str | bytes:
        """
        Fetch URL content with caching for frequently accessed URLs.

        Args:
            url: URL to fetch
            as_bytes: Whether to return the response as bytes

        Returns:
            Union[str, bytes]: Response content

        Raises:
            requests.RequestException: If the request fails
        """
        response = self.session.get(url)
        response.raise_for_status()
        return response.content if as_bytes else response.text

    def process_anime_lists(self) -> None:
        """
        Process anime data from Anime-Lists XML source.

        Extracts anime IDs and related information from the Anime-Lists XML file
        and stores them in temporary entries.
        """
        self.logger.info("Scanning Anime-Lists")
        content = self._fetch_url(
            "https://raw.githubusercontent.com/Anime-Lists/anime-lists/master/anime-list-master.xml",
            as_bytes=True,
        )
        root = html.fromstring(content)

        for anime in root.xpath("//anime"):
            anidb_id = str(anime.xpath("@anidbid")[0])
            if not anidb_id:
                continue

            anidb_id = int(anidb_id[1:]) if anidb_id[0] == "a" else int(anidb_id)
            entry = AniMap(anidb_id=anidb_id)

            try:
                entry.tvdb_id = int(anime.xpath("@tvdbid")[0])

                tvdb_season = str(anime.xpath("@defaulttvdbseason")[0])
                try:
                    if tvdb_season == "a":
                        # entry.tvdb_mappings = ["s-1:"]
                        continue
                    else:
                        entry.tvdb_mappings = [f"s{int(tvdb_season)}:"]

                    episode_offset = int(anime.xpath("@episodeoffset")[0])
                    if episode_offset:
                        entry.tvdb_mappings[0] += f"e{episode_offset + 1}-"
                except (ValueError, IndexError):
                    pass

            except (ValueError, IndexError):
                pass

            imdb_id = str(anime.xpath("@imdbid")[0])
            if imdb_id.startswith("tt"):
                imdb_ids = imdb_id.split(",")
                entry.imdb_id = imdb_ids[0] if len(imdb_ids) == 1 else imdb_ids

            self.temp_entries[anidb_id] = entry

    def process_manami_project(self) -> None:
        """
        Process anime data from the Manami Project.

        Extracts anime IDs from the Manami Project database and updates existing entries
        with AniList and MAL IDs.
        """
        self.logger.info("Scanning Manami-Project")
        content = json.loads(
            self._fetch_url(
                "https://raw.githubusercontent.com/manami-project/anime-offline-database/master/anime-offline-database.json"
            )
        )

        for anime in content["data"]:
            if "sources" not in anime:
                continue

            ids: dict[str, int | None] = {
                "anidb": None,
                "mal": None,
                "anilist": None,
            }

            for source in anime["sources"]:
                if "anidb.net" in source:
                    ids["anidb"] = int(source.partition("anime/")[2])
                elif "myanimelist" in source:
                    ids["mal"] = int(source.partition("anime/")[2])
                elif "anilist.co" in source:
                    ids["anilist"] = int(source.partition("anime/")[2])

            if ids["anidb"] and ids["anilist"] and ids["anidb"] in self.temp_entries:
                entry = self.temp_entries[ids["anidb"]]
                if ids["mal"]:
                    entry.mal_id = ids["mal"]
                entry.anilist_id = ids["anilist"]
                self.anime_entries[ids["anilist"]] = entry
                del self.temp_entries[ids["anidb"]]

    def process_aggregations(self) -> None:
        """
        Process anime data from AnimeAggregations.

        Updates existing entries with additional IDs from the AnimeAggregations database,
        including IMDB, MAL, and TMDB IDs.
        """
        self.logger.info("Scanning AnimeAggregations")
        content = json.loads(
            self._fetch_url(
                "https://raw.githubusercontent.com/notseteve/AnimeAggregations/main/aggregate/AnimeToExternal.json"
            )
        )

        for anidb_id_str, anime in content["animes"].items():
            anidb_id = int(anidb_id_str)

            entry = next(
                (
                    data
                    for data in self.anime_entries.values()
                    if data.anidb_id == anidb_id
                ),
                self.temp_entries.get(anidb_id),
            )

            if not entry:
                continue

            resources = anime["resources"]

            if "IMDB" in resources and not entry.imdb_id:
                entry.imdb_id = (
                    resources["IMDB"][0]
                    if len(resources["IMDB"]) == 1
                    else resources["IMDB"]
                )

            if "MAL" in resources and not entry.mal_id:
                entry.mal_id = [int(id) for id in resources["MAL"]]

            if "TMDB" in resources:
                tv_ids = [
                    int(id[3:]) for id in resources["TMDB"] if id.startswith("tv")
                ]
                movie_ids = [
                    int(id[6:]) for id in resources["TMDB"] if id.startswith("movie")
                ]

                if tv_ids and not entry.tmdb_show_id:
                    entry.tmdb_show_id = tv_ids[0] if len(tv_ids) == 1 else tv_ids
                if movie_ids and not entry.tmdb_movie_id:
                    entry.tmdb_movie_id = (
                        movie_ids[0] if len(movie_ids) == 1 else movie_ids
                    )

    def process_edits(self) -> None:
        """
        Process manual edits from mappings.edits.json.

        Applies manual corrections and additions to the collected anime entries
        from a local edits file.
        """
        self.logger.info("Scanning Anime ID Edits")
        edits_path = self.base_dir / "mappings.edits.json"

        if not edits_path.exists():
            self.logger.warning("mappings.edits.json not found")
            return

        with edits_path.open("r") as f:
            edits: dict[str, Any] = json.load(f)

        for anilist_id_str, ids in edits.items():
            if anilist_id_str.startswith("$"):
                continue
            anilist_id = int(anilist_id_str)
            if not all(key in AniMap.model_fields for key in ids.keys()):
                for key in ids.keys():
                    if key not in AniMap.model_fields:
                        self.logger.warning(
                            f"Unknown field '{key}' in edit for ID {anilist_id}"
                        )
                continue
            if anilist_id in self.anime_entries:
                existing_entry = self.anime_entries[anilist_id]
                for key, value in ids.items():
                    setattr(existing_entry, key, value)
            else:
                entry = AniMap(anilist_id=anilist_id, **ids)
                self.anime_entries[anilist_id] = entry

    def save_results(self) -> None:
        """
        Save processed anime entries to JSON file.

        Consolidates all collected anime entries and saves them to mappings.json,
        organizing them by AniList ID.
        """
        schema = {
            "title": "Anime ID Mappings",
            "type": "object",
            "patternProperties": {"^[0-9]+$": AniMap.model_json_schema()},
        }
        with Path(self.base_dir / "mappings.schema.json").open("w", newline="\n") as f:
            json.dump(schema, f, indent=2)

        for entry in self.temp_entries.values():
            if entry.anilist_id:
                self.anime_entries[entry.anilist_id] = entry

        output_dict: dict[int | str, dict[str, Any]] = {}
        for anilist_id, entry in sorted(self.anime_entries.items()):
            sorted_entry = {
                k: v
                for k, v in sorted(
                    entry.model_dump(exclude={"anilist_id"}, exclude_none=True).items()
                )
            }
            output_dict[anilist_id] = sorted_entry

        output_path = self.base_dir / "mappings.json"
        with output_path.open("w", newline="\n") as f:
            json.dump(output_dict, f, indent=2)

    def update_readme(self) -> None:
        """
        Update the README.md file with the latest generation timestamp.

        Only updates if changes were detected in JSON files.
        """
        self.logger.info("Checking for changes")
        repo = Repo(path=self.base_dir)

        if any(item.a_path.endswith(".json") for item in repo.index.diff(None)):
            self.logger.info("Saving Anime ID Changes")
            readme_path = self.base_dir / "README.md"

            with readme_path.open("r") as f:
                data = f.readlines()

            data[2] = (
                f"Last generated at: {datetime.now(UTC).strftime('%B %d, %Y %I:%M %p')} UTC\n"
            )

            with readme_path.open("w", newline="\n") as f:
                f.writelines(data)
        else:
            self.logger.info("No Anime ID Changes Detected")

    def run(self) -> None:
        """
        Execute the complete anime ID collection process.

        Runs all processing steps in sequence and handles any errors that occur.
        """
        self.logger.info("Starting Anime IDs Collection")

        try:
            self.process_anime_lists()
            self.process_manami_project()
            self.process_aggregations()
            self.process_edits()

            self.save_results()
            self.update_readme()

        except Exception as e:
            self.logger.error(f"Error during execution: {str(e)}", exc_info=True)
            sys.exit(1)

        self.logger.info("Anime IDs Collection Finished")


if __name__ == "__main__":
    collector = AnimeIDCollector()
    collector.run()
