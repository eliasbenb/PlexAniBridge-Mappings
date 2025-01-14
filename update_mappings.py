import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationInfo, field_validator, model_validator

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


class AniMap(BaseModel):
    """
    Model representing an anime mapping.

    Attributes:
        anilist_id: Optional AniList ID
        anidb_id: Optional AniDB ID
        tvdb_id: Optional TVDB ID
        tvdb_season: Optional TVDB season number
        tvdb_epoffset: Episode offset for TVDB matching
        mal_id: Optional MyAnimeList ID(s)
        imdb_id: Optional IMDB ID(s)
        tmdb_show_id: Optional TMDB show ID
        tmdb_movie_id: Optional TMDB movie ID
    """

    anilist_id: int | None = None
    anidb_id: int | None = None
    tvdb_id: int | None = None
    tvdb_season: int | None = None
    tvdb_epoffset: int | None = None
    mal_id: int | list[int] | None = None
    imdb_id: str | list[str] | None = None
    tmdb_show_id: int | list[int] | None = None
    tmdb_movie_id: int | list[int] | None = None

    @model_validator(mode="after")
    def id_validator(self):
        if not any(
            getattr(self, field) is not None
            for field in self.model_fields
            if field.endswith("_id")
        ):
            raise ValueError("At least one ID field must be provided")
        return self

    @field_validator("tvdb_id")
    def tvdb_validator(cls, v: int, info: ValidationInfo):
        if v is not None and (cls.tvdb_season is None or cls.tvdb_epoffset is None):
            raise ValueError("TVDB ID requires season and episode offset")


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

            tvdb_id = str(anime.xpath("@tvdbid")[0])
            try:
                if tvdb_id:
                    entry.tvdb_id = int(tvdb_id)
            except ValueError:
                pass

            tvdb_season = str(anime.xpath("@defaulttvdbseason")[0])
            try:
                if tvdb_season:
                    entry.tvdb_season = int(tvdb_season) if tvdb_season != "a" else -1
            except ValueError:
                pass

            try:
                entry.tvdb_epoffset = int(str(anime.xpath("@episodeoffset")[0]))
            except ValueError:
                if entry.tvdb_season is not None:
                    entry.tvdb_epoffset = 0

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
        Process manual edits from mappings_edits.json.

        Applies manual corrections and additions to the collected anime entries
        from a local edits file.
        """
        self.logger.info("Scanning Anime ID Edits")
        edits_path = self.base_dir / "mappings_edits.json"

        if not edits_path.exists():
            self.logger.warning("mappings_edits.json not found")
            return

        with edits_path.open("r") as f:
            edits = json.load(f)

        for anilist_id_str, ids in edits.items():
            anilist_id = int(anilist_id_str)
            entry = AniMap(anilist_id=anilist_id, **ids)
            self.anime_entries[anilist_id] = entry

    def save_results(self) -> None:
        """
        Save processed anime entries to JSON file.

        Consolidates all collected anime entries and saves them to mappings.json,
        organizing them by AniList ID.
        """
        ani_map_schema = AniMap.model_json_schema()
        schema = {
            "title": "Anime ID Mappings",
            "type": "object",
            "patternProperties": {"^[0-9]+$": ani_map_schema},
        }
        with Path(self.base_dir / "mappings.schema.json").open("w") as f:
            json.dump(schema, f, indent=2)

        for entry in self.temp_entries.values():
            if entry.anilist_id:
                self.anime_entries[entry.anilist_id] = entry

        output_dict: dict[int | str, dict[str, Any]] = {
            "$schema": "https://cdn.jsdelivr.net/gh/eliasbenb/PlexAniBridge-Mappings@main/mappings.schema.json",
        }
        for anilist_id, entry in sorted(self.anime_entries.items()):
            output_dict[anilist_id] = entry.model_dump(
                exclude={"anilist_id"}, exclude_none=True
            )

        output_path = self.base_dir / "mappings.json"
        with output_path.open("w") as f:
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

            data[2] = f"Last generated at: {
                datetime.now(UTC).strftime('%B %d, %Y %I:%M %p')
            } UTC\n"

            with readme_path.open("w") as f:
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
