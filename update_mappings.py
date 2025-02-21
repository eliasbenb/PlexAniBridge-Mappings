import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator

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


class AnimeIDCollector:
    """
    A class to collect and aggregate anime IDs from various sources.

    This class handles the collection and processing of anime IDs from multiple sources
    including Anime-Lists, Manami-Project, and AnimeAggregations. It consolidates the data
    and saves it to a JSON file.
    """

    SCHEMA_VERSION = "v2"
    SCHEMA_URL = f"https://raw.githubusercontent.com/eliasbenb/PlexAniBridge-Mappings/{SCHEMA_VERSION}/mappings.schema.json"

    def __init__(self) -> None:
        """Initialize the AnimeIDCollector with necessary attributes and setup."""
        self.base_dir: Path = Path(__file__).parent.resolve()
        self.logger: logging.Logger = self._setup_logger()
        self.session: requests.Session = requests.Session()

        self.anilist_ep_counts: dict[int, int] = {}
        self.tvdb_ep_counts: dict[int, dict[str, int]] = {}

        self.anilist_entries: dict[int, AniMap] = {}
        self.anidb_entries: dict[int, AniMap] = {}

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
            str | bytes: Response content

        Raises:
            requests.RequestException: If the request fails
        """
        response = self.session.get(url)
        response.raise_for_status()
        return response.content if as_bytes else response.text

    def load_episode_counts(self) -> None:
        """Load the episode counts for AniList and TVDB IDs from the data folder."""
        anilist_path = self.base_dir / "data" / "anilist_episode_counts.json"
        tvdb_path = self.base_dir / "data" / "tvdb_episode_counts.json"

        self.logger.info("Loading episode counts")

        if anilist_path.exists():
            with anilist_path.open("r") as f:
                self.anilist_ep_counts = {int(k): v for k, v in json.load(f).items()}

        if tvdb_path.exists():
            with tvdb_path.open("r") as f:
                self.tvdb_ep_counts = {int(k): v for k, v in json.load(f).items()}

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

            ids: dict[str, int] = {}

            for source in anime["sources"]:
                if "anidb.net" in source:
                    ids["anidb_id"] = int(source.partition("anime/")[2])
                elif "myanimelist" in source:
                    ids["mal_id"] = int(source.partition("anime/")[2])
                elif "anilist.co" in source:
                    ids["anilist_id"] = int(source.partition("anime/")[2])

            if not ids:
                continue
            entry = AniMap(**ids)

            if "anilist_id" in ids:
                self.anilist_entries[ids["anilist_id"]] = entry
            if "anidb_id" in ids:
                self.anidb_entries[ids["anidb_id"]] = entry

    def process_anime_lists(self) -> None:
        """
        Process anime data from Anime-Lists XML source.

        Extracts anime IDs and related information from the Anime-Lists XML file
        and updates entries with TVDB mappings.
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

            entry = self.anidb_entries.get(anidb_id, AniMap(anidb_id=anidb_id))

            try:
                tvdb_id_str = anime.xpath("@tvdbid")[0]
                if tvdb_id_str and tvdb_id_str != "0":
                    entry.tvdb_id = int(tvdb_id_str)

                    try:
                        tvdb_season = str(anime.xpath("@defaulttvdbseason")[0])
                    except (ValueError, IndexError):
                        tvdb_season = "a"
                    try:
                        episode_offset = int(anime.xpath("@episodeoffset")[0])
                    except (ValueError, IndexError):
                        episode_offset = 0

                    entry.tvdb_mappings = {}

                    if tvdb_season == "a":
                        pass
                    elif (
                        entry.anilist_id
                        and self.anilist_ep_counts.get(entry.anilist_id) is not None
                    ):
                        anilist_ep_count = self.anilist_ep_counts[entry.anilist_id]
                        tvdb_ep_count = self.tvdb_ep_counts.get(entry.tvdb_id, {}).get(
                            tvdb_season
                        )

                        if tvdb_ep_count is None:
                            pass
                        elif anilist_ep_count > tvdb_ep_count - episode_offset:
                            logging.debug(
                                f"AniList entry {entry.anilist_id} needs more episodes than TVDB entry {entry.tvdb_id} has available "
                                f"({anilist_ep_count} > {tvdb_ep_count - episode_offset}). Manual mapping may be required."
                            )
                        elif episode_offset == 0 and anilist_ep_count == tvdb_ep_count:
                            entry.tvdb_mappings[f"s{tvdb_season}"] = ""
                        elif anilist_ep_count == 1:
                            entry.tvdb_mappings[f"s{tvdb_season}"] = (
                                f"e{episode_offset + 1}"
                            )
                        else:
                            entry.tvdb_mappings[f"s{tvdb_season}"] = (
                                f"e{episode_offset + 1}-e{anilist_ep_count + episode_offset}"
                            )
                    else:
                        entry.tvdb_mappings[f"s{tvdb_season}"] = (
                            f"e{episode_offset + 1}-"
                        )
            except (ValueError, IndexError):
                pass

            try:
                imdb_id = str(anime.xpath("@imdbid")[0])
                if imdb_id.startswith("tt"):
                    imdb_ids = imdb_id.split(",")
                    entry.imdb_id = imdb_ids[0] if len(imdb_ids) == 1 else imdb_ids
            except (ValueError, IndexError):
                pass

            self.anidb_entries[anidb_id] = entry
            if entry.anilist_id:
                self.anilist_entries[entry.anilist_id] = entry

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
            entry = self.anidb_entries.get(anidb_id)

            if not entry:
                continue

            resources = anime["resources"]

            if "IMDB" in resources:
                existing_imdb = (
                    [entry.imdb_id]
                    if isinstance(entry.imdb_id, str)
                    else (entry.imdb_id or [])
                )
                entry.imdb_id = list(set(existing_imdb) | set(resources["IMDB"]))

            if "MAL" in resources:
                existing_mal = (
                    [entry.mal_id]
                    if isinstance(entry.mal_id, int)
                    else (entry.mal_id or [])
                )
                entry.mal_id = list(
                    set(existing_mal) | set(int(mid) for mid in resources["MAL"])
                )

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
            edits: dict[str, dict[str, Any]] = json.load(f)

        for anilist_id_str, fields in edits.items():
            if anilist_id_str.startswith("$"):
                continue
            anilist_id = int(anilist_id_str)

            skip_entry = False
            for key in fields.keys():
                if key not in AniMap.model_fields:
                    self.logger.warning(
                        f"Unknown field '{key}' in edit for ID {anilist_id}"
                    )
                    skip_entry = True
            if skip_entry:
                continue

            if anilist_id in self.anilist_entries:
                existing_entry = self.anilist_entries[anilist_id]
                for key, value in fields.items():
                    if key == "tvdb_mappings" and value:
                        value = {**(existing_entry.tvdb_mappings or {}), **value}
                    setattr(existing_entry, key, value)
            else:
                entry = AniMap(anilist_id=anilist_id, **fields)
                self.anilist_entries[anilist_id] = entry

    def save_results(self) -> None:
        """
        Save processed anime entries to JSON file.

        Consolidates all collected anime entries and saves them to mappings.json,
        organizing them by AniList ID.
        """

        def sort_value(v: Any) -> Any:
            match v:
                case dict():

                    def key_sorter(item):
                        k = item[0]
                        if isinstance(k, str) and k.startswith("s") and k[1:].isdigit():
                            return (0, int(k[1:]))  # Sort by the numeric part
                        elif isinstance(k, str) and k.isdigit():
                            return (
                                1,
                                int(k),
                            )
                        return (2, k)

                    sorted_items = sorted(v.items(), key=key_sorter)
                    return {k: sort_value(v) for k, v in sorted_items}
                case list():
                    return sorted(map(sort_value, v))
                case _:
                    return v

        schema = {
            "title": "Anime ID Mappings",
            "type": "object",
            "patternProperties": {"^[0-9]+$": AniMap.model_json_schema()},
        }
        with Path(self.base_dir / "mappings.schema.json").open("w", newline="\n") as f:
            json.dump(schema, f, indent=2)

        if self.anidb_entries:
            self.anilist_entries.update(
                {
                    entry.anilist_id: entry
                    for entry in self.anidb_entries.values()
                    if entry.anilist_id
                }
            )

        output_dict = {
            str(anilist_id): sort_value(
                entry.model_dump(exclude={"anilist_id"}, exclude_none=True)
            )
            for anilist_id, entry in sorted(self.anilist_entries.items())
        }

        output_path = self.base_dir / "mappings.json"
        with output_path.open("w", newline="\n") as f:
            json.dump(output_dict, f, indent=2)

        edits_path = self.base_dir / "mappings.edits.json"
        if edits_path.exists():
            with edits_path.open("r") as f:
                edits = json.load(f)
                edits.pop("$schema", None)
                edits = sort_value(edits)
                edits = {"$schema": self.SCHEMA_URL, **edits}

            with edits_path.open("w", newline="\n") as f:
                json.dump(edits, f, indent=2)

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
            self.load_episode_counts()

            self.process_manami_project()
            self.process_anime_lists()
            self.process_aggregations()
            self.process_edits()

            self.save_results()
            self.update_readme()

        except Exception as e:
            self.logger.error(f"Error during execution: {str(e)}", exc_info=True)
            sys.exit(1)

        self.logger.info("Anime IDs Collection Finished")


class TVDBMapping(BaseModel, validate_assignment=True):
    """Model for parsing and validating TVDB episode mapping patterns.

    Handles conversion between string patterns and episode mapping objects.
    """

    season: int = Field(ge=0)
    start: int = Field(default=1, gt=0)
    end: int | None = Field(default=None, gt=0)
    ratio: int = Field(default=1)

    @staticmethod
    def check_overlap(ranges: list["TVDBMapping"]) -> bool:
        """Check if any episode ranges overlap."""
        if len(ranges) <= 1:
            return False
        sorted_ranges = sorted(
            ranges, key=lambda x: (x.start, float("inf") if x.end is None else x.end)
        )
        return any(
            curr.end is None or curr.end >= next_range.start
            for curr, next_range in zip(sorted_ranges, sorted_ranges[1:])
        )

    @classmethod
    def from_string(cls, season: int, s: str) -> list[Self]:
        """Parse a string pattern into a TVDBMapping instance.
        Args:
            season (int): Season number
            s (str): Pattern string in format 'e{start}-e{end}|{ratio},e{start2}-e{end2}|{ratio2}'
                    Examples:
                    - 'e1-e12|2'
                    - 'e12-,e2'
                    - 'e1-e5,e8-e10'
                    - '' (empty string for full season)
        Returns:
            Self | None: New TVDBMapping instance if pattern is valid, None otherwise
        """
        PATTERN = re.compile(
            r"""
            (?:^|,)
            (?:
                (?P<is_ep_range>                # Episode range (e.g. e1-e4)
                    e(?P<range_start>\d+)
                    -
                    e(?P<range_end>\d+)
                )
                |
                (?P<is_open_ep_range_after>     # Open range after (e.g. e1-)
                    e(?P<after_start>\d+)-(?=\||$|,)
                )
                |
                (?P<is_single_ep>               # Single episode (e.g. e2)
                    e(?P<single_ep>\d+)(?!-)
                )
                |
                (?P<is_open_ep_range_before>    # Open range before (e.g. -e5)
                    -e(?P<before_end>\d+)
                )
            )
            (?:\|(?P<ratio>-?\d+))?            # Optional ratio for each range
            """,
            re.VERBOSE,
        )

        if not s:
            return [cls(season=season)]

        range_matches = list(PATTERN.finditer(s))

        episode_ranges = []
        for match in range_matches:
            groups = match.groupdict()
            ratio = int(groups["ratio"]) if groups["ratio"] else 1

            # Explicit start and end episode range
            if groups["is_ep_range"]:
                start = int(groups["range_start"])
                end = int(groups["range_end"])
            # Single episode
            elif groups["is_single_ep"]:
                start = end = int(groups["single_ep"])
            # Open range with unknown start and explicit end
            elif groups["is_open_ep_range_before"]:
                start = 1
                end = int(groups["before_end"])
            # Open range with explicit start and unknown end
            elif groups["is_open_ep_range_after"]:
                start = int(groups["after_start"])
                end = None
            else:
                continue

            episode_ranges.append(cls(season=season, start=start, end=end, ratio=ratio))

        return episode_ranges

    def __str__(self) -> str:
        season = f"S{self.season:02d}"
        if self.start == 1 and self.end is None:
            return season
        result = f"{season}E{self.start:02d}"
        return result + (
            "+"
            if self.end is None and self.start != 1
            else f"-E{self.end:02d}"
            if self.end and self.end != self.start
            else ""
        )

    def __hash__(self) -> int:
        return hash(repr(self))

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.ratio == 0:
            raise ValueError("Ratio must not be zero")
        if self.end is not None:
            if self.start > self.end:
                raise ValueError(
                    "Start episode must be less than or equal to end episode"
                )
            if self.ratio > 0 and (self.end - self.start + 1) % self.ratio != 0:
                raise ValueError(
                    "A positive ratio must divide the episode range evenly"
                )
        return self


class AniMap(BaseModel, validate_assignment=True):
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
    tvdb_mappings: dict[str, str] | None = None

    @field_validator("tvdb_mappings")
    @classmethod
    def validate_tvdb_mappings(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if not v:
            return v

        season_groups = {}
        for season_str, mapping_str in v.items():
            season = int(season_str.lstrip("s"))
            mappings = TVDBMapping.from_string(season, mapping_str)
            if not mappings:
                raise ValueError(f"Invalid mapping: {mapping_str}")
            season_groups.setdefault(season, []).extend(mappings)

        if any(TVDBMapping.check_overlap(maps) for maps in season_groups.values()):
            raise ValueError("Overlapping episode ranges detected")
        return v

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """
        Dumps the model to a dictionary, flattening single-item lists to scalar values.
        """
        data = super().model_dump(**kwargs)

        for key, value in data.items():
            if not key.endswith("_id"):
                continue
            if value is not None and isinstance(value, list) and len(value) == 1:
                data[key] = value[0]

        return data


if __name__ == "__main__":
    collector = AnimeIDCollector()
    collector.run()
