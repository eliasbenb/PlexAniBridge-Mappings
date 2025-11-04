"""Script to update and manage anime ID mappings from various sources."""

import contextlib
import itertools
import json
import logging
import re
import sys
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from lxml.html import HtmlElement
from pydantic import BaseModel, Field, field_validator, model_validator
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

try:
    import requests
    from git import Repo
    from lxml import html
except ImportError:
    print("Requirements Error: Requirements are not installed")
    sys.exit(1)


class SerializationHandler:
    """Elegant handler for YAML and JSON serialization with automatic sorting."""

    def __init__(self) -> None:
        """Initialize the SerializationHandler with YAML settings."""
        self.yaml = YAML()
        self.yaml.indent(mapping=2, sequence=4, offset=2)
        self.yaml.preserve_quotes = True
        self.yaml.map_indent = 2
        self.yaml.sequence_indent = 4
        # Add custom representer for automatic sorting of regular dicts
        self.yaml.representer.add_representer(dict, self._represent_sorted_dict)

    @staticmethod
    def sort_key(key: str | int) -> tuple:
        """Universal sort key function for both YAML and JSON."""
        if isinstance(key, str):
            if key.startswith("$"):  # Special keys like $includes
                return (-1, key)
            if key.startswith("s") and key[1:].isdigit():  # Season keys like s1, s2
                return (0, int(key[1:]))
            if key.isdigit():  # Numeric string keys
                return (1, int(key))
        elif isinstance(key, int):
            return (1, key)
        return (2, str(key))  # Everything else sorted alphabetically

    def _represent_sorted_dict(self, dumper, data):
        """Custom representer that automatically sorts dictionary keys."""
        return dumper.represent_mapping(
            "tag:yaml.org,2002:map",
            sorted(data.items(), key=lambda x: self.sort_key(x[0])),
        )

    def load_yaml(self, file_path: Path) -> Any:
        """Load YAML file."""
        with file_path.open("r") as f:
            return self.yaml.load(f)

    def save_yaml(self, data: Any, file_path: Path) -> None:
        """Save YAML file with automatic sorting."""
        # Sort the data recursively while preserving structure
        sorted_data = self._sort_yaml_data(data)

        with file_path.open("w", newline="\n") as f:
            self.yaml.dump(sorted_data, f)

    def _sort_yaml_data(self, data: Any) -> Any:
        """Recursively sort YAML data while preserving CommentedMap structure."""
        if isinstance(data, CommentedMap):
            # Sort keys in place using move_to_end to preserve comments
            sorted_keys = sorted(data.keys(), key=self.sort_key)
            for key in sorted_keys:
                data.move_to_end(key)

            # Recursively sort nested structures
            for key, value in data.items():
                data[key] = self._sort_yaml_data(value)

            return data
        elif isinstance(data, dict):
            # Regular dict - convert to sorted structure
            return {
                k: self._sort_yaml_data(v)
                for k, v in sorted(data.items(), key=lambda x: self.sort_key(x[0]))
            }
        elif isinstance(data, list):
            # Sort list contents recursively
            return [self._sort_yaml_data(item) for item in data]
        else:
            return data

    def save_json(self, data: Any, file_path: Path) -> None:
        """Save JSON file with automatic sorting and formatting."""
        encoder = SortingJSONEncoder()
        with file_path.open("w", newline="\n") as f:
            f.write(encoder.encode(data))


class SortingJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder with automatic sorting and formatting."""

    def __init__(self, **kwargs) -> None:
        """Initialize the SortingJSONEncoder with indent settings."""
        super().__init__(indent=2, **kwargs)

    def encode(self, o):
        """Override encode to apply sorting recursively."""
        return super().encode(self._sort_recursively(o))

    def _sort_recursively(self, obj: Any) -> Any:
        """Recursively sort dictionaries and lists."""
        if isinstance(obj, dict):
            # Sort dictionary by keys using the same logic as YAML
            sorted_dict = {}
            for key in sorted(obj.keys(), key=SerializationHandler.sort_key):
                sorted_dict[key] = self._sort_recursively(obj[key])
            return sorted_dict
        elif isinstance(obj, list):
            # Sort lists of comparable items, otherwise preserve order
            try:
                return sorted([self._sort_recursively(item) for item in obj])
            except TypeError:
                # If items aren't comparable, just process them recursively
                return [self._sort_recursively(item) for item in obj]
        else:
            return obj


class EpisodeMapping(BaseModel, validate_assignment=True):
    """Model for storing episode mappings between external sources and AniList.

    The model is used to validate and parse episode mappings from a string pattern.
    """

    season: int = Field(ge=0, description="The source season number")
    start: int = Field(
        default=1, gt=0, description="Start of the episode range in the mapping"
    )
    end: int | None = Field(
        default=None,
        gt=0,
        description=(
            "End of the episode range in the mapping. None indicates an open-ended "
            "range."
        ),
    )
    ratio: int = Field(
        default=1,
        description="The 'worth' of each episode in the range. Positive values "
        "indicate that 1 source episode corresponds to N AniList episodes, while "
        "negative values indicate that N source episodes correspond to 1 AniList "
        "episode.",
    )

    @staticmethod
    def check_overlap(ranges: list["EpisodeMapping"]) -> bool:
        """Check if any episode ranges overlap."""
        if len(ranges) <= 1:
            return False
        sorted_ranges = sorted(
            ranges, key=lambda x: (x.start, float("inf") if x.end is None else x.end)
        )
        return any(
            curr.end is None or curr.end >= next_range.start
            for curr, next_range in itertools.pairwise(sorted_ranges)
        )

    @classmethod
    def from_string(cls, season: int, s: str) -> list[Self]:
        """Parse a string pattern into an EpisodeMapping instance.

        Args:
            season (int): Season number
            s (str): Pattern string in format
                'e{start}-e{end}|{ratio},e{start2}-e{end2}|{ratio2}'
                    Examples:
                    - 'e1-e12|2'
                    - 'e12-,e2'
                    - 'e1-e5,e8-e10'
                    - '' (empty string for full season)

        Returns:
            list[EpisodeMapping]: List of parsed EpisodeMapping instances
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

    @staticmethod
    def to_string(mappings: list["EpisodeMapping"]) -> str:
        """Convert a list of EpisodeMapping objects to their string representation.

        Args:
            mappings (list[EpisodeMapping]): List of mapping objects

        Returns:
            str: Simplified string representation
        """
        parts: list[str] = []
        for mapping in mappings:
            if mapping.start == 1 and mapping.end is None and mapping.ratio == 1:
                return ""

            if mapping.start == 1 and mapping.end is None:
                parts.append(f"e{mapping.start}-|{mapping.ratio}")
            elif mapping.start == mapping.end:
                ratio_suffix = "" if mapping.ratio == 1 else f"|{mapping.ratio}"
                parts.append(f"e{mapping.start}{ratio_suffix}")
            else:
                parts.append(
                    f"e{mapping.start}-{f'e{mapping.end}' if mapping.end else ''}"
                    f"{'' if mapping.ratio == 1 else f'|{mapping.ratio}'}"
                )
        return ",".join(parts)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        """Validate the episode range and ratio."""
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
    """Model for storing anime ID mappings and related information."""

    anidb_id: int | None = Field(
        default=None, title="AniDB ID", description="The AniDB ID"
    )
    anilist_id: int | None = Field(
        default=None, title="AniList ID", description="The AniList ID"
    )
    imdb_id: str | list[str] | None = Field(
        default=None,
        title="IMDB ID",
        description="The IMDB ID(s) (format: 'tt0123456')",
    )
    mal_id: int | list[int] | None = Field(
        default=None, title="MAL ID", description="The MyAnimeList ID(s)"
    )
    tmdb_movie_id: int | list[int] | None = Field(
        default=None, title="TMDB Movie ID", description="The TMDB movie ID(s)"
    )
    tmdb_show_id: int | None = Field(
        default=None, title="TMDB Show ID", description="The TMDB show ID"
    )
    tvdb_id: int | None = Field(
        default=None, title="TVDB ID", description="The TVDB ID"
    )
    tmdb_mappings: dict[str, str] | None = Field(
        default=None,
        title="TMDB Mappings",
        description=(
            "Mapping of TMDB seasons to episode patterns.\n\nPattern Format: "
            "'e{start}-e{end}|{ratio},e{start2}-e{end2}|{ratio2},...,e{startN}-e{endN}|{ratioN}'\n\n"
            "Attributes:\n"
            "\t- {start}: Start of the episode range\n"
            "\t- {end}: End of the episode range. None indicates an open-ended range.\n"
            "\t- {ratio}: The 'worth' of each episode in the range. Positive values "
            "indicate that 1 TMDB episode corresponds to N AniList episodes, while "
            "negative values indicate that N TMDB episodes correspond to 1 AniList "
            "episode."
        ),
        examples=[{"s1": "e1-e12|2", "s2": "e13-"}, {"s1": ""}, {"s1": "e4-e6|-2"}],
    )
    tvdb_mappings: dict[str, str] | None = Field(
        default=None,
        title="TVDB Mappings",
        description=(
            "Mapping of TVDB seasons to episode patterns.\n\nPattern Format: "
            "'e{start}-e{end}|{ratio},e{start2}-e{end2}|{ratio2},...,e{startN}-e{endN}|{ratioN}'\n\n"
            "Attributes:\n"
            "\t- {start}: Start of the episode range\n"
            "\t- {end}: End of the episode range. None indicates an open-ended range.\n"
            "\t- {ratio}: The 'worth' of each episode in the range. Positive values "
            "indicate that 1 TVDB episode corresponds to N AniList episodes, while "
            "negative values indicate that N TVDB episodes correspond to 1 AniList "
            "episode."
        ),
        examples=[{"s1": "e1-e12|2", "s2": "e13-"}, {"s1": ""}, {"s1": "e4-e6|-2"}],
    )

    @field_validator("tmdb_mappings", "tvdb_mappings")
    @classmethod
    def validate_episode_mappings(
        cls, v: dict[str, str] | None
    ) -> dict[str, str] | None:
        """Validate episode mappings for correct format and overlapping ranges."""
        if not v:
            return v

        season_groups = {}
        for season_str, mapping_str in v.items():
            season = int(season_str.lstrip("s"))
            mappings = EpisodeMapping.from_string(season, mapping_str)
            if not mappings:
                raise ValueError(f"Invalid mapping: {mapping_str}")
            season_groups.setdefault(season, []).extend(mappings)

        if any(EpisodeMapping.check_overlap(maps) for maps in season_groups.values()):
            raise ValueError("Overlapping episode ranges detected")
        return v

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Serializes the model, flattening to scalar values when possible."""
        data = super().model_dump(**kwargs)

        for key, value in data.items():
            if not key.endswith("_id"):
                continue
            if value is not None and isinstance(value, list) and len(value) == 1:
                data[key] = value[0]

        for field in ("tmdb_mappings", "tvdb_mappings"):
            if data.get(field):
                simplified_mappings = {}
                for season_str, mapping_str in data[field].items():
                    season = int(season_str.lstrip("s"))

                    mappings = EpisodeMapping.from_string(season, mapping_str)
                    simplified_str = EpisodeMapping.to_string(mappings)

                    simplified_mappings[season_str] = simplified_str

                data[field] = simplified_mappings

        return data


class ProblemEnum(StrEnum):
    """Enum for categorizing problems with anime entries."""

    EP_OVERFLOW = "AniList Episode Count Overflow (AniList > TVDB)"
    NEGATIVE_EP_OFFSET = "Negative Episode Offset"
    REDUNDANT_EDIT = "Redundant Edit in mappings.edits.yaml"
    UNKNOWN_TVDB_SEASON = "Unknown TVDB Season"
    UNKNOWN_TVDB_EP_COUNT = "Unknown TVDB Episode Count"
    UNKNOWN_ANILIST_EP_COUNT = "Unknown AniList Episode Count"
    AMBIGUOUS_TMDB_SHOW = "Ambiguous TMDB Show Mapping"


class Problem(BaseModel):
    """Model for storing problems associated with anime entries."""

    problem: ProblemEnum
    details: str

    def __eq__(self, other: Any) -> bool:
        """Equality comparison based on problem type."""
        if isinstance(other, ProblemEnum):
            return self.problem == other
        elif isinstance(other, Problem):
            return self.problem == other.problem
        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on problem type for use in sets."""
        return hash(self.problem)


class AnimeIDCollector:
    """A class to collect and aggregate anime IDs from various sources.

    This class handles the collection and processing of anime IDs from multiple sources
    including Anime-Lists, Manami-Project, and AnimeAggregations. It consolidates the
    data and saves it to a JSON file.
    """

    SCHEMA_VERSION = "v2"
    SCHEMA_URL = f"https://raw.githubusercontent.com/eliasbenb/PlexAniBridge-Mappings/{SCHEMA_VERSION}/mappings.schema.json"
    LINK_CONFIGS = (
        (
            "tvdb_id",
            "https://www.thetvdb.com/?tab=series&id={value}",
            "https://thetvdb.com/images/icon.png",
            "TVDB",
        ),
        (
            "mal_id",
            "https://myanimelist.net/anime/{value}",
            "https://myanimelist.net/favicon.ico",
            "MAL",
        ),
        (
            "anidb_id",
            "https://anidb.net/anime/{value}",
            "https://anidb.net/favicon.ico",
            "AniDB",
        ),
        (
            "imdb_id",
            "https://www.imdb.com/title/{value}",
            "https://www.imdb.com/favicon.ico",
            "IMDB",
        ),
        (
            "tmdb_movie_id",
            "https://www.themoviedb.org/movie/{value}",
            "https://www.themoviedb.org/favicon.ico",
            "TMDB Movie",
        ),
        (
            "tmdb_show_id",
            "https://www.themoviedb.org/tv/{value}",
            "https://www.themoviedb.org/favicon.ico",
            "TMDB Show",
        ),
    )

    def __init__(self) -> None:
        """Initialize the AnimeIDCollector with necessary attributes and setup."""
        self.base_dir: Path = Path(__file__).parent.resolve()
        self.logger: logging.Logger = self._setup_logger()
        self.session: requests.Session = requests.Session()
        self.generated_on: str = datetime.now(UTC).strftime("%B %d, %Y %I:%M %p")

        self.serializer = SerializationHandler()

        self.anilist_ep_counts: dict[int, int] = {}
        self.tvdb_ep_counts: dict[int, dict[str, int]] = {}

        self.anilist_entries: dict[int, AniMap] = {}
        self.anidb_entries: dict[int, AniMap] = {}

        self.problematic: dict[int, set[Problem]] = {}

        self.edits_yaml_content: CommentedMap | None = None

    def _remove_problem(self, anilist_id: int, problem_type: ProblemEnum) -> None:
        """Remove a problem of the given type for the specified AniList entry."""
        problems = self.problematic.get(anilist_id)
        if not problems:
            return

        self.problematic[anilist_id] = {
            problem for problem in problems if problem.problem != problem_type
        }

    @staticmethod
    def _icon_link(url: str, icon: str, alt: str) -> str:
        """Generate an HTML link with an icon."""
        return (
            f"<a href='{url}'><img src='{icon}' alt='{alt}' width='20' height='20'></a>"
        )

    @staticmethod
    def _iter_values(value: Any) -> list[Any]:
        """Normalize scalar or list values to a list."""
        if isinstance(value, list):
            return value
        return [value]

    def _setup_logger(self) -> logging.Logger:
        """Set up and configure the logger.

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
        """Fetch URL content with caching for frequently accessed URLs.

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

        self.logger.info("Loading Episode Counts")

        if anilist_path.exists():
            with anilist_path.open("r") as f:
                self.anilist_ep_counts = {int(k): v for k, v in json.load(f).items()}

        if tvdb_path.exists():
            with tvdb_path.open("r") as f:
                self.tvdb_ep_counts = {int(k): v for k, v in json.load(f).items()}

    def process_manami_project(self) -> None:
        """Process anime data from the Manami Project.

        Extracts anime IDs from the Manami Project database and updates existing entries
        with AniList and MAL IDs.
        """
        self.logger.info("Scanning Manami-Project")
        content = json.loads(
            self._fetch_url(
                "https://github.com/manami-project/anime-offline-database/releases/download/latest/anime-offline-database-minified.json"
            )
        )

        for anime in content["data"]:
            if "sources" not in anime:
                continue

            ids: dict = {}

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
                self.problematic[ids["anilist_id"]] = set()
                if "anidb_id" in ids:
                    self.anidb_entries[ids["anidb_id"]] = entry

    def process_anime_lists(self) -> None:
        """Process anime data from Anime-Lists XML source.

        Extracts anime IDs and related information from the Anime-Lists XML file
        and updates entries with TVDB mappings.
        """
        self.logger.info("Scanning Anime-Lists")

        def get_xpath_str(element: HtmlElement, xpath: str, default=None) -> str | None:
            try:
                value = element.xpath(xpath)[0]
                return str(value) if value else default
            except (ValueError, IndexError):
                return default

        def get_xpath_int(element: HtmlElement, xpath: str, default=None) -> int | None:
            value = get_xpath_str(element, xpath)
            try:
                return int(value) if value is not None else default
            except ValueError:
                return default

        def process_tvdb_mapping(
            entry: AniMap, tvdb_season: str, episode_offset: int
        ) -> None:
            if not tvdb_season.isdigit():
                if entry.anilist_id is not None:
                    self.problematic.setdefault(entry.anilist_id, set()).add(
                        Problem(
                            problem=ProblemEnum.UNKNOWN_TVDB_SEASON,
                            details=f"Ignored ambiguous TVDB season from Anime-Lists "
                            f"`{tvdb_season}` "
                            f"`{{anilist_id: {entry.anilist_id}, tvdb_id: "
                            f"{entry.tvdb_id}, season: {tvdb_season}}}`",
                        )
                    )
                return
            if episode_offset < 0:
                if entry.anilist_id is not None:
                    self.problematic[entry.anilist_id].add(
                        Problem(
                            problem=ProblemEnum.NEGATIVE_EP_OFFSET,
                            details=(
                                f"Ignored ambiguous negative episode offset from "
                                f"Anime-Lists `{{anilist_id: {entry.anilist_id}, "
                                f"tvdb_id: {entry.tvdb_id}, season: {tvdb_season}, "
                                f"offset: {episode_offset}}}`"
                            ),
                        )
                    )
                return

            anilist_ep_count = None
            if entry.anilist_id is not None:
                anilist_ep_count = self.anilist_ep_counts.get(entry.anilist_id)

            tvdb_ep_count = None
            if entry.tvdb_id is not None:
                tvdb_season_counts = self.tvdb_ep_counts.get(entry.tvdb_id)
                if tvdb_season_counts is not None:
                    tvdb_ep_count = tvdb_season_counts.get(tvdb_season)

            if not anilist_ep_count:
                if entry.tvdb_mappings is None:
                    entry.tvdb_mappings = {}
                entry.tvdb_mappings[f"s{tvdb_season}"] = f"e{episode_offset + 1}-"
                if entry.anilist_id is not None:
                    self.problematic[entry.anilist_id].add(
                        Problem(
                            problem=ProblemEnum.UNKNOWN_ANILIST_EP_COUNT,
                            details=(
                                f"AniList episode count is currently unknown "
                                f"`(non-issue) {{anilist_id: {entry.anilist_id}}}`"
                            ),
                        )
                    )
                return

            if not tvdb_ep_count:
                if entry.anilist_id is not None:
                    self.problematic[entry.anilist_id].add(
                        Problem(
                            problem=ProblemEnum.UNKNOWN_TVDB_EP_COUNT,
                            details=(
                                f"TVDB episode count is currently unknown (non-issue) "
                                f"`{{anilist_id: {entry.anilist_id}, tvdb_id: "
                                f"{entry.tvdb_id}, season: {tvdb_season}}}`"
                            ),
                        )
                    )
            elif anilist_ep_count > tvdb_ep_count - episode_offset:
                if entry.anilist_id is not None:
                    self.problematic[entry.anilist_id].add(
                        Problem(
                            problem=ProblemEnum.EP_OVERFLOW,
                            details=(
                                f"AniList episode count is larger than TVDB episode "
                                f"count (`{anilist_ep_count} > "
                                f"{tvdb_ep_count - episode_offset}`) `{{anilist_id: "
                                f"{entry.anilist_id}, tvdb_id: {entry.tvdb_id}, "
                                f"season: {tvdb_season}, offset: {episode_offset}}}`"
                            ),
                        )
                    )
                return

            if entry.tvdb_mappings is None:
                entry.tvdb_mappings = {}

            if episode_offset == 0 and anilist_ep_count == tvdb_ep_count:
                entry.tvdb_mappings[f"s{tvdb_season}"] = ""
            else:
                entry.tvdb_mappings[f"s{tvdb_season}"] = (
                    f"e{episode_offset + 1}-e{anilist_ep_count + episode_offset}"
                )

        def process_imdb_id(entry: AniMap, imdb_id: str) -> None:
            """Process IMDB ID for an entry."""
            if imdb_id and imdb_id.startswith("tt"):
                imdb_ids = imdb_id.split(",")
                entry.imdb_id = imdb_ids[0] if len(imdb_ids) == 1 else imdb_ids

        def process_tmdb_ids(
            entry: AniMap, tmdb_show_id: str | None, tmdb_movie_id: str | None
        ) -> None:
            if tmdb_show_id:
                tmdb_show_id = tmdb_show_id.strip()
                if tmdb_show_id.isdigit():
                    parsed_tmdb_show_id = int(tmdb_show_id)
                    if entry.tmdb_show_id is None:
                        entry.tmdb_show_id = parsed_tmdb_show_id
                        if entry.anilist_id is not None:
                            self._remove_problem(
                                entry.anilist_id, ProblemEnum.AMBIGUOUS_TMDB_SHOW
                            )
                    elif entry.tmdb_show_id != parsed_tmdb_show_id:
                        self.logger.debug(
                            (
                                "Conflicting TMDB show ID for AniDB %s "
                                "(existing: %s, anime-lists: %s)"
                            ),
                            entry.anidb_id,
                            entry.tmdb_show_id,
                            parsed_tmdb_show_id,
                        )
                else:
                    self.logger.debug(
                        "Non-numeric TMDB show ID `%s` for AniDB %s",
                        tmdb_show_id,
                        entry.anidb_id,
                    )

            if tmdb_movie_id:
                tmdb_movie_id = tmdb_movie_id.strip()
                if tmdb_movie_id.isdigit():
                    parsed_tmdb_movie_id = int(tmdb_movie_id)
                    if entry.tmdb_movie_id is None:
                        entry.tmdb_movie_id = parsed_tmdb_movie_id
                    elif isinstance(entry.tmdb_movie_id, list):
                        if parsed_tmdb_movie_id not in entry.tmdb_movie_id:
                            entry.tmdb_movie_id.append(parsed_tmdb_movie_id)
                            entry.tmdb_movie_id.sort()
                    elif entry.tmdb_movie_id != parsed_tmdb_movie_id:
                        entry.tmdb_movie_id = sorted(
                            {entry.tmdb_movie_id, parsed_tmdb_movie_id}
                        )
                else:
                    self.logger.debug(
                        "Non-numeric TMDB movie ID `%s` for AniDB %s",
                        tmdb_movie_id,
                        entry.anidb_id,
                    )

        content = self._fetch_url(
            "https://raw.githubusercontent.com/Anime-Lists/anime-lists/master/anime-list-master.xml",
            as_bytes=True,
        )
        root = html.fromstring(content)

        for anime in root.xpath("//anime"):
            anidb_id_str = get_xpath_str(anime, "@anidbid")
            if not anidb_id_str:
                continue

            anidb_id = (
                int(anidb_id_str[1:]) if anidb_id_str[0] == "a" else int(anidb_id_str)
            )
            entry = self.anidb_entries.get(anidb_id)
            if not entry:
                continue

            tvdb_id = get_xpath_int(anime, "@tvdbid")
            if tvdb_id and tvdb_id != 0:
                entry.tvdb_id = tvdb_id
                tvdb_season = get_xpath_str(anime, "@defaulttvdbseason", "a")
                episode_offset = get_xpath_int(anime, "@episodeoffset", 0)

                entry.tvdb_mappings = {}
                if tvdb_season is not None and episode_offset is not None:
                    process_tvdb_mapping(entry, tvdb_season, episode_offset)

            imdb_id = get_xpath_str(anime, "@imdbid")
            if imdb_id is not None:
                process_imdb_id(entry, imdb_id)

            process_tmdb_ids(
                entry,
                get_xpath_str(anime, "@tmdbtv"),
                get_xpath_str(anime, "@tmdbid"),
            )

    def process_aggregations(self) -> None:
        """Process anime data from AnimeAggregations.

        Updates existing entries with additional IDs from the AnimeAggregations
        database, including IMDB, MAL, and TMDB IDs.
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

            resources: dict[str, str] = anime["resources"]

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
                tv_ids = sorted(
                    {int(id[3:]) for id in resources["TMDB"] if id.startswith("tv")}
                )
                movie_ids = [
                    int(id[6:]) for id in resources["TMDB"] if id.startswith("movie")
                ]

                if tv_ids:
                    if len(tv_ids) > 1:
                        if entry.anilist_id is not None:
                            problems = self.problematic.setdefault(
                                entry.anilist_id, set()
                            )
                            details = (
                                "Multiple TMDB show IDs detected in AnimeAggregations "
                                "`{show_ids}` `{context}`"
                            ).format(
                                show_ids=tv_ids,
                                context={
                                    "anilist_id": entry.anilist_id,
                                    "anidb_id": entry.anidb_id,
                                },
                            )
                            problems.add(
                                Problem(
                                    problem=ProblemEnum.AMBIGUOUS_TMDB_SHOW,
                                    details=details,
                                )
                            )
                        else:
                            self.logger.warning(
                                (
                                    "Multiple TMDB show IDs detected in "
                                    "AnimeAggregations for AniDB %s: %s"
                                ),
                                entry.anidb_id,
                                tv_ids,
                            )
                    elif entry.tmdb_show_id is None:
                        entry.tmdb_show_id = tv_ids[0]
                        if entry.anilist_id is not None:
                            self._remove_problem(
                                entry.anilist_id,
                                ProblemEnum.AMBIGUOUS_TMDB_SHOW,
                            )

                if movie_ids and not entry.tmdb_movie_id:
                    entry.tmdb_movie_id = (
                        movie_ids[0] if len(movie_ids) == 1 else movie_ids
                    )

    def process_wikidata(self) -> None:
        """Process anime data from Wikidata SPARQL query.

        Extracts anime IDs from Wikidata using SPARQL query and updates existing entries
        with AniList, MAL, IMDB, TMDB, TVDB and other IDs.
        """
        self.logger.info("Scanning Wikidata")

        query = """
        SELECT DISTINCT ?item ?itemLabel ?anidbId ?anilistId ?malId ?imdbId ?plexId
            ?tmdbMovieId ?tmdbSeriesId ?tvdbMovieId ?tvdbSeriesId WHERE {
          ?item (p:P31/ps:P31/(wdt:P279*)) wd:Q1107.
          OPTIONAL { ?item wdt:P5646 ?anidbId. }
          ?item wdt:P8729 ?anilistId.
          OPTIONAL { ?item wdt:P4086 ?malId. }
          OPTIONAL { ?item wdt:P345 ?imdbId. }
          # OPTIONAL { ?item wdt:P11460 ?plexId. }
          OPTIONAL { ?item wdt:P4947 ?tmdbMovieId. }
          OPTIONAL { ?item wdt:P4983 ?tmdbSeriesId. }
          # OPTIONAL { ?item wdt:P12196 ?tvdbMovieId. }
          OPTIONAL { ?item wdt:P4835 ?tvdbSeriesId. }
        }
        LIMIT 10000
        """

        endpoint_url = "https://query.wikidata.org/sparql"
        params = {"query": query, "format": "json"}
        headers = {"Accept": "application/sparql-results+json"}

        response = self.session.get(endpoint_url, params=params, headers=headers)
        response.raise_for_status()
        results = response.json()

        for item in results.get("results", {}).get("bindings", []):
            try:
                anilist_id = int(item["anilistId"]["value"])
            except (KeyError, TypeError, ValueError):
                continue

            ids: dict = {"anilist_id": anilist_id}

            with contextlib.suppress(KeyError, TypeError, ValueError):
                ids["anidb_id"] = int(item["anidbId"]["value"])
            with contextlib.suppress(KeyError, TypeError, ValueError):
                ids["mal_id"] = int(item["malId"]["value"])
            with contextlib.suppress(KeyError, TypeError, ValueError):
                ids["imdb_id"] = item["imdbId"]["value"]
            with contextlib.suppress(KeyError, TypeError, ValueError):
                ids["tmdb_movie_id"] = int(item["tmdbMovieId"]["value"])
            with contextlib.suppress(KeyError, TypeError, ValueError):
                ids["tmdb_show_id"] = int(item["tmdbSeriesId"]["value"])
            with contextlib.suppress(KeyError, TypeError, ValueError):
                ids["tvdb_id"] = int(item["tvdbSeriesId"]["value"])

            entry = AniMap(**ids)

            if anilist_id in self.anilist_entries:
                existing_entry = self.anilist_entries[anilist_id]
                for key, value in ids.items():
                    if not value:
                        continue
                    curr_value = getattr(existing_entry, key)
                    if curr_value is None:
                        setattr(existing_entry, key, value)
                        if key == "tmdb_show_id":
                            self._remove_problem(
                                anilist_id, ProblemEnum.AMBIGUOUS_TMDB_SHOW
                            )
                    elif isinstance(curr_value, list):
                        if value not in curr_value:
                            self.logger.debug(
                                f"Conflicting `{key}` for ID `{anilist_id}`, `{value} "
                                f"not in `{curr_value}`"
                            )
                    elif curr_value != value:
                        self.logger.debug(
                            f"Conflicting `{key}` for ID `{anilist_id}`, `{value} != "
                            f"{curr_value}`"
                        )
            else:
                self.anilist_entries[anilist_id] = entry
                self.problematic[anilist_id] = set()

    def process_luceo(self) -> None:
        """Process anime data from Luceo's TMDB mappings.

        We only extract TMDB mappings from this source and ignore all other fields.
        """
        self.logger.info("Scanning Luceo's TMDB Mappings")
        content: dict[str, Any] = self.serializer.yaml.load(
            self._fetch_url(
                "https://raw.githubusercontent.com"
                "/LuceoEtzio/PlexAniBridge-Custom-Mappings/main/All TMDB.yaml"
            )
        )

        for anilist_id_str, fields in content.items():
            try:
                anilist_id = int(anilist_id_str)
            except (TypeError, ValueError):
                continue

            if "tmdb_mappings" not in fields:
                continue

            if anilist_id not in self.anilist_entries:
                entry = AniMap(anilist_id=anilist_id)
                self.anilist_entries[anilist_id] = entry

            entry = self.anilist_entries[anilist_id]
            try:
                entry.tmdb_mappings = fields["tmdb_mappings"]
            except (TypeError, ValueError):
                self.logger.warning(
                    f"Invalid tmdb_mappings for AniList ID `{anilist_id}`"
                )
                continue

    def process_edits(self) -> None:
        """Process manual edits from mappings.edits.yaml.

        Applies manual corrections and additions to the collected anime entries
        from a local edits file.
        """
        self.logger.info("Scanning Anime ID Edits")
        edits_path = self.base_dir / "mappings.edits.yaml"

        if not edits_path.exists():
            self.logger.warning("mappings.edits.yaml not found")
            return

        self.edits_yaml_content = self.serializer.load_yaml(edits_path)

        if not isinstance(self.edits_yaml_content, dict):
            self.logger.warning(
                "mappings.edits.yaml does not contain a valid dictionary"
            )
            return

        edits: dict[str, dict[str, Any]] = self.edits_yaml_content

        for anilist_id_str, fields in edits.items():
            if anilist_id_str is None or (
                isinstance(anilist_id_str, str) and anilist_id_str.startswith("$")
            ):
                continue
            anilist_id = int(anilist_id_str)

            skip_entry = False
            for key in fields:
                if key not in AniMap.model_fields:
                    self.logger.warning(
                        f"Unknown field `{key}` in edit for ID `{anilist_id}`"
                    )
                    skip_entry = True
            if skip_entry:
                continue

            if anilist_id in self.anilist_entries:
                existing_entry = self.anilist_entries[anilist_id]
                if anilist_id not in self.problematic:
                    self.problematic[anilist_id] = set()

                for key, value in fields.items():
                    curr_value = getattr(existing_entry, key)
                    if curr_value == value:
                        self.problematic[anilist_id].add(
                            Problem(
                                problem=ProblemEnum.REDUNDANT_EDIT,
                                details=(
                                    f"The value for `{key}` is already `{value}` "
                                    "and is redundant in 'mappings.edits.yaml' "
                                    f"`{{anilist_id: {anilist_id_str}}}`"
                                ),
                            )
                        )
                    else:
                        setattr(existing_entry, key, value)
                        if key == "tmdb_show_id" and value is not None:
                            self._remove_problem(
                                anilist_id, ProblemEnum.AMBIGUOUS_TMDB_SHOW
                            )
            else:
                entry = AniMap(anilist_id=anilist_id, **fields)
                self.anilist_entries[anilist_id] = entry
                self.problematic[anilist_id] = set()

            if (
                anilist_id in self.problematic
                and self.anilist_entries[anilist_id].tvdb_mappings
            ):
                self.problematic[anilist_id] -= {
                    ProblemEnum.EP_OVERFLOW,
                    ProblemEnum.NEGATIVE_EP_OFFSET,
                    ProblemEnum.UNKNOWN_TVDB_SEASON,
                    ProblemEnum.UNKNOWN_TVDB_EP_COUNT,
                    ProblemEnum.UNKNOWN_ANILIST_EP_COUNT,
                }

    def dump_problems(self) -> None:
        """Dump problematic entries to markdown file with detailed information."""
        self.logger.info("Dumping Problems")

        # Group problems by type
        problem_groups = {p: [] for p in ProblemEnum}
        for anilist_id, problems in self.problematic.items():
            for problem in problems:
                problem_groups[problem.problem].append((anilist_id, problem))

        markdown_content = "# PlexAniBridge Mapping Problems\n\n"
        markdown_content += f"Generated on: {self.generated_on} UTC\n\n"

        for problem_type, entries in problem_groups.items():
            if not entries:
                continue

            markdown_content += f"## {problem_type}\n\n"
            markdown_content += f"Total: {len(entries)} entries\n\n"

            markdown_content += "| AniList ID | Details | Links |\n"
            markdown_content += "|-----------|---------|-------|\n"

            for anilist_id, problem in sorted(entries, key=lambda x: x[0]):
                entry = self.anilist_entries.get(anilist_id)
                links = [
                    self._icon_link(
                        f"https://anilist.co/anime/{anilist_id}",
                        "https://anilist.co/favicon.ico",
                        "AniList",
                    )
                ]

                if entry:
                    for attr, url_template, icon, alt in self.LINK_CONFIGS:
                        attr_value = getattr(entry, attr, None)
                        if not attr_value:
                            continue
                        for value in self._iter_values(attr_value):
                            links.append(
                                self._icon_link(
                                    url_template.format(value=value), icon, alt
                                )
                            )

                markdown_content += (
                    f"| {anilist_id} | {problem.details} | {' '.join(links)} |\n"
                )

            markdown_content += "\n"

        with (self.base_dir / "problems.md").open("w", newline="\n") as f:
            f.write(markdown_content)

    def save_results(self) -> None:
        """Save processed anime entries to JSON file, organized by AniList ID."""
        # Generate and save schema
        schema = {
            "title": "Anime ID Mappings",
            "type": "object",
            "patternProperties": {"^[0-9]+$": AniMap.model_json_schema()},
            "properties": {"$includes": {"type": "array", "items": {"type": "string"}}},
        }
        self.serializer.save_json(schema, self.base_dir / "mappings.schema.json")

        # Merge entries
        if self.anidb_entries:
            self.anilist_entries.update(
                {e.anilist_id: e for e in self.anidb_entries.values() if e.anilist_id}
            )

        # Prepare output data
        output_dict = {
            str(id): entry.model_dump(exclude={"anilist_id"}, exclude_none=True)
            for id, entry in self.anilist_entries.items()
        }

        # Save JSON with automatic sorting
        self.serializer.save_json(output_dict, self.base_dir / "mappings.json")

        # Save YAML edits with automatic sorting (if they exist)
        edits_path = self.base_dir / "mappings.edits.yaml"
        if edits_path.exists() and self.edits_yaml_content is not None:
            self.serializer.save_yaml(self.edits_yaml_content, edits_path)

    def update_readme(self) -> None:
        """Update the README.md file with the generation timestamp and entry counts.

        Only updates if changes were detected in JSON files.
        """
        self.logger.info("Checking for changes")
        repo = Repo(path=self.base_dir)

        readme_path = self.base_dir / "README.md"

        mappings_count = len(self.anilist_entries)
        edits_count = 0
        if self.edits_yaml_content:
            edits_count = len(
                [
                    key
                    for key in self.edits_yaml_content
                    if not (isinstance(key, str) and key.startswith("$"))
                ]
            )

        mappings_badge = f"https://img.shields.io/badge/Mappings-{mappings_count:,}-blue?style=for-the-badge&logo=database&logoColor=white"
        edits_badge = f"https://img.shields.io/badge/Edits-{edits_count:,}-purple?style=for-the-badge&logo=pencil&logoColor=white"

        with readme_path.open("r") as f:
            data = f.readlines()

        if any(
            item.a_path and item.a_path.endswith(".json")
            for item in repo.index.diff(None)
        ):
            self.logger.info("Saving Anime ID Changes")
            data[2] = f"Last generated at: {self.generated_on} UTC\n"
        else:
            self.logger.info("No Anime ID Changes Detected")

        data[4] = f"![Mappings]({mappings_badge}) ![Edits]({edits_badge})\n"

        with readme_path.open("w", newline="\n") as f:
            f.writelines(data)

    def run(self) -> None:
        """Execute the complete anime ID collection process.

        Runs all processing steps in sequence and handles any errors that occur.
        """
        self.logger.info("Starting Anime IDs Collection")

        try:
            self.load_episode_counts()

            self.process_manami_project()
            self.process_anime_lists()
            self.process_aggregations()
            self.process_wikidata()
            self.process_luceo()
            self.process_edits()

            self.dump_problems()
            self.save_results()
            self.update_readme()

        except Exception:
            self.logger.error("Error during execution: ", exc_info=True)
            sys.exit(1)

        self.logger.info("Anime IDs Collection Finished")


if __name__ == "__main__":
    collector = AnimeIDCollector()
    collector.run()
