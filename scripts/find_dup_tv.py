#!/usr/bin/env python3

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def find_dup(data: dict[str, dict[str, Any]]):
    seen = defaultdict(list)
    duplicates = {}
    dup_count = 0

    for k, entry in data.items():
        if "tvdb_id" not in entry:
            continue
        if "tvdb_season" not in entry or entry["tvdb_season"] is None:
            continue
        if "tvdb_epoffset" not in entry or entry["tvdb_epoffset"] is None:
            continue

        key = (entry["tvdb_id"], entry.get("tvdb_season"), entry["tvdb_epoffset"])
        seen[key].append(k)

        if len(seen[key]) == 2:
            duplicates[key] = seen[key]
            dup_count += 2
        elif len(seen[key]) > 2:
            duplicates[key] = seen[key]
            dup_count += 1

    return duplicates, dup_count


if __name__ == "__main__":
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mappings.json")
    if not data_path.exists():
        print(f"File {data_path} does not exist.")
        exit(1)

    with data_path.open("r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error while parsing JSON: {e}")
            exit(1)

    res = find_dup(data)

    print(
        "| TVDB ID | Season | Offset | AniList IDs                                                          |"
    )
    print(
        "| ------- | ------ | ------ | -------------------------------------------------------------------- |"
    )
    for dup, keys in res[0].items():
        print(f"| {dup[0]:>7} | {dup[1]:>6} | {dup[2]:>6} | {', '.join(keys):<68} |")
    print()
    print("Total duplicates:", res[1])
