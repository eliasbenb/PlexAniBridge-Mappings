interface AnimeMappings {
    [key: string]: AniMap;
}

interface AniMap {
    anidb_id: number | null;
    anilist_id: number | null;
    imdb_id: string | string[] | null;
    mal_id: number | number[] | null;
    tmdb_movie_id: number | number[] | null;
    tmdb_show_id: number | number[] | null;
    tvdb_id: number | null;
    tvdb_mappings: Record<string, string> | null;
}

interface Indexes {
    anidb_id: Record<string, string[]>;
    anilist_id: Record<string, string[]>;
    imdb_id: Record<string, string[]>;
    mal_id: Record<string, string[]>;
    tmdb_movie_id: Record<string, string[]>;
    tmdb_show_id: Record<string, string[]>;
    tvdb_id: Record<string, string[]>;
}

interface CacheEntry {
    data: AnimeMappings;
    indexes: Indexes;
    timestamp: number;
}

export type { AnimeMappings, AniMap, CacheEntry, Indexes };
