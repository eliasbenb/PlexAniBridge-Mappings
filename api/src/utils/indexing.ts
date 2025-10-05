import { AnimeMappings, Indexes } from '../types';

export function createIndexes(mappings: AnimeMappings): Indexes {
    const indexes: Indexes = {
        anidb_id: {},
        anilist_id: {},
        imdb_id: {},
        mal_id: {},
        tmdb_movie_id: {},
        tmdb_show_id: {},
        tvdb_id: {},
    };

    const addToIndex = <T>(
        indexName: keyof Indexes,
        value: T,
        mappingId: string
    ) => {
        if (value === null || value === undefined) return;

        const key = value.toString();
        if (!indexes[indexName][key]) {
            indexes[indexName][key] = [];
        }
        indexes[indexName][key].push(mappingId);
    };

    Object.entries(mappings).forEach(([id, mapping]) => {
        addToIndex('anidb_id', mapping.anidb_id, id);
        addToIndex('anilist_id', mapping.anilist_id, id);
        addToIndex('tvdb_id', mapping.tvdb_id, id);

        if (typeof mapping.imdb_id === 'string') {
            addToIndex('imdb_id', mapping.imdb_id, id);
        } else if (Array.isArray(mapping.imdb_id)) {
            mapping.imdb_id.forEach(imdbId => addToIndex('imdb_id', imdbId, id));
        }

        if (typeof mapping.mal_id === 'number') {
            addToIndex('mal_id', mapping.mal_id, id);
        } else if (Array.isArray(mapping.mal_id)) {
            mapping.mal_id.forEach(malId => addToIndex('mal_id', malId, id));
        }

        if (typeof mapping.tmdb_movie_id === 'number') {
            addToIndex('tmdb_movie_id', mapping.tmdb_movie_id, id);
        } else if (Array.isArray(mapping.tmdb_movie_id)) {
            mapping.tmdb_movie_id.forEach(tmdbId => addToIndex('tmdb_movie_id', tmdbId, id));
        }

        if (typeof mapping.tmdb_show_id === 'number') {
            addToIndex('tmdb_show_id', mapping.tmdb_show_id, id);
        }
    });

    return indexes;
}
