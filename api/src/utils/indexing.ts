import { AnimeMappings, Indexes, AniMap, Condition, Filter } from '../types';

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
        } else if (Array.isArray(mapping.tmdb_show_id)) {
            mapping.tmdb_show_id.forEach(tmdbId => addToIndex('tmdb_show_id', tmdbId, id));
        }
    });

    return indexes;
}

export function evaluateCondition(mapping: AniMap, condition: Condition): boolean {
    const { field, op, value } = condition;
    const mappingValue = mapping[field];

    if (op === "eq") {
        if (Array.isArray(mappingValue)) {
            return mappingValue.some((v) => v === value);
        }
        return mappingValue === value;
    } else if (op === "neq") {
        if (Array.isArray(mappingValue)) {
            return !mappingValue.some((v) => v === value);
        }
        return mappingValue !== value;
    } else if (op === "in") {
        if (!Array.isArray(value)) {
            throw new Error(`Operator 'in' requires the value to be an array`);
        }
        if (Array.isArray(mappingValue)) {
            return mappingValue.some((v) => value.includes(v as any));
        }
        return value.includes(mappingValue as any);
    } else if (op === "nin") {
        if (!Array.isArray(value)) {
            throw new Error(`Operator 'nin' requires the value to be an array`);
        }
        if (Array.isArray(mappingValue)) {
            return !mappingValue.some((v) => value.includes(v as any));
        }
        return !value.includes(mappingValue as any);
    } else if (op === "gt") {
        if (Array.isArray(mappingValue)) {
            return mappingValue.some((v) => Number(v) > Number(value));
        }
        return Number(mappingValue) > Number(value);
    } else if (op === "gte") {
        if (Array.isArray(mappingValue)) {
            return mappingValue.some((v) => Number(v) >= Number(value));
        }
        return Number(mappingValue) >= Number(value);
    } else if (op === "lt") {
        if (Array.isArray(mappingValue)) {
            return mappingValue.some((v) => Number(v) < Number(value));
        }
        return Number(mappingValue) < Number(value);
    } else if (op === "lte") {
        if (Array.isArray(mappingValue)) {
            return mappingValue.some((v) => Number(v) <= Number(value));
        }
        return Number(mappingValue) <= Number(value);
    }
    return false;
}

export function evaluateFilter(mapping: AniMap, filter: Filter): boolean {
    if ("and" in filter) {
        return filter.and.every((f) => evaluateFilter(mapping, f));
    } else if ("or" in filter) {
        return filter.or.some((f) => evaluateFilter(mapping, f));
    } else {
        return evaluateCondition(mapping, filter);
    }
}

export function getInitialCandidates(condition: Condition, indexes: Indexes): Set<string> | null {
    const { field, op, value } = condition;

    if (op === "eq" && typeof value === 'string' || typeof value === 'number') {
        const key = value.toString();
        return indexes[field][key] ? new Set(indexes[field][key]) : new Set();
    }

    if (op === "in" && Array.isArray(value)) {
        const candidates = new Set<string>();
        value.forEach(val => {
            const key = val.toString();
            if (indexes[field][key]) {
                indexes[field][key].forEach(id => candidates.add(id));
            }
        });
        return candidates;
    }

    return null;
}
