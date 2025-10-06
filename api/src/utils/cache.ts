import rawMappingsJson from '../../../mappings.json';
import { AnimeMappings, AniMap, Indexes } from '../types';
import { createIndexes } from './indexing';

type RawAniMap = Partial<Omit<AniMap, 'anilist_id'>> & { anilist_id?: number | null };
type RawMappings = Record<string, RawAniMap>;

const rawMappings = rawMappingsJson as unknown as RawMappings;

function normalizeMappings(source: RawMappings): AnimeMappings {
    const normalized: AnimeMappings = {};

    for (const [id, mapping] of Object.entries(source)) {
        const parsedId = Number.parseInt(id, 10);

        normalized[id] = {
            anidb_id: mapping.anidb_id ?? null,
            anilist_id: mapping.anilist_id ?? (Number.isNaN(parsedId) ? null : parsedId),
            imdb_id: mapping.imdb_id ?? null,
            mal_id: mapping.mal_id ?? null,
            tmdb_movie_id: mapping.tmdb_movie_id ?? null,
            tmdb_show_id: mapping.tmdb_show_id ?? null,
            tvdb_id: mapping.tvdb_id ?? null,
            tvdb_mappings: mapping.tvdb_mappings ?? null,
        };
    }

    return normalized;
}

const mappings = normalizeMappings(rawMappings);
const indexes = createIndexes(mappings);

export async function getMappings(): Promise<{ mappings: AnimeMappings, indexes: Indexes }> {
    return { mappings, indexes };
}
