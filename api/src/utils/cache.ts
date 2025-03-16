import { CacheEntry, AnimeMappings, Indexes } from '../types';
import { CACHE_TTL } from '../config/constants';
import { createIndexes } from './indexing';

let cache: CacheEntry | null = null;

export async function getMappings(url: string): Promise<{ mappings: AnimeMappings, indexes: Indexes }> {
    const now = Date.now();

    if (cache && now - cache.timestamp < CACHE_TTL) {
        return { mappings: cache.data, indexes: cache.indexes };
    }

    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch mappings: ${response.status} ${response.statusText}`);
    }

    const data = (await response.json()) as AnimeMappings;
    for (const [id, mapping] of Object.entries(data)) {
        mapping.anilist_id = parseInt(id);
    }
    const indexes = createIndexes(data);

    cache = {
        data,
        indexes,
        timestamp: now,
    };

    return { mappings: data, indexes };
}
