import { OpenAPIRoute } from 'chanfana';
import { z } from 'zod';
import { AppContext } from '../../types';
import { getMappings } from '../../utils/cache';
import { DEFAULT_CDN_URL } from '../../config/constants';
import { AniMapSchema, ErrorSchemas } from '../../types';

export class SearchMappings extends OpenAPIRoute {
    schema = {
        tags: ['Mappings'],
        operationId: 'Search Mappings',
        request: {
            query: z.object({
                anidb_id: z.number().int().optional(),
                anilist_id: z.number().int().optional(),
                imdb_id: z.string().optional(),
                mal_id: z.number().int().optional(),
                tmdb_movie_id: z.number().int().optional(),
                tmdb_show_id: z.number().int().optional(),
                tvdb_id: z.number().int().optional(),
            }),
        },
        responses: {
            '200': {
                description: 'Search results',
                content: {
                    "application/json": {
                        schema: z.object({
                            count: z.number().openapi({ description: 'Number of results' }),
                            results: z.array(AniMapSchema),
                        })
                    }
                }
            },
            '400': {
                description: 'Bad request error',
                content: {
                    "application/json": {
                        schema: ErrorSchemas.BadRequestError,
                    }
                }
            },
            '500': {
                description: 'Internal server error',
                content: {
                    "application/json": {
                        schema: ErrorSchemas.InternalServerError,
                    }
                }
            },
        },
    }

    async handle(c: AppContext) {
        try {
            const data = await this.getValidatedData<typeof this.schema>();
            const query = data.query;

            if (Object.keys(query).length === 0) {
                return c.json({
                    error: 'At least one search parameter is required',
                }, 400);
            }

            const url = c.env.CDN_URL || DEFAULT_CDN_URL;
            const { mappings, indexes } = await getMappings(url);

            const resultIds = new Map<string, boolean>();
            let hasResults = false;

            const queryParams: [keyof typeof query, keyof typeof indexes][] = [
                ['anidb_id', 'anidb_id'],
                ['anilist_id', 'anilist_id'],
                ['imdb_id', 'imdb_id'],
                ['mal_id', 'mal_id'],
                ['tmdb_movie_id', 'tmdb_movie_id'],
                ['tmdb_show_id', 'tmdb_show_id'],
                ['tvdb_id', 'tvdb_id']
            ];

            for (const [queryKey, indexKey] of queryParams) {
                const queryValue = query[queryKey];
                if (!queryValue) continue;

                const matchingIds = indexes[indexKey][queryValue];
                if (!matchingIds) continue;

                if (!hasResults) {
                    matchingIds.forEach(id => resultIds.set(id, true));
                    hasResults = true;
                    continue;
                }

                matchingIds.forEach(id => resultIds.set(id, true));
            }

            const results = Array.from(resultIds.keys()).map(id => mappings[id]);

            return c.json({
                count: results.length,
                results,
            });
        } catch (error) {
            console.error('Search mappings error:', error);
            return c.json({
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            }, 500);
        }
    }
}
