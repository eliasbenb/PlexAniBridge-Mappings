import { OpenAPIRoute } from 'chanfana';
import { z } from 'zod';
import { AppContext } from '../types';
import { getMappings } from '../utils/cache';
import { DEFAULT_CDN_URL } from '../config/constants';
import { AniMapSchema, ErrorSchemas } from '../types';

export class GetAllMappings extends OpenAPIRoute {
    schema = {
        request: {
            query: z.object({
                page: z.number().int().min(1).optional().default(1),
                limit: z.number().int().min(-1).optional().default(500),
            }),
        },
        responses: {
            '200': {
                description: 'List of all mappings',
                content: {
                    "application/json": {
                        schema: z.object({
                            count: z.number(),
                            results: AniMapSchema,
                            pageInfo: z.object({
                                totalPages: z.number(),
                                currentPage: z.number(),
                                pageSize: z.number(),
                                hasNextPage: z.boolean(),
                            }),
                        })
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
            const url = c.env.CDN_URL || DEFAULT_CDN_URL;
            const { mappings } = await getMappings(url);

            const { page, limit } = (await this.getValidatedData<typeof this.schema>()).query;
            const start = (page - 1) * limit;
            const end = limit === -1 ? Object.keys(mappings).length : start + limit;

            const results = Object.values(mappings).slice(start, end);
            return c.json({
                count: results.length,
                results,
                pageInfo: {
                    totalPages: Math.ceil(Object.keys(mappings).length / limit),
                    currentPage: page,
                    pageSize: limit,
                    hasNextPage: end < Object.keys(mappings).length,
                },
            });
        } catch (error) {
            return c.json({
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            }, 500);
        }
    }
}
