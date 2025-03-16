import { OpenAPIRoute } from 'chanfana';
import { z } from 'zod';
import { AppContext } from '../types';
import { getMappings } from '../utils/cache';
import { DEFAULT_CDN_URL } from '../config/constants';
import { AniMapSchema, ErrorSchemas } from '../types';

export class GetAllMappings extends OpenAPIRoute {
    schema = {
        responses: {
            '200': {
                description: 'List of all mappings',
                content: {
                    "application/json": {
                        schema: z.object({
                            count: z.number(),
                            mappings: AniMapSchema,
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

            return c.json({
                count: Object.keys(mappings).length,
                mappings: Object.values(mappings),
            });
        } catch (error) {
            return c.json({
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            }, 500);
        }
    }
}
