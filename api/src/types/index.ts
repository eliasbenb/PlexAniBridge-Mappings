import { z } from 'zod';

import { Context } from 'hono';

export type Env = Record<string, never>;

export type AppContext = Context<{ Bindings: Env }>;

export interface AnimeMappings {
    [id: string]: AniMap;
}

export interface AniMap {
    anidb_id: number | null;
    anilist_id: number | null;
    imdb_id: string | string[] | null;
    mal_id: number | number[] | null;
    tmdb_movie_id: number | number[] | null;
    tmdb_show_id: number | null;
    tvdb_id: number | null;
    tvdb_mappings: Record<string, string> | null;
}

export interface Indexes {
    anidb_id: Record<string, string[]>;
    anilist_id: Record<string, string[]>;
    imdb_id: Record<string, string[]>;
    mal_id: Record<string, string[]>;
    tmdb_movie_id: Record<string, string[]>;
    tmdb_show_id: Record<string, string[]>;
    tvdb_id: Record<string, string[]>;
}


export const AniMapSchema = z.object({
    anidb_id: z.number().nullable().openapi({
        type: 'integer',
        description: 'The matching AniDB ID for the entry',
        example: 12820
    }),
    anilist_id: z.number().int().nullable(),
    imdb_id: z.union([z.string(), z.array(z.string()), z.null()]),
    mal_id: z.union([z.number().int(), z.array(z.number().int()), z.null()]),
    tmdb_movie_id: z.union([z.number().int(), z.array(z.number().int()), z.null()]),
    tmdb_show_id: z.number().int().nullable(),
    tvdb_id: z.number().int().nullable(),
    tmdb_mappings: z.record(
        z.string(),
        z.string()
    ).nullable(),
    tvdb_mappings: z.record(
        z.string(),
        z.string()
    ).nullable()
}).openapi({
    title: 'Anime Mapping',
    description: 'Anime mapping data that connects IDs across different anime databases',
    properties: {
        anidb_id: {
            type: 'integer',
            nullable: true,
            description: 'The matching AniDB ID for the entry',
            example: 12820
        },
        anilist_id: {
            type: 'integer',
            nullable: true,
            description: 'The AniList ID of the anime',
            example: 98310
        },
        imdb_id: {
            oneOf: [
                { type: 'string' },
                { type: 'array', items: { type: 'string' } },
                { type: 'null' }
            ],
            description: 'The matching IMDb ID(s) for the entry. Can be a single ID or an array of IDs',
            example: 'tt9288776'
        },
        mal_id: {
            oneOf: [
                { type: 'integer' },
                { type: 'array', items: { type: 'integer' } },
                { type: 'null' }
            ],
            description: 'The matching MyAnimeList ID(s) for the entry. Can be a single ID or an array of IDs',
            example: [34915, 36186, 36529]
        },
        tmdb_movie_id: {
            oneOf: [
                { type: 'integer' },
                { type: 'array', items: { type: 'integer' } },
                { type: 'null' }
            ],
            description: 'The matching TMDB ID(s) for the movie entry. Can be a single ID or an array of IDs',
            example: 532067
        },
        tmdb_show_id: {
            type: 'integer',
            nullable: true,
            description: 'The matching TMDB ID for the show entry',
            example: 83095
        },
        tvdb_id: {
            type: 'integer',
            nullable: true,
            description: 'The matching TVDB ID for the show entry',
            example: 333234
        },
        tmdb_mappings: {
            type: 'object',
            nullable: true,
            additionalProperties: { type: 'string' },
            description: 'A dictionary mapping TMDB seasons to episode patterns',
            example: {
                's0': 'e1',
                's1': 'e1-e13'
            }
        },
        tvdb_mappings: {
            type: 'object',
            nullable: true,
            additionalProperties: { type: 'string' },
            description: 'A dictionary mapping TVDB seasons to episode patterns',
            example: {
                's0': 'e1',
                's1': 'e1-e13'
            }
        }
    },
    example: {
        anidb_id: 12820,
        anilist_id: 98310,
        imdb_id: 'tt9288776',
        mal_id: [34915, 36186, 36529],
        tmdb_movie_id: null,
        tmdb_show_id: 83095,
        tvdb_id: 333234,
        tmdb_mappings: {
            's0': 'e1',
            's1': 'e1-e13'
        },
        tvdb_mappings: {
            's0': 'e1',
            's1': 'e1-e13'
        }
    }
});



export const ErrorSchemas = {
    InternalServerError: z.object({
        error: z.string(),
    }).openapi({
        description: 'Internal server error',
    }),

    NotFoundError: z.object({
        error: z.string(),
    }).openapi({
        description: 'Resource not found',
    }),

    BadRequestError: z.object({
        error: z.string(),
    }).openapi({
        description: 'Bad request error',
    })
};
