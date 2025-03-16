import { fromHono, OpenAPIRoute } from 'chanfana'
import { Hono } from 'hono'
import { z } from 'zod'
import { type Context } from 'hono';
import { AnimeMappings, AniMap, Indexes, CacheEntry } from './types';

export type Env = {
	CDN_URL: string;
}

export type AppContext = Context<{ Bindings: Env }>

let cache: CacheEntry | null = null;
const CACHE_TTL = 3600000;
const SCHEMA_VERSION = 'v2';
const CDN_URL = `https://github.com/eliasbenb/PlexAniBridge-Mappings/raw/refs/heads/${SCHEMA_VERSION}/mappings.json`;

function createIndexes(mappings: AnimeMappings): Indexes {
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

async function getMappings(url: string): Promise<{ mappings: AnimeMappings, indexes: Indexes }> {
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

const AniMapSchema = z.object({
	anidb_id: z.number().nullable(),
	anilist_id: z.number().nullable(),
	imdb_id: z.union([z.string(), z.array(z.string()), z.null()]),
	mal_id: z.union([z.number(), z.array(z.number()), z.null()]),
	tmdb_movie_id: z.union([z.number(), z.array(z.number()), z.null()]),
	tmdb_show_id: z.union([z.number(), z.array(z.number()), z.null()]),
	tvdb_id: z.number().nullable(),
	tvdb_mappings: z.record(z.string(), z.string()).nullable(),
}).openapi({
	type: 'object', description: 'Anime mapping data', example: {
		anidb_id: 1,
		anilist_id: 1,
		imdb_id: 'tt123456',
		mal_id: 1,
		tmdb_movie_id: 1,
		tmdb_show_id: 1,
		tvdb_id: 1,
		tvdb_mappings: {
			's1': '1',
		},
	}
});

const InternalServerErrorSchema = z.object({
	error: z.string(),
}).openapi({
	description: 'Internal server error',
});
const NotFoundErrorSchema = z.object({
	error: z.string(),
}).openapi({
	description: 'Resource not found',
});
const BadRequestErrorSchema = z.object({
	error: z.string(),
}).openapi({
	description: 'Bad request error',
});

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
						schema: InternalServerErrorSchema,
					}
				}
			},
		},
	}

	async handle(c: AppContext) {
		try {
			const url = c.env.CDN_URL || CDN_URL;
			const { mappings } = await getMappings(url);

			return c.json({
				count: Object.keys(mappings).length,
				mappings,
			});
		} catch (error) {
			return c.json({
				error: error instanceof Error ? error.message : 'Unknown error occurred',
			}, 500);
		}
	}
}

export class GetMappingById extends OpenAPIRoute {
	schema = {
		request: {
			params: z.object({
				id: z.string(),
			}),
		},
		responses: {
			'200': {
				description: 'Mapping details',
				content: {
					"application/json": {
						schema: AniMapSchema,
					}
				}
			},
			'404': {
				description: 'Mapping not found',
				content: {
					"application/json": {
						schema: NotFoundErrorSchema,
					}
				}
			},
			'500': {
				description: 'Internal server error',
				content: {
					"application/json": {
						schema: InternalServerErrorSchema,
					}
				}
			},
		},
	}

	async handle(c: AppContext) {
		try {
			const url = c.env.CDN_URL || CDN_URL;
			const { mappings } = await getMappings(url);

			const data = await this.getValidatedData<typeof this.schema>();
			const id = data.params.id;

			if (!mappings[id]) {
				return c.json({
					error: `Mapping with ID ${id} not found`,
				}, 404);
			}

			return c.json(mappings[id]);
		} catch (error) {
			return c.json({
				error: error instanceof Error ? error.message : 'Unknown error occurred',
			}, 500);
		}
	}
}

export class SearchMappings extends OpenAPIRoute {
	schema = {
		request: {
			query: z.object({
				anidb_id: z.string().optional(),
				anilist_id: z.string().optional(),
				imdb_id: z.string().optional(),
				mal_id: z.string().optional(),
				tmdb_movie_id: z.string().optional(),
				tmdb_show_id: z.string().optional(),
				tvdb_id: z.string().optional(),
			}),
		},
		responses: {
			'200': {
				description: 'Search results',
				content: {
					"application/json": {
						schema: z.object({
							count: z.number().openapi({ description: 'Number of results' }),
							results: z.array(z.object({
								id: z.string(),
								mapping: AniMapSchema,
							})),
						})
					}
				}
			},
			'400': {
				description: 'Bad request error',
				content: {
					"application/json": {
						schema: BadRequestErrorSchema,
					}
				}
			},
			'500': {
				description: 'Internal server error',
				content: {
					"application/json": {
						schema: InternalServerErrorSchema,
					}
				}
			},
		},
	}

	async handle(c: AppContext) {
		try {
			const url = c.env.CDN_URL || CDN_URL;
			const { mappings, indexes } = await getMappings(url);

			const data = await this.getValidatedData<typeof this.schema>();
			const query = data.query;

			if (!Object.keys(query).length) {
				return c.json({
					error: 'At least one search parameter is required',
				}, 400);
			}

			const resultIds = new Set<string>();
			if (query.anidb_id && indexes.anidb_id[query.anidb_id]) {
				indexes.anidb_id[query.anidb_id].forEach(id => resultIds.add(id));
			}
			if (query.anilist_id && indexes.anilist_id[query.anilist_id]) {
				indexes.anilist_id[query.anilist_id].forEach(id => resultIds.add(id));
			}
			if (query.imdb_id && indexes.imdb_id[query.imdb_id]) {
				indexes.imdb_id[query.imdb_id].forEach(id => resultIds.add(id));
			}
			if (query.mal_id && indexes.mal_id[query.mal_id]) {
				indexes.mal_id[query.mal_id].forEach(id => resultIds.add(id));
			}
			if (query.tmdb_movie_id && indexes.tmdb_movie_id[query.tmdb_movie_id]) {
				indexes.tmdb_movie_id[query.tmdb_movie_id].forEach(id => resultIds.add(id));
			}
			if (query.tmdb_show_id && indexes.tmdb_show_id[query.tmdb_show_id]) {
				indexes.tmdb_show_id[query.tmdb_show_id].forEach(id => resultIds.add(id));
			}
			if (query.tvdb_id && indexes.tvdb_id[query.tvdb_id]) {
				indexes.tvdb_id[query.tvdb_id].forEach(id => resultIds.add(id));
			}

			const results = Array.from(resultIds).map(id => mappings[id]);

			return c.json({
				count: results.length,
				results,
			});
		} catch (error) {
			return c.json({
				error: error instanceof Error ? error.message : 'Unknown error occurred',
			}, 500);
		}
	}
}

const conditionSchema = z.object({
	field: z.enum([
		"anidb_id",
		"anilist_id",
		"imdb_id",
		"mal_id",
		"tmdb_movie_id",
		"tmdb_show_id",
		"tvdb_id",
	]),
	op: z.enum(["eq", "neq", "in", "nin", "gt", "gte", "lt", "lte"]),
	value: z.union([z.string(), z.number(), z.array(z.union([z.string(), z.number()]))]),
});
type Condition = z.infer<typeof conditionSchema>;

type LogicalFilter = { and: Filter[] } | { or: Filter[] };
type Filter = Condition | LogicalFilter;

const filterSchema: z.ZodType<Filter> = z.lazy(() =>
	z.union([
		conditionSchema,
		z.object({ and: z.array(filterSchema) }),
		z.object({ or: z.array(filterSchema) }),
	])
)

function evaluateCondition(mapping: AniMap, condition: Condition): boolean {
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

function evaluateFilter(mapping: AniMap, filter: Filter): boolean {
	if ("and" in filter) {
		return filter.and.every((f) => evaluateFilter(mapping, f));
	} else if ("or" in filter) {
		return filter.or.some((f) => evaluateFilter(mapping, f));
	} else {
		return evaluateCondition(mapping, filter);
	}
}

function getInitialCandidates(condition: Condition, indexes: Indexes): Set<string> | null {
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

export class AdvancedSearchMappings extends OpenAPIRoute {
	schema = {
		request: {
			body: {
				description: 'Search request body',
				required: true,
				content: {
					'application/json': {
						schema: z.object({
							filters: filterSchema.openapi({
								description: 'Filter conditions', type: 'object', example: {
									and: [
										{ field: 'anilist_id', op: 'eq', value: 1 },
										{ field: 'mal_id', op: 'eq', value: 1 },
									]
								}
							}),
						})
					},
				},
			},
		},
		responses: {
			'200': {
				description: 'Search results',
				content: {
					"application/json": {
						schema: z.object({
							count: z.number().openapi({ description: 'Number of results' }),
							results: z.array(z.object({
								id: z.string(),
								mapping: AniMapSchema,
							})),
						})
					}
				}
			},
			'400': {
				description: 'Bad request error',
				content: {
					"application/json": {
						schema: BadRequestErrorSchema,
					}
				}
			},
			'500': {
				description: 'Internal server error',
				content: {
					"application/json": {
						schema: InternalServerErrorSchema,
					}
				}
			},
		},
	};

	async handle(c: AppContext) {
		try {
			const url = c.env.CDN_URL || CDN_URL;
			const { mappings, indexes } = await getMappings(url);

			const data = await this.getValidatedData<typeof this.schema>();
			const filters = data.body.filters;

			let initialCandidates: Set<string> | null = null;

			if (!("and" in filters) && !("or" in filters) && filters.op === "eq") {
				initialCandidates = getInitialCandidates(filters, indexes);
			}

			let results: Array<AniMap>;

			if (initialCandidates !== null && initialCandidates.size === 0) {
				results = [];
			} else if (initialCandidates !== null) {
				results = Array.from(initialCandidates)
					.filter(id => {
						try {
							return evaluateFilter(mappings[id], filters);
						} catch (e) {
							return false;
						}
					})
					.map(id => mappings[id]);
			} else {
				results = Object.entries(mappings)
					.filter(([_, mapping]) => {
						try {
							return evaluateFilter(mapping, filters);
						} catch (e) {
							return false;
						}
					})
					.map(([_, mapping]) => mapping);
			}

			return c.json({
				count: results.length,
				results,
			});
		} catch (error) {
			return c.json(
				{
					error: error instanceof Error ? error.message : 'Unknown error occurred',
				},
				500
			);
		}
	}
}


const app = new Hono<{ Bindings: Env }>();
const openapi = fromHono(app, {});

openapi.get('/all', GetAllMappings);
openapi.get('/find/:id', GetMappingById);
openapi.get('/search', SearchMappings);
openapi.post('/search', AdvancedSearchMappings);

export default app;