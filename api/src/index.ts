import { fromHono } from 'chanfana';
import { Hono } from 'hono';
import { errorMiddleware } from './middleware/error';
import { GetAllMappings, SearchMappings } from './routes/v2';
import { Env } from './types';

import pkg from '../package.json';

const version = pkg.version;

const app = new Hono<{ Bindings: Env }>();
const openapi = fromHono(app, {
	schema: {
		info: {
			title: 'PlexAniBridge Mappings API',
			version: version,
			description: 'API to query the PlexAniBridge mappings database',
			license: {
				name: 'MIT',
				url: 'https://opensource.org/licenses/MIT',
			},
		},
	},
	docs_url: '/',
	redoc_url: '/redocs',
});

app.use('*', errorMiddleware);

openapi.get('/api/v2/all', GetAllMappings);
openapi.get('/api/v2/search', SearchMappings);

export default app;
