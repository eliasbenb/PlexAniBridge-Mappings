import { fromHono } from 'chanfana';
import { Hono } from 'hono';
import { Context } from 'hono';
import { errorMiddleware } from './middleware/error';
import { GetAllMappings, SearchMappings } from './routes';

export type Env = {
	CDN_URL: string;
}

export type AppContext = Context<{ Bindings: Env }>;

const app = new Hono<{ Bindings: Env }>();
const openapi = fromHono(app, {});

app.use('*', errorMiddleware);

openapi.get('/all', GetAllMappings);
openapi.get('/search', SearchMappings);

export default app;
