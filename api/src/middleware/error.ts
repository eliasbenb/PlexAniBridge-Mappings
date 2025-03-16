import { Context } from 'hono';
import { Env } from '../types';

export const errorMiddleware = async (
    c: Context<{ Bindings: Env }>,
    next: () => Promise<void>
) => {
    try {
        await next();
    } catch (error) {
        console.error('Unhandled error:', error);
        return c.json(
            {
                error: error instanceof Error ? error.message : 'An unexpected error occurred',
            },
            500
        );
    }
};
