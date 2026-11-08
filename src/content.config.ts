import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { glob } from 'astro/loaders';

const notes = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/notes' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    draft: z.boolean().default(true),
    approved: z.boolean().default(false),
    kind: z.enum(['research', 'demo']).default('research'),
    date: z.coerce.date(),
    lang: z.enum(['en', 'zh-CN']).default('en'),
    sourceUrl: z.url().optional(),
    sourcePageId: z.string().optional(),
    sourceSha256: z.string().optional(),
    series: z.string().optional(),
    references: z.array(z.object({ id: z.string(), title: z.string(), url: z.url(), bibtex: z.string().optional() })).default([]),
  }),
});
export const collections = { notes };
