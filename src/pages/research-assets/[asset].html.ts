import type { APIRoute } from 'astro';
import manifest from '../../../docs/research/manifest.json';
const sources = import.meta.glob('../../../docs/research/embeds/*.html', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;

export function getStaticPaths() {
  const review = import.meta.env.DEV || import.meta.env.MODE === 'review';
  return review ? manifest.embeds.map(asset => ({ params: { asset: asset.sha256 }, props: { asset } })) : [];
}
export const GET: APIRoute = ({ props }) => {
  const asset = props.asset as (typeof manifest.embeds)[number];
  const raw = sources[`../../../${asset.sourcePath}`];
  const staticContent = raw.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, '');
  const title = asset.title.replaceAll('&', '&amp;').replaceAll('<', '&lt;');
  const page = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex,nofollow"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:; base-uri 'none'; form-action 'none'"><title>${title} — Jiatian Yuan</title><style>:root{--foreground:#242e29;--muted-foreground:#667069;--border:#dbe0d5;color-scheme:light}body{margin:0;padding:20px;background:#f7f8f3;font-family:'Segoe UI','Microsoft YaHei',sans-serif;overflow-wrap:anywhere}svg{max-width:100%}a{color:#234d3c}h1{font-size:20px}</style></head><body><main><h1>${title}</h1>${staticContent}</main></body></html>`;
  return new Response(page, { headers: { 'Content-Type': 'text/html; charset=utf-8' } });
};
