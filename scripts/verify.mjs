import { readdir, readFile, access } from 'node:fs/promises';
import { resolve, relative } from 'node:path';
import assert from 'node:assert/strict';
import { load } from 'cheerio';

const root = resolve('dist');
const review = process.argv.includes('--review');
const research = JSON.parse(await readFile('docs/research/manifest.json', 'utf8'));
async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  return (await Promise.all(entries.map(e => e.isDirectory() ? walk(resolve(dir, e.name)) : resolve(dir, e.name)))).flat();
}
const files = await walk(root);
const pages = new Map();
for (const file of files.filter(f => f.endsWith('.html'))) pages.set(file, load(await readFile(file, 'utf8')));
for (const route of ['index.html', 'research/index.html', 'publications/index.html', 'notes/index.html', 'projects/index.html', 'about/index.html', '404.html']) {
  assert(pages.has(resolve(root, route)), `Missing route: ${route}`);
}
let links = 0;
for (const [file, $] of pages) {
  if (relative(root, file).replaceAll('\\', '/').startsWith('research-assets/')) {
    assert.equal($('script').length, 0, `Embedded source scripts must not execute: ${file}`);
    assert($('meta[http-equiv="Content-Security-Policy"]').length, `Missing embedded content policy: ${file}`);
    continue;
  }
  assert.equal($('h1').length, 1, `Expected one h1: ${file}`);
  assert.equal($('main').length, 1, `Expected main landmark: ${file}`);
  assert($('meta[name="viewport"]').length, `Missing viewport: ${file}`);
  assert($('title').text().includes('Jiatian Yuan'), `Missing title: ${file}`);
  assert.equal($('.katex-error').length, 0, `Invalid formula: ${file}`);
  for (const el of $('[href], [src]').toArray()) {
    const href = $(el).attr('href') ?? $(el).attr('src');
    if (!href || /^(https?:|mailto:|data:)/.test(href)) continue;
    const url = new URL(href, `https://jiatianyy.github.io/${relative(root, file).replaceAll('\\', '/')}`);
    let target = resolve(root, '.' + decodeURIComponent(url.pathname));
    if (url.pathname.endsWith('/')) target = resolve(target, 'index.html');
    else if (!/\.[a-z0-9]+$/i.test(url.pathname)) target = resolve(target, 'index.html');
    await access(target).catch(() => { throw new Error(`Broken link ${href} in ${file}`); });
    if (url.hash && pages.has(target)) assert(pages.get(target)(`[id]`).toArray().some(e => pages.get(target)(e).attr('id') === decodeURIComponent(url.hash.slice(1))), `Missing anchor ${href} in ${file}`);
    links++;
  }
}
const demo = resolve(root, 'notes/iterative-computation-demo/index.html');
if (review) {
  let formulaCount = 0;
  for (const note of research.pages) {
    const $note = pages.get(resolve(root, `notes/loop-reasoning/${note.slug}/index.html`));
    assert($note, `Missing migrated note: ${note.slug}`);
    assert.equal($note('html').attr('lang'), 'zh-CN');
    assert.equal($note('[id^="source-block-"]').length, note.blocks.length, `Missing source blocks: ${note.slug}`);
    assert($note(`a[href="${note.url}"]`).length, `Missing provenance: ${note.slug}`);
    assert($note('.review-banner').length, `Missing draft label: ${note.slug}`);
    formulaCount += $note('.katex-display').length;
    for (const frame of $note('iframe').toArray()) assert.equal($note(frame).attr('sandbox'), '', 'Embedded diagrams must be sandboxed');
  }
  for (const asset of research.embeds) assert(pages.has(resolve(root, `research-assets/${asset.sha256}.html`)), `Missing embedded diagram: ${asset.title}`);
  assert(formulaCount >= 486, `Expected migrated equation blocks, found ${formulaCount}`);
  const $ = pages.get(demo);
  assert($, 'Review build must contain demonstration note');
  assert($('.katex').length >= 3, 'Expected rendered inline and display math');
  assert($('.katex-display math').length, 'Expected accessible display math');
  assert($('pre.astro-code span[style]').length, 'Expected highlighted code tokens');
  assert($('.toc a[href="#a-shared-update"]').length, 'Expected generated heading in TOC');
  assert($('#ref-universal-transformers').length, 'Expected citation target');
  assert($('details code').text().includes('@misc'), 'Expected BibTeX');
  assert($('meta[name="robots"]').attr('content').includes('noindex'), 'Review build must discourage indexing');
} else {
  for (const file of pages.keys()) assert(!relative(root, file).replaceAll('\\', '/').startsWith('notes/loop-reasoning/'), 'Research drafts leaked to production');
  assert(!files.some(file => relative(root, file).replaceAll('\\', '/').startsWith('research-assets/')), 'Draft diagrams leaked to production');
  assert(!pages.has(demo), 'Demo must never appear in production');
  for (const [, $] of pages) assert.equal($('.review-banner').length, 0, 'Unreviewed entry leaked to production');
}
console.log(`Verified ${pages.size} pages, ${links} local links/assets/anchors, and ${review ? 'math, code, references, TOC' : 'production draft exclusion'}.`);
