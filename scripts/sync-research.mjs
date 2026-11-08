import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';

const manifest = JSON.parse(await readFile('docs/research/manifest.json', 'utf8'));
const check = process.argv.includes('--check');
const byId = new Map(manifest.pages.map(p => [p.pageId, p]));
const byReference = new Map(manifest.embeds.map(e => [e.reference, e]));
const hash = text => createHash('sha256').update(text).digest('hex');
const html = text => text.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
function rewriteLinks(text) {
  return text.replace(/https:\/\/chatgpt\.com\/space\/(page_[a-zA-Z0-9]+)(#[^\s)"<>]+)?/g, (original, id, fragment) => {
    const page = byId.get(id);
    if (!page || (fragment && !page.blocks.some(b => b.id === fragment.slice(1)))) return original;
    return `/notes/loop-reasoning/${page.slug}/${fragment ? '#source-block-' + fragment.slice(1) : ''}`;
  });
}
function adaptBlock(markdown, page, blockId) {
  let fence = false;
  return rewriteLinks(markdown).split('\n').map(line => {
    if (/^```/.test(line)) { fence = !fence; return line === '```codex-prompt' ? '```text' : line; }
    if (fence) return line;
    return line.replace(/^(#{1,5}) /, '$1# ').replace(/\[([^\]]+)\]\((?:library-file:|project-file:)[^)\s]+\)/g, (_, label) => `[${label}（原页附件）](${page.url}#${blockId})`).replace(/!\[([^\]]*)\]\((visualize:[^)\s]+)\)/g, (_, title, reference) => {
      const asset = byReference.get(reference);
      assert(asset, `Missing embedded source: ${reference}`);
      const url = `/research-assets/${asset.sha256}.html`;
      return `<figure class="research-embed"><iframe src="${url}" title="${html(title)}" loading="lazy" sandbox=""></iframe><figcaption>${html(title)} · <a href="${url}" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>`;
    });
  }).join('\n');
}
await mkdir('src/content/notes/loop-reasoning', { recursive: true });
let formulas = 0;
for (const page of manifest.pages) {
  const raw = await readFile(page.sourcePath, 'utf8');
  assert(raw.length === page.characters || raw === raw.slice(0, page.characters) + '\n', `Source length mismatch: ${page.slug}`);
  const blocks = page.blocks.map(b => raw.slice(b.start, b.end));
  assert.equal(blocks.join('\n\n'), raw.slice(0, page.characters), `Incomplete source: ${page.slug}`);
  formulas += (raw.match(/```math/g) || []).length;
  const body = page.blocks.map((b, i) => `<span id="source-block-${b.id}"></span>\n\n${adaptBlock(blocks[i], page, b.id)}`).join('\n\n');
  const description = blocks.find(b => !b.startsWith('#') && !b.startsWith('[') && !b.startsWith('>'))?.replace(/\n/g, ' ').slice(0, 180) || page.title;
  const frontmatter = ['---', `title: ${JSON.stringify(page.title)}`, `description: ${JSON.stringify(description)}`, 'draft: true', 'approved: false', 'kind: research', 'date: 2026-11-08', 'lang: zh-CN', `sourceUrl: ${JSON.stringify(page.url)}`, `sourcePageId: ${JSON.stringify(page.pageId)}`, `sourceSha256: ${hash(raw.slice(0, page.characters))}`, 'series: loop-reasoning', 'references: []', '---'].join('\n');
  const content = `${frontmatter}\n\n${body}\n`;
  const target = `src/content/notes/loop-reasoning/${page.slug}.md`;
  if (check) assert.equal(await readFile(target, 'utf8'), content, `Run npm run sync:research: ${target}`);
  else await writeFile(target, content, 'utf8');
}
for (const asset of manifest.embeds) {
  let raw = await readFile(asset.sourcePath, 'utf8');
  if (hash(raw) !== asset.sha256 && hash(raw.slice(0, -1)) === asset.sha256) {
    assert(!check, `Source asset needs newline restoration: ${asset.title}`);
    raw = raw.slice(0, -1);
    await writeFile(asset.sourcePath, raw, 'utf8');
  }
  assert.equal(hash(raw), asset.sha256, `Source asset digest mismatch: ${asset.title}`);
}
console.log(`${check ? 'Checked' : 'Generated'} ${manifest.pages.length} complete research notes, ${formulas} formula blocks and ${manifest.embeds.length} source-verified embedded figures.`);
