# Project instructions

This is Jiatian Yuan's static academic website: Astro + TypeScript + MDX + KaTeX.

- Work on a `codex/` feature branch. Do not merge to main, push, create a PR, or deploy without the user's authorization for that step.
- Never invent personal history, affiliation, publications, awards, email, project results, or metrics. Missing information stays explicitly pending.
- Keep `src/data/site.ts` as the shared profile and research-interest source.
- Notes belong in `src/content/notes/`. Follow `docs/CONTENT.md`. Default to `draft: true`, `approved: false`.
- Only the author can approve research claims. Do not change approval flags on the author's behalf unless specifically instructed.
- Demo entries always use `kind: demo`; they must never be emitted by a production build.
- Use static HTML and CSS where possible. Keep accessibility, keyboard focus, readable math, and mobile layouts.
- Run `npm run validate`, then `npm run build:review` and `node scripts/verify.mjs --review` for note, math, or layout changes. Browser-check desktop and narrow widths when browser access is available.
- Commit `package-lock.json`. Never commit dependencies, `dist/`, secrets, or local review screenshots.
- GitHub Pages deployment is triggered only by a push to main after review. Do not enable deployment from a development branch.
