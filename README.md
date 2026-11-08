# Jiatian Yuan — academic website

Astro + strict TypeScript + MDX + KaTeX. Static output for https://jiatianyy.github.io.

## Local development (Windows PowerShell)

Use Node.js 24+ and Git. From this repository:

```powershell
npm.cmd ci
npm.cmd run dev
```

Open the Local URL printed by Astro (normally http://127.0.0.1:4321). Development mode includes clearly marked drafts and a technical demo. It is for local review only.

```powershell
npm.cmd run validate
npm.cmd run build:review
node scripts/verify.mjs --review
npm.cmd run build
npm.cmd run preview
```

Production builds exclude drafts, unapproved notes, and all demos. `build:review` overwrites `dist` with a review build; always run `build` again before packaging production output. CI does this automatically in an isolated build job.

## Structure

- `src/pages/`: Home, Research, Publications, Research Notes, Projects, About/CV, 404.
- `src/content/notes/`: schema-validated MDX articles.
- `src/data/`: shared identity, research interests, and note visibility rules.
- `src/styles/global.css`: responsive typography and layout.
- `scripts/verify.mjs`: route, local asset/link/fragment, math, code, bibliography and publication-gate checks.
- `docs/CONTENT.md`: verified content and author review procedure.
- `docs/DEVELOPMENT.md`: development and release process.
- `docs/research/`: 24 complete source Pages and 46 original embedded diagrams, with provenance and a migration manifest; see its README for coverage and publication status.

Research source snapshots generate Markdown notes through `npm run sync:research`; `npm run check:research` verifies they have not drifted. Both Markdown and MDX are supported. Migrated research notes remain drafts for website review, while their source files are publicly readable in this repository by the author's upload request.

## Deployment after review

Development takes place on `codex/academic-site-v1`, based on the repository's existing initial main commit. Submit website changes through a pull request and preserve upstream history. No website changes should be merged or deployed until the author approves them.

In GitHub repository **Settings → Pages → Build and deployment**, choose **GitHub Actions**. Review and merge the feature PR into main. `.github/workflows/deploy.yml` builds production output and deploys Pages only for main pushes. Feature pushes and PRs run validation only. This work does not change remote Pages settings.

## Pending author input

Biography, affiliation, education, employment, contact email, CV, authored publications, and project records remain placeholders. The supplied name and research interests are the only personal claims used.

## References for maintainers

- [Astro Markdown configuration](https://docs.astro.build/en/guides/markdown-content/)
- [Astro GitHub Pages guide](https://docs.astro.build/en/guides/deploy/github/)
- The optional demo references [Universal Transformers](https://arxiv.org/abs/1807.03819), as reading material, not an authored publication.
