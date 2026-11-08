# Development conventions

- Use a `codex/` feature branch and focused commits. Preserve remote history and user changes.
- Prefer typed shared data and small Astro components. Avoid client JavaScript unless the interaction needs it.
- Keep titles, descriptions, canonical URLs, navigation landmarks, heading order and visible focus consistent.
- Bundle math fonts locally through KaTeX. No analytics, external fonts, or tracking scripts are required.
- Production command: `npm run validate`. Review command: `npm run build:review`, then `node scripts/verify.mjs --review`.
- Check desktop and 390px/360px widths for wrapping, horizontal overflow, navigation, formulas and code. Verify the keyboard path and reduced motion behavior.
- Check external links when adding content. The automated verifier checks local links, assets and fragments; it deliberately does not depend on third-party network availability.
- Both workflows install the committed lockfile with `npm ci`. Feature CI validates production and review output; Pages uploads only the validated production output.
- For releases, verify GitHub Pages is configured for Actions. Use the `github-pages` environment and configure reviewer protection there if desired. Workflow configuration alone does not enable repository Pages settings.
- Dependency major upgrades require checking Astro, MDX and remark/rehype compatibility and repeating formula verification.
