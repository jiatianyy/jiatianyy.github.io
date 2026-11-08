# Content and publication rules

## Personal and research facts

Use supplied or primary-source verified information. Never infer a person's credentials from a name match. Keep unprovided information as “to be added.” Verify paper authorship, title, author order, venue, year, and DOI against the original paper or publisher before listing it as an authored publication. Reading citations belong in notes, not the publication record.

## Add a note

Create `src/content/notes/descriptive-slug.mdx` (or `.md` for content that does not need JSX):

```yaml
---
title: "Title"
description: "One sentence summary"
date: 2026-11-08
draft: true
approved: false
kind: research
references: []
---
```

Use `##` and `###` headings for the generated article TOC. Use `$...$` for inline math, `$$` on separate lines for display equations, and fenced code with a language name for Shiki highlighting.

References have `id`, `title`, `url`, and optional `bibtex`. Cite one using a Markdown link such as `[1](#ref-paper-id)`; the layout emits matching anchors and expandable BibTeX. Use primary sources and verify metadata. Keep claims, hypotheses, reproduction results, and outside results distinct. Label preliminary evidence and include enough methods to interpret it.

## Approval

1. Preview locally and check text, math, references, heading links and mobile layout.
2. Have the author review the exact research content and personal information.
3. Only with author authorization, set `draft: false` and `approved: true` for a research entry. Demo entries remain `kind: demo` and are always excluded from production.
4. Run validation and inspect the production preview. Submit changes for review on a feature branch, then merge only when authorized.

Approval flags are editorial metadata, not an authentication mechanism. Repository review and branch protection are the final control. Local draft filtering does not make source files private: pushing draft source to this public repository makes it publicly readable. Remove confidential drafts from the checkout before any authorized push.
