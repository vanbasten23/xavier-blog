# Xavier Wei's Technical Blogs

Website: https://vanbasten23.github.io/xavier-blog/

## Writing a post

Add a Markdown file to `src/pages/blog/`. It shows up on the homepage automatically. Start it with:

```yaml
---
layout: ../../layouts/Layout.astro
title: "Post title"
description: "One-line summary"
pubDate: 2026-10-31
---
```

Put images in `public/images/` and reference them as `/images/<name>.png`.

## Running locally

```sh
npm install
npm run dev      # http://localhost:4321/xavier-blog/
npm run build    # outputs to ./dist/
```

Pushing to `main` deploys to GitHub Pages.
