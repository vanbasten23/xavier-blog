import { defineConfig } from 'astro/config';
import { satteri } from '@astrojs/markdown-satteri';

const BASE = '/xavier-blog';

// Prefix a root-relative URL ("/images/foo.png") with the site base, so posts
// can keep writing "/images/..." and still work under BASE on GitHub Pages.
const withBase = (url) =>
  url.startsWith('/') && !url.startsWith('//') && !url.startsWith(BASE + '/')
    ? BASE + url
    : url;

const prefixBasePlugin = {
  name: 'prefix-base',
  // Markdown image syntax: ![alt](/images/foo.png)
  element: {
    filter: ['img'],
    visit(node, ctx) {
      const src = node.properties?.src;
      if (typeof src === 'string' && withBase(src) !== src) {
        ctx.setProperty(node, 'src', withBase(src));
      }
    },
  },
  // Raw HTML in Markdown: <img src="/images/foo.png" />
  raw(node) {
    const value = node.value.replace(
      /(\ssrc\s*=\s*)(["'])(\/[^"']*)\2/g,
      (_, attr, quote, url) => `${attr}${quote}${withBase(url)}${quote}`,
    );
    if (value !== node.value) return { type: 'raw', value };
  },
};

export default defineConfig({
  markdown: {
    shikiConfig: {
      theme: 'rose-pine-dawn',
    },
    processor: satteri({ hastPlugins: [prefixBasePlugin] }),
  },
  site: 'https://vanbasten23.github.io',
  base: BASE,
});
