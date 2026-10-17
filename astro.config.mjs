import { defineConfig } from 'astro/config';

export default defineConfig({
  markdown: {
    shikiConfig: {
      theme: 'rose-pine-dawn',
    },
  },
  site: 'https://vanbasten23.github.io',
  base: '/xavier-blog',
});
