// @ts-check
import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeMathjax from 'rehype-mathjax';
import { i18n } from './src/i18n/config.mjs';

// https://astro.build/config
export default defineConfig({
  site: 'https://blog.kaguratart.com',
  i18n,
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeMathjax],
    shikiConfig: {
      // 只用一个主题，light。深色用 CSS 覆盖。
      theme: 'github-light',
      wrap: true,
    },
  },
});
