import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { routeLocales } from '../src/i18n/config.mjs';

export function getSourcePosts(rootDir = process.cwd()) {
  const blogDir = join(rootDir, 'src/content/blog');

  return readdirSync(blogDir, { withFileTypes: true })
    .filter(entry => entry.isFile() && entry.name.endsWith('.md'))
    .map(entry => entry.name)
    .sort();
}

export function findMissingTranslations({ rootDir = process.cwd(), locales = routeLocales } = {}) {
  const blogDir = join(rootDir, 'src/content/blog');
  const sourcePosts = getSourcePosts(rootDir);
  const missing = [];

  for (const source of sourcePosts) {
    for (const locale of locales) {
      const expectedPath = join(blogDir, locale, source);
      if (!existsSync(expectedPath)) {
        missing.push({
          source,
          locale,
          expected: `src/content/blog/${locale}/${source}`,
        });
      }
    }
  }

  return missing;
}
