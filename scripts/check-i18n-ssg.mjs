import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const distDir = new URL('../dist/', import.meta.url);
const targetLocales = ['en', 'ja', 'fr', 'de'];
const zhBlogDir = join(distDir.pathname, 'blog');

const articleSlugs = readdirSync(zhBlogDir, { withFileTypes: true })
  .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.'))
  .map((entry) => entry.name);

if (articleSlugs.length === 0) {
  throw new Error('No localized Chinese blog pages found in dist/blog');
}

function readDist(pathname) {
  return readFileSync(join(distDir.pathname, pathname), 'utf8');
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

for (const slug of articleSlugs) {
  const zhArticle = readDist(`blog/${slug}/index.html`);
  assert(
    !zhArticle.includes('translate.google.com'),
    'Language switcher must not link to external Google Translate URLs.',
  );
  assert(!zhArticle.includes('href="/ko/'), 'Korean routes should not appear in the language menu.');
  assert(!zhArticle.includes('href="/es/'), 'Spanish routes should not appear in the language menu.');
  assert(!zhArticle.includes('한국어'), 'Korean label should not appear in the language menu.');
  assert(!zhArticle.includes('Español'), 'Spanish label should not appear in the language menu.');
  for (const locale of targetLocales) {
    assert(
      zhArticle.includes(`href="/${locale}/blog/${slug}"`),
      `Chinese article page should link to the local ${locale} route for ${slug}.`,
    );
  }
}

for (const locale of targetLocales) {
  for (const slug of articleSlugs) {
    const articlePath = join(distDir.pathname, `${locale}/blog/${slug}/index.html`);
    assert(
      existsSync(articlePath),
      `${locale} SSG article page should be generated for ${slug}: dist/${locale}/blog/${slug}/index.html`,
    );

    const articleHtml = readFileSync(articlePath, 'utf8');
    assert(
      !articleHtml.includes('translate.google.com'),
      `${locale} article ${slug} should not link to external Google Translate URLs.`,
    );
  }
}

for (const slug of articleSlugs) {
  const enArticle = readDist(`en/blog/${slug}/index.html`);
  assert(
    enArticle.includes('lang="en"') || enArticle.includes('lang="en-US"'),
    `English SSG article ${slug} should render with English html lang.`,
  );
}

console.log('i18n SSG checks passed');
