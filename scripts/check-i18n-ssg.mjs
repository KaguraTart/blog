import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const distDir = new URL('../dist/', import.meta.url);
const articleSlug = 'paper-g1-cloudbrain-agent-full-paper-plan-v1-20260520';
const targetLocales = ['en', 'ja', 'fr', 'de'];

function readDist(pathname) {
  return readFileSync(join(distDir.pathname, pathname), 'utf8');
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

const zhArticle = readDist(`blog/${articleSlug}/index.html`);
assert(
  !zhArticle.includes('translate.google.com'),
  'Language switcher must not link to external Google Translate URLs.',
);
for (const locale of targetLocales) {
  assert(
    zhArticle.includes(`href="/${locale}/blog/${articleSlug}"`),
    `Chinese article page should link to the local ${locale} SSG article route.`,
  );
}
assert(!zhArticle.includes('href="/ko/'), 'Korean routes should not appear in the language menu.');
assert(!zhArticle.includes('href="/es/'), 'Spanish routes should not appear in the language menu.');
assert(!zhArticle.includes('한국어'), 'Korean label should not appear in the language menu.');
assert(!zhArticle.includes('Español'), 'Spanish label should not appear in the language menu.');

for (const locale of targetLocales) {
  const articlePath = join(distDir.pathname, `${locale}/blog/${articleSlug}/index.html`);
  assert(
    existsSync(articlePath),
    `${locale} SSG article page should be generated under dist/${locale}/blog/...`,
  );

  const articleHtml = readFileSync(articlePath, 'utf8');
  assert(
    !articleHtml.includes('translate.google.com'),
    `${locale} article should not link to external Google Translate URLs.`,
  );
  assert(
    !articleHtml.includes('完整论文方案 v1：面向低空交通云脑的可验证 LLM Agent'),
    `${locale} article should not render the untranslated Chinese title.`,
  );
}

const enArticle = readDist(`en/blog/${articleSlug}/index.html`);
assert(
  enArticle.includes('Complete Paper Plan') || enArticle.includes('Verifiable LLM Agent'),
  'English SSG article should contain translated English article text.',
);

console.log('i18n SSG checks passed');
