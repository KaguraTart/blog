import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { findMissingTranslations } from './i18n_coverage.mjs';

const root = mkdtempSync(join(tmpdir(), 'blog-i18n-coverage-'));
const blogDir = join(root, 'src/content/blog');
mkdirSync(blogDir, { recursive: true });
const sourceText = '---\ntitle: 新文章\ndescription: test\npubDate: 2026-05-25\ntags: []\ncategory: Tech\n---\n\n正文\n';
writeFileSync(join(blogDir, 'new-paper.md'), sourceText);

const sourceHash = createHash('sha1').update(sourceText, 'utf8').digest('hex');

for (const locale of ['en', 'ja', 'fr']) {
  mkdirSync(join(blogDir, locale), { recursive: true });
  writeFileSync(
    join(blogDir, locale, 'new-paper.md'),
    `---\ntitle: translated\ndescription: test\npubDate: 2026-05-25\ntags: []\ncategory: Tech\nsourceHash: "${sourceHash}"\n---\n\nbody\n`,
  );
}
mkdirSync(join(blogDir, 'de'), { recursive: true });
writeFileSync(join(blogDir, 'de', 'new-paper.md'), `---\ntitle: translated\ndescription: test\npubDate: 2026-05-25\ntags: []\nsourceHash: \"wronghash\"\n---\n\nbody\n`);

writeFileSync(join(blogDir, 'en', 'new-paper-unchanged.md'), `---\ntitle: translated\ndescription: test\npubDate: 2026-05-25\ntags: []\nsourceHash: \"not-needed\"\n---\n\nbody\n`);

const missing = findMissingTranslations({ rootDir: root, locales: ['en', 'ja', 'fr', 'de'] });

assert.deepEqual(missing, [
  { source: 'new-paper.md', locale: 'de', expected: 'src/content/blog/de/new-paper.md', status: 'stale', expectedHash: sourceHash, actualHash: 'wronghash' },
]);

console.log('i18n coverage unit test passed');
