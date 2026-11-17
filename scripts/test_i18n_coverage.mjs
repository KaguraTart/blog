import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { findMissingTranslations } from './i18n_coverage.mjs';

const root = mkdtempSync(join(tmpdir(), 'blog-i18n-coverage-'));
const blogDir = join(root, 'src/content/blog');

const freshSource = '---\ntitle: 新文章\ndescription: test\npubDate: 2026-05-25\ntags: []\ncategory: Tech\n---\n\n正文\n';
mkdirSync(join(blogDir, '2026/05'), { recursive: true });
writeFileSync(join(blogDir, '2026/05/new-paper.md'), freshSource);
mkdirSync(join(blogDir, 'en/2026/05'), { recursive: true });
writeFileSync(
  join(blogDir, 'en/2026/05/new-paper.md'),
  `---\ntitle: translated\ndescription: test\npubDate: 2026-05-25\ntags: []\ncategory: Tech\nsourceHash: "${createHash('sha1').update(freshSource, 'utf8').digest('hex')}"\n---\n\nbody\n`,
);

const staleSource = '---\ntitle: 旧文章\ndescription: test\npubDate: 2026-06-01\ntags: []\ncategory: Tech\n---\n\n旧正文\n';
mkdirSync(join(blogDir, '2026/06'), { recursive: true });
writeFileSync(join(blogDir, '2026/06/stale-paper.md'), staleSource);
mkdirSync(join(blogDir, 'en/2026/06'), { recursive: true });
writeFileSync(
  join(blogDir, 'en/2026/06/stale-paper.md'),
  '---\ntitle: translated\ndescription: test\npubDate: 2026-06-01\ntags: []\ncategory: Tech\nsourceHash: "wronghash"\n---\n\nbody\n',
);

const missing = findMissingTranslations({ rootDir: root });

assert.deepEqual(missing, [
  {
    source: '2026/06/stale-paper.md',
    locale: 'en',
    expected: 'src/content/blog/en/2026/06/stale-paper.md',
    status: 'stale',
    expectedHash: createHash('sha1').update(staleSource, 'utf8').digest('hex'),
    actualHash: 'wronghash',
  },
]);

console.log('i18n coverage unit test passed');
