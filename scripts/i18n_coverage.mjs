import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { locales, routeLocales } from '../src/i18n/config.mjs';

const localeDirNames = new Set(locales);

function collectMarkdownFiles(dir, relativeBase = '') {
  const files = [];

  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.')) continue;

    const relativePath = relativeBase ? `${relativeBase}/${entry.name}` : entry.name;
    const fullPath = join(dir, entry.name);

    if (entry.isDirectory()) {
      if (relativeBase === '' && localeDirNames.has(entry.name)) continue;
      files.push(...collectMarkdownFiles(fullPath, relativePath));
    } else if (entry.isFile() && entry.name.endsWith('.md')) {
      files.push({ relativePath, fullPath });
    }
  }

  return files;
}

function getSourcePostEntries(rootDir = process.cwd()) {
  const blogDir = join(rootDir, 'src/content/blog');

  const entries = collectMarkdownFiles(blogDir)
    .map(entry => ({
      name: entry.relativePath,
      hash: createHash('sha1').update(readFileSync(entry.fullPath, 'utf8'), 'utf8').digest('hex'),
    }))
    .sort((a, b) => a.name.localeCompare(b.name));

  return entries;
}

export function getSourcePosts(rootDir = process.cwd()) {
  return getSourcePostEntries(rootDir).map(entry => entry.name);
}

function sha1(text) {
  return createHash('sha1').update(text, 'utf8').digest('hex');
}

function parseYamlString(value) {
  const trimmed = value.trim();
  if (trimmed.length >= 2 && trimmed[0] === trimmed[trimmed.length - 1] && (trimmed[0] === '"' || trimmed[0] === "'")) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function readSourceHash(frontmatter) {
  const match = frontmatter.match(/^\s*sourceHash\s*:\s*(.+)\s*$/m);
  if (!match) {
    return null;
  }
  return parseYamlString(match[1]);
}

export function getSourcePostHashes(rootDir = process.cwd()) {
  const result = new Map();
  for (const entry of getSourcePostEntries(rootDir)) {
    result.set(entry.name, entry.hash);
  }

  return result;
}

function getFrontmatter(content) {
  if (!content.startsWith('---\n')) {
    return null;
  }

  const end = content.indexOf('\n---', 4);
  if (end === -1) {
    return null;
  }

  return content.slice(4, end);
}

function isTranslationFresh(localePath, expectedHash) {
  const content = readFileSync(localePath, 'utf8');
  const frontmatter = getFrontmatter(content);
  if (!frontmatter) {
    return false;
  }
  const currentHash = readSourceHash(frontmatter);
  return currentHash !== null && expectedHash !== null && currentHash === expectedHash;
}

export function findMissingTranslations({ rootDir = process.cwd(), locales = routeLocales } = {}) {
  const blogDir = join(rootDir, 'src/content/blog');
  const sourcePosts = getSourcePosts(rootDir);
  const sourceHashes = getSourcePostHashes(rootDir);
  const missing = [];

  for (const source of sourcePosts) {
    const expectedHash = sourceHashes.get(source) ?? null;
    for (const locale of locales) {
      const expectedPath = join(blogDir, locale, ...source.split('/'));
      if (!existsSync(expectedPath)) {
        missing.push({
          source,
          locale,
          expected: `src/content/blog/${locale}/${source}`,
          status: 'missing',
        });
        continue;
      }

      if (!isTranslationFresh(expectedPath, expectedHash)) {
        const content = readFileSync(expectedPath, 'utf8');
        const frontmatter = getFrontmatter(content);
        missing.push({
          source,
          locale,
          expected: `src/content/blog/${locale}/${source}`,
          status: 'stale',
          expectedHash,
          actualHash: frontmatter ? readSourceHash(frontmatter) : null,
        });
      }
    }
  }

  return missing;
}
