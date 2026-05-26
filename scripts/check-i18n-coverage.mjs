import { routeLocales } from '../src/i18n/config.mjs';
import { findMissingTranslations, getSourcePosts } from './i18n_coverage.mjs';

const missing = findMissingTranslations();

if (missing.length > 0) {
  console.error('Blog i18n translations not ready:');
  for (const item of missing) {
    const reason = item.status === 'stale' ? 'stale source' : 'missing file';
    console.error(`- [${item.status}] ${item.source} -> ${item.expected} (${reason})`);
  }
  console.error('\nRun `npm run translate:i18n` to refresh stale/missing translations, review the generated files, and stage them before committing.');
  process.exit(1);
}

console.log(`i18n coverage checks passed: ${getSourcePosts().length} source posts x ${routeLocales.length} locales`);
