import { routeLocales } from '../src/i18n/config.mjs';
import { findMissingTranslations, getSourcePosts } from './i18n_coverage.mjs';

const missing = findMissingTranslations();

if (missing.length > 0) {
  console.error('Missing blog i18n translations:');
  for (const item of missing) {
    console.error(`- ${item.source} -> ${item.expected}`);
  }
  console.error('\nRun `npm run translate:i18n`, review the generated files, and stage them before committing.');
  process.exit(1);
}

console.log(`i18n coverage checks passed: ${getSourcePosts().length} source posts x ${routeLocales.length} locales`);
