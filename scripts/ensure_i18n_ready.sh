#!/usr/bin/env bash
set -euo pipefail

echo "[i18n] Generating missing local translations..."
npm run translate:i18n

echo "[i18n] Checking translation coverage..."
npm run check:i18n:coverage

if ! git diff --quiet -- src/content/blog; then
  echo
  echo "[i18n] Translation files changed after generation but are not staged."
  echo "       Review and stage these files before committing:"
  git status --short -- src/content/blog
  exit 1
fi

untracked="$(git ls-files --others --exclude-standard -- src/content/blog)"
if [ -n "$untracked" ]; then
  echo
  echo "[i18n] New translation files were generated but are not staged."
  echo "       Stage them before committing:"
  printf '%s\n' "$untracked"
  exit 1
fi

echo "[i18n] Local translations are ready."
