#!/usr/bin/env node
// Post-build: inject <details> wrapper around long code blocks
import { readFileSync, writeFileSync, readdirSync } from 'fs';
import { join, extname } from 'path';

const MAX_LINES = 10;

function processHtml(content) {
  // Find all pre blocks and inject details/summary
  // Pattern: <pre class="astro-code...">...</pre>
  let count = 0;
  let wrapped = 0;
  
  const result = content.replace(
    /<pre class="astro-code[^"]*"([^>]*)>([\s\S]*?)<\/pre>/g,
    (match, attrs, inner) => {
      count++;
      // Count lines
      const lineCount = (inner.match(/<span/g) || []).length || inner.split('\n').length;
      if (lineCount <= MAX_LINES) return match;
      
      wrapped++;
      const hidden = lineCount - MAX_LINES;
      
      // Extract code for summary (first line)
      const firstLine = inner.split('\n').slice(0, 1).join('\n');
      
      return `<details class="code-details" data-lines="${lineCount}">
  <summary class="code-summary">
    <svg class="code-summary-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
    <span class="code-toggle-text">Show ${hidden} more lines</span>
    <span class="code-lines-count">(${lineCount} lines)</span>
  </summary>
  <pre class="astro-code"${attrs}>${inner}</pre>
</details>`;
    }
  );
  
  console.error(`[postbuild-collapse] Blocks: ${count}, Wrapped: ${wrapped}`);
  return result;
}

// Walk dist directory and process all .html files
const distDir = './dist';
const htmlFiles = [];

function findHtml(dir) {
  const entries = readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      findHtml(full);
    } else if (extname(entry.name) === '.html') {
      htmlFiles.push(full);
    }
  }
}

findHtml(distDir);

for (const file of htmlFiles) {
  const content = readFileSync(file, 'utf8');
  const processed = processHtml(content);
  writeFileSync(file, processed, 'utf8');
  console.error(`[postbuild-collapse] Processed: ${file}`);
}
