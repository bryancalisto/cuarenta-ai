#!/usr/bin/env node
/**
 * Verify the UI's translations against the code that uses them.
 *
 * Natural-language keys (`t('Sign in')`) make missing translations degrade gracefully, which
 * is exactly why they can rot silently: rename an English string and the Spanish one becomes
 * dead weight, delete a `t()` call and its translation lingers. This script is the guard, and
 * it checks three things:
 *
 *   1. every `t('...')` key in the sources has a Spanish entry (a key with no entry shows
 *      English to a Spanish player — the failure this file exists to prevent);
 *   2. every Spanish entry is used somewhere (an orphan means an English string was renamed
 *      and its translation was left behind — it would silently stop being used);
 *   3. `{placeholders}` match on both sides (a translation that drops `{email}` loses
 *      information, one that invents `{id}` prints it literally).
 *
 * Run: npm run check:i18n   (also part of `npm run check`)
 */

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SOURCES = [join(ROOT, 'src')];
const I18N_FILE = join(ROOT, 'src', 'lib', 'i18n.svelte.ts');

/** Every file with UI text in it. */
function walk(dir) {
	const out = [];
	for (const entry of readdirSync(dir)) {
		const full = join(dir, entry);
		if (statSync(full).isDirectory()) out.push(...walk(full));
		else if (/\.(svelte|ts)$/.test(entry)) out.push(full);
	}
	return out;
}

/** Keys passed to `t('...')` in the sources (including the i18n module's own fallbacks). */
function usedKeys() {
	const keys = new Map(); // key -> first file that uses it
	for (const dir of SOURCES) {
		for (const file of walk(dir)) {
			const source = readFileSync(file, 'utf8');
			// Either quote style: `t('Sign in')` and `t("AI's turn")` are both keys (a key
			// containing an apostrophe has to be double-quoted).
			for (const match of source.matchAll(/\bt\(\s*(['"])((?:\\.|(?!\1)[^\\])*)\1/g)) {
				if (!keys.has(match[2])) keys.set(match[2], relative(ROOT, file));
			}
			// `t(cond ? 'A' : 'B')` — both branches are keys.
			for (const match of source.matchAll(
				/\bt\(\s*[^)]*\?\s*(['"])((?:\\.|(?!\1)[^\\])*)\1\s*:\s*(['"])((?:\\.|(?!\3)[^\\])*)\3/g
			)) {
				for (const key of [match[2], match[4]]) {
					if (!keys.has(key)) keys.set(key, relative(ROOT, file));
				}
			}
		}
	}
	return keys;
}

/** The Spanish dictionary, parsed out of the i18n module. */
function spanishDictionary() {
	const source = readFileSync(I18N_FILE, 'utf8');
	const start = source.indexOf('const ES: Record<string, string> = {');
	if (start < 0) throw new Error('cannot find the ES dictionary in i18n.svelte.ts');
	const body = source.slice(start, source.indexOf('\n};', start));
	// Values may sit on the line after their key; pull them up so one entry is one line.
	const flat = body.replace(/:\s*\n\s*/g, ': ');
	// Either quote style, on both sides. This used to be single-quotes-only, which made a
	// double-quoted key (`"AI's turn": ...`, needed when the key contains an apostrophe)
	// completely invisible: it was never checked for use NOR for having a translation.
	const entry = /^\s*(["'])((?:\\.|(?!\1)[^\\])*)\1\s*:\s*(["'])((?:\\.|(?!\3)[^\\])*)\3\s*,?\s*$/;
	const entries = new Map();
	for (const line of flat.split('\n')) {
		const match = line.match(entry);
		if (match) entries.set(match[2], match[4]);
	}
	return entries;
}

function placeholders(text) {
	return [...text.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();
}

const keys = usedKeys();
const es = spanishDictionary();
const problems = [];

for (const [key, file] of keys) {
	if (!es.has(key)) problems.push(`missing translation: ${JSON.stringify(key)} (used in ${file})`);
}

for (const [key, value] of es) {
	if (!keys.has(key)) problems.push(`unused translation: ${JSON.stringify(key)} (→ ${JSON.stringify(value)})`);
}

for (const [key, value] of es) {
	const want = placeholders(key);
	const got = placeholders(value);
	if (want.join(',') !== got.join(',')) {
		problems.push(
			`placeholder mismatch for ${JSON.stringify(key)}: ` +
				`English has [${want.join(', ')}], Spanish has [${got.join(', ')}]`
		);
	}
}

if (problems.length > 0) {
	console.error(`\ncheck-i18n: ${problems.length} problem(s)\n`);
	for (const problem of problems) console.error(`  ✗ ${problem}`);
	console.error('');
	process.exit(1);
}

console.log(
	`check-i18n: OK (${keys.size} keys, ${es.size} Spanish translations, placeholders match)`
);
