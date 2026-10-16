/**
 * Formatting for the dashboard.
 *
 * WHY HERE AND NOT IN THE COMPONENTS: every one of these reads the current locale, and two
 * components showing the same number in two formats is the classic way a dashboard starts lying.
 * They are plain functions called from markup, so they re-run when the language changes (Svelte
 * tracks the `locale()` read inside them during that render).
 */
import type { GameStatus } from '$lib/admin-api';
import { locale, t } from '$lib/i18n.svelte';

/**
 * A game's status, in words. The three labels already exist in the player UI, so the dashboard
 * reuses them instead of inventing a second vocabulary for the same three states.
 */
export function statusLabel(status: GameStatus): string {
	if (status === 'finished') return t('Finished');
	if (status === 'canceled') return t('Canceled');
	return t('In progress');
}

/**
 * The wire win rate, as a 0-100 percentage.
 *
 * The contract does not say whether `winRate` is a fraction (0.5) or a percentage (50), and no
 * single value can tell them apart -- except that a fraction can never exceed 1. So a value above
 * 1 is read as a percentage and everything else as a fraction, which is unambiguous for the two
 * ends (0 and 1 mean the same thing either way) and for any real rate above 1%.
 */
export function rateToPercent(rate: number | null | undefined): number | null {
	if (rate === null || rate === undefined || !Number.isFinite(rate)) return null;
	return rate > 1 ? Math.min(100, rate) : rate * 100;
}

/** The percentage of `wins` out of `total`, or 0 when nothing was played. Never NaN. */
export function sharePercent(wins: number, total: number): number {
	if (!Number.isFinite(wins) || !Number.isFinite(total) || total <= 0) return 0;
	return Math.min(100, Math.max(0, (wins / total) * 100));
}

export function formatNumber(value: number, fractionDigits = 0): string {
	if (!Number.isFinite(value)) return '—';
	return new Intl.NumberFormat(locale(), {
		minimumFractionDigits: fractionDigits,
		maximumFractionDigits: fractionDigits
	}).format(value);
}

/** A 0-100 percentage, formatted for the current locale (so 12.5 is "12,5 %" in Spanish). */
export function formatPercent(percent: number, fractionDigits = 1): string {
	if (!Number.isFinite(percent)) return '—';
	return new Intl.NumberFormat(locale(), {
		style: 'percent',
		minimumFractionDigits: 0,
		maximumFractionDigits: fractionDigits
	}).format(percent / 100);
}

/** A duration in seconds as "3m 20s". Kept short: the cards are small and this is not a stopwatch. */
export function formatDuration(seconds: number): string {
	if (!Number.isFinite(seconds) || seconds < 0) return '—';
	const whole = Math.round(seconds);
	const minutes = Math.floor(whole / 60);
	const rest = whole % 60;
	if (minutes === 0) return t('{seconds}s', { seconds: rest });
	return t('{minutes}m {seconds}s', { minutes, seconds: rest });
}

/**
 * An ISO timestamp from the API in the reader's own format, or a dash. The backend sends UTC
 * instants; `toLocaleString` renders them in the browser's zone, which is what an operator
 * comparing a game to a log file needs.
 */
export function formatDateTime(value: string | null | undefined): string {
	if (!value) return '—';
	const date = new Date(value);
	if (Number.isNaN(date.getTime())) return '—';
	return date.toLocaleString(locale(), { dateStyle: 'short', timeStyle: 'short' });
}

/**
 * A `YYYY-MM-DD` day from the timeseries, as LOCAL midnight.
 *
 * WHY NOT `new Date('2026-10-01')`: that is parsed as UTC midnight, so anywhere west of Greenwich
 * (Ecuador included) it renders as the day before -- every label on the chart one day off, which is
 * exactly the kind of bug that looks like the data is wrong. A day from the API is a calendar day,
 * not an instant, so it is read as one. Full timestamps (`startedAt`) are instants and keep their
 * zone.
 */
export function parseDay(value: string): Date {
	return new Date(/^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T00:00:00` : value);
}
