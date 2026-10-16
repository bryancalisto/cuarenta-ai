/**
 * The dashboard's palette, in one module.
 *
 * WHY IT EXISTS: the H/AI colour pair is used twice as CSS (the seat chips in
 * `routes/layout.css`) and twice as SVG (the seat-kind bars and the seat markers in the replay),
 * and the whole point of the pair is that a mixed team is readable at a glance -- two shades of
 * the same green would quietly destroy that. The chip rules in `layout.css` carry these same hex
 * values; keep the two in step.
 */
import type { GameStatus, SeatKind } from '$lib/admin-api';

/**
 * The status pill's colours, as whole Tailwind class names: the parser reads the source, so a
 * class assembled from a variable would never be generated.
 */
export const STATUS_CLASS: Record<GameStatus, string> = {
	finished: 'bg-emerald-400/15 text-emerald-300',
	canceled: 'bg-rose-400/15 text-rose-300',
	running: 'bg-sky-400/15 text-sky-300'
};

export const SEAT_COLORS: Record<SeatKind, string> = {
	human: '#34d399', // emerald-400, the same green the chips use
	ai: '#a78bfa' // violet-400, deliberately a different hue family, not a different shade
};

/** Series colours for the rest of the charts. Amber is the app's accent, sky its secondary. */
export const CHART_COLORS = {
	games: '#f59e0b',
	plies: '#38bdf8',
	accent: '#f59e0b',
	info: '#38bdf8'
} as const;
