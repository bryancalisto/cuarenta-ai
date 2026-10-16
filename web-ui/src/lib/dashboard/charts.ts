/**
 * The two charts' prop shapes.
 *
 * WHY NOT INSIDE THE COMPONENTS: a Svelte 5 component can no longer `export` anything from its
 * instance script, so the page would have to import the types from somewhere else anyway. A plain
 * module also means both charts agree on what a "series" and a "bar" are.
 */

/** One point of a line/area series. `y === null` is a day with nothing to plot (see TimeseriesPoint). */
export type LinePoint = { x: string; y: number | null };

export type LineSeries = {
	/** Already translated by the page: the chart renders its own legend. */
	name: string;
	color: string;
	points: LinePoint[];
	/** Which y axis this series is measured against. Defaults to 'left'. */
	axis?: 'left' | 'right';
	/** Fill the area under the line as well as drawing it. */
	area?: boolean;
	dashed?: boolean;
};

export type BarDatum = {
	/** The x tick, and the identity of the bar. */
	label: string;
	/** Value in the chart's own unit (a percentage for the win-rate charts). */
	value: number;
	/** One line under the label in the tooltip, e.g. "7 of 12 games". */
	detail?: string;
	color?: string;
};
