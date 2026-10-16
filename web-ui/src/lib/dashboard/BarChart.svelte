<script lang="ts">
	/**
	 * A responsive bar chart, drawn as SVG from D3 scales -- same division of labour as LineChart:
	 * D3 computes the geometry, Svelte renders it.
	 *
	 * WHY THE LABELS ROTATE ON THEIR OWN: composition labels ("H+AI vs AI+AI") are long and there
	 * are a handful of them, so on a phone they collide while on a desktop they have plenty of room.
	 * Rather than guess at a breakpoint, the chart measures what it was given: when a band is
	 * narrower than a 6-character label, the labels tilt.
	 */
	import { max, scaleBand, scaleLinear } from 'd3';
	import type { BarDatum } from './charts';
	import { CHART_COLORS } from './theme';
	import { t } from '$lib/i18n.svelte';

	let {
		bars,
		width,
		height = 210,
		yLabel = '',
		xLabel = '',
		domainMax,
		valueFormat = (value: number) => String(Math.round(value))
	}: {
		bars: BarDatum[];
		width: number;
		height?: number;
		yLabel?: string;
		/** What the bars are grouped by, e.g. "Simulations per turn". */
		xLabel?: string;
		/** A fixed top for the value axis (100 for the win-rate charts), so bars stay comparable. */
		domainMax?: number;
		valueFormat?: (value: number) => string;
	} = $props();

	/** A missing or non-finite value would produce `NaN` in the path data and a broken SVG. */
	const safe = (value: number) => (Number.isFinite(value) ? value : 0);

	const labels = $derived(bars.map((bar) => bar.label));
	const longest = $derived(labels.reduce((longest, label) => Math.max(longest, label.length), 0));
	const rotateTicks = $derived(
		longest > 6 && width / Math.max(1, bars.length) < longest * 7 + 18
	);

	const margin = $derived({
		top: 16,
		right: 12,
		// Room for the tick labels (tilted when they are long), plus a line for the axis title.
		bottom: (rotateTicks ? 56 : 34) + (xLabel ? 16 : 0),
		left: yLabel ? 52 : 42
	});
	const innerW = $derived(Math.max(20, width - margin.left - margin.right));
	const innerH = $derived(Math.max(20, height - margin.top - margin.bottom));

	const xScale = $derived(
		scaleBand<string>()
			.domain(labels)
			.range([0, innerW])
			.padding(0.28)
	);

	const yScale = $derived.by(() => {
		const top = max(bars.map((bar) => safe(bar.value))) ?? 0;
		const upper = Math.max(domainMax ?? 0, top);
		return scaleLinear()
			.domain([0, upper > 0 ? upper : 1])
			.nice()
			.range([innerH, 0]);
	});

	const yTicks = $derived(yScale.ticks(4));

	function barTitle(bar: BarDatum): string {
		const value = valueFormat(safe(bar.value));
		return bar.detail ? `${bar.label} · ${value} · ${bar.detail}` : `${bar.label} · ${value}`;
	}
</script>

<svg width={width} height={height} role="img" aria-label={t('Bar chart')} class="block">
	<g transform="translate({margin.left}, {margin.top})">
		{#each yTicks as tick (tick)}
			<line x1="0" x2={innerW} y1={yScale(tick)} y2={yScale(tick)} stroke="rgba(148,163,184,0.16)" />
			<text
				x={-6}
				y={yScale(tick)}
				text-anchor="end"
				dominant-baseline="middle"
				fill="var(--text-muted)"
				font-size="10">{valueFormat(tick)}</text
			>
		{/each}

		<line x1="0" x2={innerW} y1={innerH} y2={innerH} stroke="rgba(148,163,184,0.35)" />

		{#each bars as bar, barIndex (barIndex)}
			{@const left = xScale(bar.label) ?? 0}
			{@const centre = left + xScale.bandwidth() / 2}
			{@const value = safe(bar.value)}
			{@const top = yScale(value)}
			<rect
				x={left}
				y={top}
				width={xScale.bandwidth()}
				height={Math.max(0, innerH - top)}
				rx="4"
				fill={bar.color ?? CHART_COLORS.accent}
			>
				<title>{barTitle(bar)}</title>
			</rect>
			<text
				x={centre}
				y={top - 4}
				text-anchor="middle"
				fill="var(--text-muted)"
				font-size="10">{valueFormat(value)}</text
			>
			{#if rotateTicks}
				<!-- Anchored at its right end and tilted, so the label grows down-left instead of
				     over its neighbour. -->
				<text
					transform="rotate(-35 {centre} {innerH + 14})"
					x={centre}
					y={innerH + 14}
					text-anchor="end"
					fill="var(--text-muted)"
					font-size="10">{bar.label}</text
				>
			{:else}
				<text
					x={centre}
					y={innerH + 16}
					text-anchor="middle"
					fill="var(--text-muted)"
					font-size="10">{bar.label}</text
				>
			{/if}
		{/each}

		{#if yLabel}
			<!-- Rotated the same way as LineChart's axis titles (local (x, y) maps to (y, -x)). -->
			<text
				transform="rotate(-90)"
				x={-innerH / 2}
				y={-margin.left + 12}
				text-anchor="middle"
				fill="var(--text-muted)"
				font-size="10">{yLabel}</text
			>
		{/if}

		{#if xLabel}
			<!-- Under the ticks, centred: the bars are grouped by whatever this names. -->
			<text
				x={innerW / 2}
				y={innerH + (rotateTicks ? 68 : 34)}
				text-anchor="middle"
				fill="var(--text-muted)"
				font-size="10">{xLabel}</text
			>
		{/if}
	</g>
</svg>
