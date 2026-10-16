<script lang="ts">
	/**
	 * A responsive multi-series line/area chart, drawn as SVG from D3 scales and shape generators.
	 *
	 * WHY NO `d3.select`: Svelte already owns this DOM, and two owners of one subtree is a bug
	 * factory -- the classic one being a chart that survives a resize with both the old and the new
	 * axis still in it. So D3 does what it is genuinely better at (scales, ticks, path geometry)
	 * and Svelte renders the result, which makes a language or unit change just another reactive
	 * update.
	 *
	 * The width comes from ChartFrame's ResizeObserver, so the SVG is drawn at its real pixel size:
	 * the labels stay 10px instead of being scaled by a viewBox.
	 */
	import {
		area,
		curveMonotoneX,
		extent,
		line,
		max,
		scaleLinear,
		scaleTime,
		type ScaleLinear
	} from 'd3';
	import type { LineSeries } from './charts';
	import { parseDay } from './format';
	import { locale, t } from '$lib/i18n.svelte';

	let {
		series,
		width,
		height = 220,
		leftLabel = '',
		rightLabel = '',
		valueFormat = (value: number) => String(Math.round(value * 10) / 10)
	}: {
		series: LineSeries[];
		width: number;
		height?: number;
		leftLabel?: string;
		rightLabel?: string;
		valueFormat?: (value: number) => string;
	} = $props();

	type Datum = { x: Date; y: number | null };

	// The right margin grows when there is a right axis, or its labels would be clipped.
	const margin = $derived({
		top: 14,
		right: rightLabel ? 42 : 14,
		bottom: 30,
		left: leftLabel ? 52 : 42
	});
	const innerW = $derived(Math.max(20, width - margin.left - margin.right));
	const innerH = $derived(Math.max(20, height - margin.top - margin.bottom));

	const rows = $derived(
		series.map((entry) => ({
			...entry,
			// `parseDay` keeps a calendar day a calendar day: a plain `new Date('2026-10-01')` is UTC
			// midnight, which draws and labels the previous day west of Greenwich.
			data: entry.points.map((point): Datum => ({ x: parseDay(point.x), y: point.y }))
		}))
	);

	const xScale = $derived.by(() => {
		const [lo, hi] = extent(rows.flatMap((entry) => entry.data.map((point) => point.x)));
		let start = lo ?? new Date(Date.now() - 86_400_000);
		let end = hi ?? new Date();
		// A single day (or none at all) collapses the domain to one instant, and then every tick
		// lands on the same x. Half a day either side keeps the axis drawable and finite.
		if (end.getTime() - start.getTime() <= 0) {
			start = new Date(start.getTime() - 43_200_000);
			end = new Date(end.getTime() + 43_200_000);
		}
		return scaleTime().domain([start, end]).range([0, innerW]);
	});

	/** Finite values of the series on one axis, so a null (a day with nothing to plot) is skipped. */
	function valuesOn(which: 'left' | 'right'): number[] {
		return rows
			.filter((entry) => (entry.axis ?? 'left') === which)
			.flatMap((entry) => entry.data.map((point) => point.y))
			.filter((value): value is number => value !== null && Number.isFinite(value));
	}

	/** Zero-based axes: these are counts and rates, so a floating baseline would exaggerate. */
	function scaleFor(values: number[]): ScaleLinear<number, number> {
		const top = max(values) ?? 0;
		return scaleLinear()
			.domain([0, top > 0 ? top : 1])
			.nice()
			.range([innerH, 0]);
	}

	const yLeft = $derived(scaleFor(valuesOn('left')));
	const yRight = $derived(scaleFor(valuesOn('right')));
	const hasRight = $derived(rows.some((entry) => (entry.axis ?? 'left') === 'right'));

	const shapes = $derived(
		rows.map((entry) => {
			const y = (entry.axis ?? 'left') === 'right' ? yRight : yLeft;
			const defined = (point: Datum) => point.y !== null && Number.isFinite(point.y);
			const linePath = line<Datum>()
				.defined(defined)
				.x((point) => xScale(point.x))
				.y((point) => y(point.y as number))
				.curve(curveMonotoneX);
			const areaPath = area<Datum>()
				.defined(defined)
				.x((point) => xScale(point.x))
				.y0(y(0))
				.y1((point) => y(point.y as number))
				.curve(curveMonotoneX);
			return {
				name: entry.name,
				color: entry.color,
				dashed: entry.dashed ?? false,
				y,
				data: entry.data,
				d: linePath(entry.data) ?? '',
				areaD: entry.area ? (areaPath(entry.data) ?? '') : ''
			};
		})
	);

	const xTicks = $derived(xScale.ticks(6));
	const yLeftTicks = $derived(yLeft.ticks(4));
	const yRightTicks = $derived(hasRight ? yRight.ticks(4) : []);
	// Locale-aware, so switching to Spanish re-labels the days instead of leaving "Oct 12".
	const dayFormat = $derived(new Intl.DateTimeFormat(locale(), { month: 'short', day: 'numeric' }));

	function pointTitle(name: string, point: Datum): string {
		const when = point.x.toLocaleDateString(locale(), {
			year: 'numeric',
			month: 'short',
			day: 'numeric'
		});
		const value = point.y === null ? t('no data') : valueFormat(point.y);
		return `${name} · ${when} · ${value}`;
	}
</script>

{#if series.length > 1}
	<!-- A legend is only worth its space when there is more than one line to tell apart. -->
	<div class="mb-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[0.7rem] text-slate-400">
		{#each series as entry, i (i)}
			<span class="inline-flex items-center gap-1.5">
				<span class="inline-block h-2 w-2 rounded-full" style="background:{entry.color}"></span>
				{entry.name}
			</span>
		{/each}
	</div>
{/if}

<svg width={width} height={height} role="img" aria-label={t('Line chart')} class="block">
	<g transform="translate({margin.left}, {margin.top})">
		<!-- Horizontal gridlines, and the left axis' own numbers. -->
		{#each yLeftTicks as tick (tick)}
			<line x1="0" x2={innerW} y1={yLeft(tick)} y2={yLeft(tick)} stroke="rgba(148,163,184,0.16)" />
			<text
				x={-6}
				y={yLeft(tick)}
				text-anchor="end"
				dominant-baseline="middle"
				fill="var(--text-muted)"
				font-size="10">{valueFormat(tick)}</text
			>
		{/each}

		{#each yRightTicks as tick (tick)}
			<text
				x={innerW + 6}
				y={yRight(tick)}
				text-anchor="start"
				dominant-baseline="middle"
				fill="var(--text-muted)"
				font-size="10">{valueFormat(tick)}</text
			>
		{/each}

		<line x1="0" x2={innerW} y1={innerH} y2={innerH} stroke="rgba(148,163,184,0.35)" />
		{#each xTicks as tick (tick.getTime())}
			<line
				x1={xScale(tick)}
				x2={xScale(tick)}
				y1={innerH}
				y2={innerH + 4}
				stroke="rgba(148,163,184,0.35)"
			/>
			<text
				x={xScale(tick)}
				y={innerH + 16}
				text-anchor="middle"
				fill="var(--text-muted)"
				font-size="10">{dayFormat.format(tick)}</text
			>
		{/each}

		{#each shapes as shape, seriesIndex (seriesIndex)}
			{#if shape.areaD}
				<path d={shape.areaD} fill={shape.color} opacity="0.16" />
			{/if}
			<path
				d={shape.d}
				fill="none"
				stroke={shape.color}
				stroke-width="2"
				stroke-linejoin="round"
				stroke-dasharray={shape.dashed ? '5 3' : undefined}
			/>
			{#each shape.data as point, index (index)}
				{#if point.y !== null && Number.isFinite(point.y)}
					<circle cx={xScale(point.x)} cy={shape.y(point.y)} r="2.5" fill={shape.color} />
					<!-- An invisible, much bigger circle: a 2.5px dot is not a hover target, and this
					     is what carries the <title> tooltip. Transparent (not `none`) so it still
					     receives pointer events. -->
					<circle cx={xScale(point.x)} cy={shape.y(point.y)} r="9" fill="transparent">
						<title>{pointTitle(shape.name, point)}</title>
					</circle>
				{/if}
			{/each}
		{/each}

		<!-- Axis titles, rotated so they read bottom-to-top beside their own axis. The rotation
		     maps local (x, y) to global (y, -x) for -90deg, which is why the offsets look swapped. -->
		{#if leftLabel}
			<text
				transform="rotate(-90)"
				x={-innerH / 2}
				y={-margin.left + 12}
				text-anchor="middle"
				fill="var(--text-muted)"
				font-size="10">{leftLabel}</text
			>
		{/if}
		{#if rightLabel}
			<text
				transform="rotate(90)"
				x={innerH / 2}
				y={-(innerW + 26)}
				text-anchor="middle"
				fill="var(--text-muted)"
				font-size="10">{rightLabel}</text
			>
		{/if}
	</g>
</svg>
