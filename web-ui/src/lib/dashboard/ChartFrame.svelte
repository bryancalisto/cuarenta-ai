<script lang="ts">
	/**
	 * The card around a chart: title, one-line explanation, the empty state, and the WIDTH.
	 *
	 * WHY IT MEASURES: a chart drawn from a fixed viewBox has to either stretch (and then its 10px
	 * axis labels are 6px on a phone, or 16px on a wide screen) or scroll. Measuring the container
	 * with a ResizeObserver and passing real pixels in keeps the SVG at 1:1, so the text is always
	 * text. The observer is torn down in the effect's cleanup, or every re-render would leave one
	 * behind watching a detached node.
	 */
	import type { Snippet } from 'svelte';
	import { t } from '$lib/i18n.svelte';

	let {
		title,
		caption,
		height = 220,
		empty = false,
		children
	}: {
		title: string;
		caption: string;
		height?: number;
		/** Nothing to plot: show the explanation instead of axes with no data in them. */
		empty?: boolean;
		children: Snippet<[number]>;
	} = $props();

	let container: HTMLDivElement | undefined = $state();
	let width = $state(0);

	$effect(() => {
		const node = container;
		if (!node) return;
		const measure = (value: number) => {
			// A fractional width would blur hairlines; whole pixels also keep the SVG's size
			// attribute stable so the observer does not retrigger itself.
			width = Math.max(0, Math.round(value));
		};
		measure(node.getBoundingClientRect().width);
		const observer = new ResizeObserver((entries) => {
			for (const entry of entries) measure(entry.contentRect.width);
		});
		observer.observe(node);
		return () => observer.disconnect();
	});

	const emptyMessage = $derived(t('No data yet.'));
</script>

<section
	class="rounded-xl sm:rounded-2xl border border-white/10 bg-[var(--card-bg)] p-3 sm:p-4 backdrop-blur-xl"
>
	<h3 class="text-sm font-semibold text-slate-100">{title}</h3>
	<p class="mt-1 text-xs leading-snug text-slate-400">{caption}</p>

	<div class="mt-3" bind:this={container}>
		{#if empty}
			<div
				class="flex items-center justify-center rounded-xl border border-dashed border-white/15 bg-slate-950/30 px-4 text-center text-xs text-slate-400"
				style="height:{Math.min(height, 140)}px"
			>
				{emptyMessage}
			</div>
		{:else if width > 0}
			{@render children(width)}
		{:else}
			<!-- Before the first measurement: reserve the height so the page does not jump when the
			     chart appears a frame later. -->
			<div style="height:{height}px"></div>
		{/if}
	</div>
</section>
