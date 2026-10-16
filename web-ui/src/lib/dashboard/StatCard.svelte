<script lang="ts">
	/**
	 * One number with a label under it.
	 *
	 * The tone only ever changes the COLOUR of the value: the H/AI pair has to mean the same thing
	 * in the cards as it does in the chips and the charts, so a card about AI seats is violet and
	 * one about human seats is green here too.
	 */
	let {
		label,
		value,
		hint = '',
		tone = 'default'
	}: {
		label: string;
		value: string;
		hint?: string;
		tone?: 'default' | 'accent' | 'human' | 'ai' | 'info';
	} = $props();

	// Whole class names, spelled out: Tailwind reads the source, so a name built at runtime would
	// never be generated.
	const TONES: Record<string, string> = {
		default: 'text-slate-100',
		accent: 'text-amber-300',
		human: 'text-emerald-300',
		ai: 'text-violet-300',
		info: 'text-sky-300'
	};
</script>

<div class="rounded-xl sm:rounded-2xl border border-white/10 bg-[var(--card-bg)] p-3 sm:p-4 backdrop-blur-xl">
	<p class="text-[0.6rem] sm:text-[0.65rem] font-semibold uppercase tracking-[0.12em] text-slate-400">
		{label}
	</p>
	<p class="mt-1 text-lg sm:text-2xl font-bold tabular-nums {TONES[tone] ?? TONES.default}">{value}</p>
	{#if hint}
		<p class="mt-0.5 text-[0.7rem] leading-snug text-slate-500">{hint}</p>
	{/if}
</div>
