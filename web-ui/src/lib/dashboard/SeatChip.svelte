<script lang="ts">
	/**
	 * The H/AI badge.
	 *
	 * WHY IT IS ITS OWN COMPONENT: the same pair of colours has to mean the same thing in the match
	 * history, in the seat grid of the replay and in the charts. The colours live in `layout.css`
	 * (`.seat-chip.human` / `.seat-chip.ai`, matching `theme.ts`); this is the one place that turns a
	 * `SeatKind` into the letter a reader sees. An unknown kind renders as "?" in grey rather than a
	 * missing badge, because a seat with no kind is information too.
	 */
	import type { SeatKind } from '$lib/admin-api';
	import { t } from '$lib/i18n.svelte';

	let { kind }: { kind: SeatKind | null } = $props();

	const text = $derived(kind === 'ai' ? t('AI') : kind === 'human' ? t('H') : '?');
	const title = $derived(
		kind === 'ai' ? t('AI seat') : kind === 'human' ? t('Human seat') : t('Unknown')
	);
</script>

<span class={`seat-chip ${kind ?? 'unknown'}`} title={title}>{text}</span>
