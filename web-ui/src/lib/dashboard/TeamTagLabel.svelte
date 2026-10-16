<script lang="ts">
	/**
	 * A team tag ("who won") as a readable badge: the team number, then one chip per seat in seat
	 * order.
	 *
	 * WHY CHIPS AND NOT THE LABEL STRING: the backend already sends `label` ("1 (H, AI)"), but a
	 * wall of "H, AI" text in a table is exactly where a human-vs-AI team stops being visible. The
	 * chips reuse the H/AI colour pair from `layout.css`, so a mixed team reads at a glance. The
	 * label is still carried as the tooltip, so nothing is lost.
	 */
	import type { SeatKind, TeamTag } from '$lib/admin-api';
	import { t } from '$lib/i18n.svelte';
	import SeatChip from './SeatChip.svelte';

	let { tag }: { tag: TeamTag | null } = $props();

	/** The seat kinds, from `seats` when the backend sent them. */
	function kindList(value: TeamTag): SeatKind[] {
		if (value.seats && value.seats.length > 0) return value.seats;
		// A payload carrying only the formatted label ("1 (H, AI)"): read the tokens out of it
		// rather than showing nothing, so the chips survive a field going missing on the wire.
		return [...value.label.matchAll(/\b(AI|H)\b/g)].map((match) =>
			match[1] === 'AI' ? 'ai' : 'human'
		);
	}

	const kinds = $derived(tag ? kindList(tag) : []);
</script>

{#if tag}
	<span class="inline-flex flex-wrap items-center gap-1" title={tag.label}>
		<span class="text-slate-200">{t('Team {team}', { team: tag.team })}</span>
		{#each kinds as kind, position (position)}
			<SeatChip {kind} />
		{/each}
	</span>
{:else}
	<!-- A canceled or still-running game has no winner: a muted dash, not an empty cell that reads
	     like a rendering bug. -->
	<span class="text-slate-600" aria-label={t('No result')}>—</span>
{/if}
