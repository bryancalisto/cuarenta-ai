<script lang="ts">
	/**
	 * One frame of a replay, as pure presentation.
	 *
	 * WHY IT IS SPLIT FROM THE VIEWER: the viewer owns the fetching and the ply index, this owns the
	 * picture of a position -- and a picture is the part that is worth rendering on its own (there is
	 * no browser in this repo, so the props-in/markup-out shape is what makes the layout checkable at
	 * all). It is also what makes "every frame looks the same, only the numbers change" true by
	 * construction.
	 *
	 * The whole point of a replay in an imperfect-information game is `hands`: it is indexed by seat,
	 * so this shows what EVERY seat held, not just the mover.
	 */
	import type { GameRow, ReplayFrame, SeatKind } from '$lib/admin-api';
	import { t } from '$lib/i18n.svelte';
	import PlayingCard from '$lib/PlayingCard.svelte';
	import SeatChip from './SeatChip.svelte';
	import TeamTagLabel from './TeamTagLabel.svelte';

	let { frame, game }: { frame: ReplayFrame; game: GameRow } = $props();

	/**
	 * Seat -> kind. Filled from BOTH the seat number and the array position: the frame's arrays are
	 * indexed by seat (0..3, the engine's own numbering) while `game.seats[].seat` carries whatever
	 * the API numbered its seats with, and a lookup that misses would leave the mover unlabelled --
	 * which is the one thing this view exists to show.
	 */
	const seatKinds = $derived.by(() => {
		const kinds = new Map<number, SeatKind>();
		(game.seats ?? []).forEach((seat, position) => {
			kinds.set(seat.seat, seat.kind);
			if (!kinds.has(position)) kinds.set(position, seat.kind);
		});
		return kinds;
	});

	function kindOf(seat: number | null | undefined): SeatKind | null {
		if (seat === null || seat === undefined) return null;
		return seatKinds.get(seat) ?? null;
	}

	const cardsOnTable = $derived(frame.table.filter((card) => card >= 0));
</script>

<!-- What the move was, in words: the frame itself only carries card indices. -->
<div class="rounded-2xl border border-white/10 bg-slate-950/40 p-3">
	{#if frame.move && frame.seat !== null}
		<div class="flex flex-wrap items-center gap-2 text-xs text-slate-200 sm:text-sm">
			<span class="font-semibold">{t('Seat {seat}', { seat: frame.seat + 1 })}</span>
			<SeatChip kind={frame.kind ?? kindOf(frame.seat)} />
			<span class="text-slate-400">{t('played')}</span>
			<span class="replay-card played"><PlayingCard cardIndex={frame.move.card} /></span>
			<span>
				{frame.move.captures.length === 0
					? t('No capture')
					: t('Captured {count} cards', { count: frame.move.captures.length })}
			</span>
			<span class="text-amber-300">{t('{points} points', { points: frame.move.points })}</span>
			{#if frame.move.caida}
				<span
					class="rounded-full bg-amber-400/20 px-2 py-0.5 text-[0.6rem] font-bold uppercase text-amber-200"
				>
					{t('caída')}
				</span>
			{/if}
			{#if frame.move.limpia}
				<span
					class="rounded-full bg-sky-400/20 px-2 py-0.5 text-[0.6rem] font-bold uppercase text-sky-200"
				>
					{t('limpia')}
				</span>
			{/if}
		</div>
	{:else}
		<p class="text-xs text-slate-300 sm:text-sm">{t('Opening deal — no move yet.')}</p>
	{/if}
</div>

<!-- The table, the deck, the turn and the scores of this frame. -->
<div class="mt-3 rounded-2xl border border-white/10 bg-slate-950/40 p-3">
	<div class="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-400">
		<span class="font-semibold text-slate-200">{t('Cards on the table')}</span>
		<span>{t('Deck: {count} cards', { count: frame.deckRemaining })}</span>
		<span class="inline-flex flex-wrap items-center gap-1">
			{t('Turn:')}
			<span class="font-semibold text-slate-200"
				>{t('Seat {seat}', { seat: frame.currentPlayer + 1 })}</span
			>
			<SeatChip kind={kindOf(frame.currentPlayer)} />
		</span>
	</div>

	<div class="mt-2 flex flex-wrap items-center gap-1">
		{#each frame.table as card, position (position)}
			{#if card >= 0}
				<!-- The card just played is ringed, so the eye finds what changed between two frames
				     instead of re-reading the whole row. -->
				<span class="replay-card" class:last-placed={card === frame.lastPlaced}>
					<PlayingCard cardIndex={card} />
				</span>
			{/if}
		{/each}
		{#if cardsOnTable.length === 0}
			<span class="text-xs text-slate-500">{t('Table is empty')}</span>
		{/if}
	</div>

	<div class="mt-2 flex flex-wrap gap-2">
		{#each frame.scores as score, team (team)}
			<span class="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
				{t('Team {team}', { team: team + 1 })}
				<strong class="ml-1 tabular-nums text-amber-300">{score ?? 0}</strong>
			</span>
		{/each}
	</div>
</div>

<!-- Every seat's hand: what the replay is for. -->
<div class="mt-3 grid gap-2 sm:grid-cols-2">
	{#each frame.hands as hand, seat (seat)}
		<article
			class={`rounded-2xl border p-2 sm:p-3 ${
				frame.seat === seat ? 'border-amber-300/60 bg-amber-400/5' : 'border-white/10 bg-slate-950/40'
			}`}
		>
			<header class="flex flex-wrap items-center gap-2 text-xs">
				<span class="font-semibold text-slate-200">{t('Seat {seat}', { seat: seat + 1 })}</span>
				<SeatChip kind={kindOf(seat)} />
				{#if frame.seat === seat}
					<span
						class="rounded-full bg-amber-400 px-2 py-0.5 text-[0.6rem] font-bold uppercase text-slate-950"
						>{t('Moved')}</span
					>
				{/if}
				{#if frame.currentPlayer === seat}
					<span
						class="rounded-full bg-sky-400/20 px-2 py-0.5 text-[0.6rem] font-bold uppercase text-sky-200"
						>{t('Turn')}</span
					>
				{/if}
			</header>
			<div class="mt-2 flex flex-wrap items-center gap-1">
				{#each hand as card, position (position)}
					<!-- `-1` is an empty slot in a hand, not a card. -->
					{#if card >= 0}
						<span class="replay-card hand"><PlayingCard cardIndex={card} /></span>
					{/if}
				{/each}
			</div>
		</article>
	{/each}
</div>

<!-- The counted piles: the size always, the cards on request (a pile can be 20 deep). -->
<div class="mt-3 grid gap-2 sm:grid-cols-2">
	{#each frame.captured as pile, team (team)}
		<details class="rounded-2xl border border-white/10 bg-slate-950/40 p-2 sm:p-3">
			<summary class="cursor-pointer text-xs text-slate-300">
				{t('Team {team}', { team: team + 1 })} ·
				{t('Captured {count} cards', { count: pile.length })}
			</summary>
			<div class="mt-2 flex flex-wrap items-center gap-1">
				{#each pile as card, position (position)}
					{#if card >= 0}
						<span class="replay-card tiny"><PlayingCard cardIndex={card} /></span>
					{/if}
				{/each}
			</div>
		</details>
	{/each}
</div>

<!-- How the match ended, beside the moves that led there. -->
<div class="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-400">
	<span class="inline-flex items-center gap-1"
		>{t('Team win')}: <TeamTagLabel tag={game.winner} /></span
	>
	<span class="inline-flex items-center gap-1"
		>{t('Team lose')}: <TeamTagLabel tag={game.loser} /></span
	>
	<span>
		{t('Score')}:
		<strong class="tabular-nums text-slate-200">{game.scoreTeam1} – {game.scoreTeam2}</strong>
	</span>
</div>
