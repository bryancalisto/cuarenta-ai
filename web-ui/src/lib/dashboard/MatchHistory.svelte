<script lang="ts">
	/**
	 * The paged match history table.
	 *
	 * WHY IT PAGES INSTEAD OF SCROLLING FOREVER: the API answers with a fixed page plus the `total`,
	 * and the count is part of the answer an operator wants ("of how many?"). Loading the table only
	 * when the section is opened is the page's job; this component fetches on mount, because being
	 * mounted IS being opened.
	 */
	import { onMount } from 'svelte';
	import {
		adminErrorMessage,
		getGames,
		isUnauthorized,
		type GameRow
	} from '$lib/admin-api';
	import { t } from '$lib/i18n.svelte';
	import { formatDateTime, statusLabel } from './format';
	import { STATUS_CLASS } from './theme';
	import TeamTagLabel from './TeamTagLabel.svelte';

	let {
		token,
		onReplay,
		onUnauthorized
	}: {
		token: string;
		onReplay: (gameId: string) => void;
		onUnauthorized: () => void;
	} = $props();

	/** Big enough to read a session's worth of games, small enough to stay one request. */
	const PAGE_SIZE = 25;

	let rows = $state<GameRow[]>([]);
	let total = $state(0);
	let offset = $state(0);
	let loading = $state(true);
	let failed = $state(false);
	/** The backend's own words, when it had any: the translation is added at render time. */
	let errorMessage = $state('');

	async function load(nextOffset: number) {
		loading = true;
		failed = false;
		errorMessage = '';
		try {
			const page = await getGames(token, PAGE_SIZE, Math.max(0, nextOffset));
			rows = page.games ?? [];
			total = page.total ?? rows.length;
			offset = page.offset ?? nextOffset;
		} catch (caught) {
			if (isUnauthorized(caught)) {
				onUnauthorized();
				return;
			}
			failed = true;
			errorMessage = adminErrorMessage(caught, '');
		} finally {
			loading = false;
		}
	}

	onMount(() => {
		void load(0);
	});

	const from = $derived(total === 0 ? 0 : offset + 1);
	const to = $derived(offset + rows.length);
	const hasPrevious = $derived(offset > 0);
	const hasNext = $derived(to < total);
</script>

<section
	class="rounded-xl sm:rounded-2xl border border-white/10 bg-[var(--card-bg)] p-3 sm:p-4 backdrop-blur-xl"
>
	<div class="flex flex-wrap items-baseline justify-between gap-2">
		<h3 class="text-sm font-semibold text-slate-100">{t('Match history')}</h3>
		<p class="text-xs text-slate-400">{t('Every game this backend recorded, newest first.')}</p>
	</div>

	{#if failed}
		<div
			role="alert"
			class="mt-3 rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200"
		>
			{errorMessage || t('Could not load the match history')}
		</div>
	{:else if loading && rows.length === 0}
		<p class="mt-3 text-xs text-slate-400">{t('Loading games...')}</p>
	{:else if rows.length === 0}
		<div
			class="mt-3 rounded-2xl border border-dashed border-white/15 bg-slate-950/30 px-4 py-6 text-center text-xs text-slate-400"
		>
			{t('No games recorded yet.')}
		</div>
	{:else}
		<!-- The table is wider than a phone: it scrolls inside this box, so the page itself never
		     does (that is what breaks the layout on a phone). -->
		<div class="mt-3 overflow-x-auto">
			<table class="w-full min-w-[54rem] border-collapse text-left text-xs sm:text-sm">
				<thead>
					<tr class="text-[0.6rem] uppercase tracking-[0.1em] text-slate-400">
						<th class="px-2 py-2 font-semibold">{t('Game')}</th>
						<th class="px-2 py-2 font-semibold">{t('Started')}</th>
						<th class="px-2 py-2 font-semibold">{t('Status')}</th>
						<th class="px-2 py-2 font-semibold">{t('Deals')}</th>
						<th class="px-2 py-2 font-semibold">{t('Team win')}</th>
						<th class="px-2 py-2 font-semibold">{t('Team lose')}</th>
						<th class="px-2 py-2 font-semibold">{t('Plys')}</th>
						<th class="px-2 py-2 font-semibold">{t('Score')}</th>
						<th class="px-2 py-2 font-semibold">{t('Replay')}</th>
					</tr>
				</thead>
				<tbody>
					{#each rows as row (row.id)}
						<tr class="border-t border-white/5">
							<td class="px-2 py-2 font-mono text-[0.7rem] text-slate-300">{row.id}</td>
							<td class="whitespace-nowrap px-2 py-2 text-slate-300"
								>{formatDateTime(row.startedAt)}</td
							>
							<td class="px-2 py-2">
								<span
									class={`rounded-full px-2 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide ${STATUS_CLASS[row.status]}`}
								>
									{statusLabel(row.status)}
								</span>
							</td>
							<td class="px-2 py-2 tabular-nums text-slate-300">{row.deals}</td>
							<td class="px-2 py-2"><TeamTagLabel tag={row.winner} /></td>
							<td class="px-2 py-2"><TeamTagLabel tag={row.loser} /></td>
							<td class="px-2 py-2 tabular-nums text-slate-300">{row.plies}</td>
							<td class="whitespace-nowrap px-2 py-2 tabular-nums text-slate-200"
								>{row.scoreTeam1} – {row.scoreTeam2}</td
							>
							<td class="px-2 py-2">
								<button
									class="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[0.7rem] font-semibold text-amber-200 transition hover:border-amber-300/60 hover:bg-amber-400/10"
									aria-label={t('Replay of game {id}', { id: row.id })}
									onclick={() => onReplay(row.id)}
								>
									{t('Replay')}
								</button>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>

		<div class="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
			<p>{t('Showing {from}-{to} of {total} games', { from, to, total })}</p>
			<div class="flex items-center gap-2">
				<button
					class="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 font-semibold text-slate-200 transition hover:border-amber-300/60 disabled:cursor-not-allowed disabled:opacity-40"
					disabled={!hasPrevious || loading}
					onclick={() => void load(offset - PAGE_SIZE)}
				>
					← {t('Previous')}
				</button>
				<button
					class="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 font-semibold text-slate-200 transition hover:border-amber-300/60 disabled:cursor-not-allowed disabled:opacity-40"
					disabled={!hasNext || loading}
					onclick={() => void load(offset + PAGE_SIZE)}
				>
					{t('Next')} →
				</button>
			</div>
		</div>
	{/if}
</section>
