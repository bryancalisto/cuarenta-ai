<script lang="ts">
	/**
	 * The turn-by-turn replay of one game: fetching, the ply pointer, and the frame chrome.
	 *
	 * The picture of a position lives in `ReplayFrameView.svelte` -- this file only decides WHICH
	 * frame is on screen. `frames[0]` is the opening deal and `frames[i]` is the position after ply
	 * `i`, so the arrows walk the game one ply at a time (with Home/End for the ends, Escape to
	 * close).
	 */
	import { adminErrorMessage, getReplay, isUnauthorized, type Replay } from '$lib/admin-api';
	import { t } from '$lib/i18n.svelte';
	import { formatDateTime, statusLabel } from './format';
	import ReplayFrameView from './ReplayFrameView.svelte';
	import { STATUS_CLASS } from './theme';

	let {
		token,
		gameId,
		onClose,
		onUnauthorized
	}: {
		token: string;
		gameId: string;
		onClose: () => void;
		onUnauthorized: () => void;
	} = $props();

	let replay = $state<Replay | null>(null);
	let loading = $state(true);
	let failed = $state(false);
	/** The backend's own words when it had any; the fallback is translated at render time. */
	let errorMessage = $state('');
	let index = $state(0);
	let panel: HTMLDivElement | undefined = $state();

	const frames = $derived(replay?.frames ?? []);
	const lastIndex = $derived(Math.max(0, frames.length - 1));
	const frame = $derived(frames[Math.min(index, lastIndex)] ?? null);

	// Re-runs when the viewer is pointed at another game. `live` is what keeps a slow answer for the
	// previous game from overwriting the panel that replaced it.
	$effect(() => {
		const id = gameId;
		let live = true;
		loading = true;
		failed = false;
		errorMessage = '';
		replay = null;
		index = 0;
		void (async () => {
			try {
				const data = await getReplay(token, id);
				if (!live) return;
				replay = data;
			} catch (caught) {
				if (!live) return;
				if (isUnauthorized(caught)) {
					onUnauthorized();
					return;
				}
				failed = true;
				errorMessage = adminErrorMessage(caught, '');
			} finally {
				if (live) loading = false;
			}
		})();
		return () => {
			live = false;
		};
	});

	$effect(() => {
		// Once there is a frame to look at, put the focus inside the dialog: it is what makes Tab walk
		// the controls instead of whatever was focused behind the overlay.
		if (replay && panel) panel.focus();
	});

	function step(delta: number) {
		index = Math.min(lastIndex, Math.max(0, index + delta));
	}

	// On the window rather than on the panel: focus may sit on one of the arrow buttons (whose
	// keydown still bubbles to the window), and the overlay covers the page, so nothing behind it
	// wants the arrow keys.
	function onKeydown(event: KeyboardEvent) {
		if (event.key === 'ArrowLeft') {
			step(-1);
			event.preventDefault();
		} else if (event.key === 'ArrowRight') {
			step(1);
			event.preventDefault();
		} else if (event.key === 'Home') {
			index = 0;
			event.preventDefault();
		} else if (event.key === 'End') {
			index = lastIndex;
			event.preventDefault();
		} else if (event.key === 'Escape') {
			onClose();
		}
	}
</script>

<svelte:window onkeydown={onKeydown} />

<div class="fixed inset-0 z-50 overflow-y-auto bg-slate-950/85 p-2 backdrop-blur-sm sm:p-4">
	<div
		bind:this={panel}
		tabindex="-1"
		role="dialog"
		aria-modal="true"
		aria-label={t('Replay of game {id}', { id: gameId })}
		class="mx-auto w-full max-w-5xl rounded-2xl border border-white/10 bg-[var(--card-bg)] p-3 shadow-2xl shadow-black/50 outline-none sm:p-5"
	>
		<header class="flex flex-wrap items-start justify-between gap-2">
			<div>
				<h2 class="text-sm font-semibold text-slate-100 sm:text-base">
					{t('Replay of game {id}', { id: gameId })}
				</h2>
				{#if replay}
					<p class="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-slate-400">
						<span
							class={`rounded-full px-2 py-0.5 font-semibold uppercase tracking-wide ${STATUS_CLASS[
								replay.game.status
							]}`}
						>
							{statusLabel(replay.game.status)}
						</span>
						<span>{t('Deals')}: {replay.game.deals}</span>
						<span>{t('Plys')}: {replay.game.plies}</span>
						<span>{formatDateTime(replay.game.startedAt)}</span>
					</p>
				{/if}
			</div>
			<button
				class="dash-button"
				onclick={onClose}
			>
				{t('Close')}
			</button>
		</header>

		{#if loading}
			<p class="mt-4 text-xs text-slate-400">{t('Loading replay…')}</p>
		{:else if failed}
			<div
				role="alert"
				class="mt-4 rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200"
			>
				{errorMessage || t('Could not load the replay')}
			</div>
		{:else if frame && replay}
			<!-- The two ends are a jump, the two inner arrows are one ply. -->
			<div class="mt-3 flex flex-wrap items-center gap-2">
				<button
					class="replay-arrow"
					disabled={index === 0}
					aria-label={t('First ply')}
					title={t('First ply')}
					onclick={() => (index = 0)}>⏮</button
				>
				<button
					class="replay-arrow"
					disabled={index === 0}
					aria-label={t('Previous ply')}
					title={t('Previous ply')}
					onclick={() => step(-1)}>←</button
				>
				<span
					class="rounded-full border border-white/10 bg-slate-950/50 px-3 py-1 text-xs tabular-nums text-slate-200"
				>
					{t('ply {current} / {total}', { current: frame.ply, total: lastIndex })}
				</span>
				<button
					class="replay-arrow"
					disabled={index === lastIndex}
					aria-label={t('Next ply')}
					title={t('Next ply')}
					onclick={() => step(1)}>→</button
				>
				<button
					class="replay-arrow"
					disabled={index === lastIndex}
					aria-label={t('Last ply')}
					title={t('Last ply')}
					onclick={() => (index = lastIndex)}>⏭</button
				>
			</div>

			<div class="mt-3">
				<ReplayFrameView {frame} game={replay.game} />
			</div>
		{:else}
			<p class="mt-4 text-xs text-slate-400">{t('No data yet.')}</p>
		{/if}
	</div>
</div>
