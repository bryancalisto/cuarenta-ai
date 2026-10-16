<script lang="ts">
	/**
	 * The admin statistics dashboard.
	 *
	 * CLIENT ONLY, on purpose: the app is a static SPA (adapter-static with a fallback page), so
	 * there is no server to fetch from and no session to read -- every request here needs the admin
	 * token out of localStorage, which only exists in the browser. Nothing is fetched until `onMount`
	 * for the same reason.
	 *
	 * The data is one `getOverview` (all the totals and breakdowns) plus one `getTimeseries` (the
	 * per-day series), requested together so the page appears once instead of twice. The match
	 * history and the replay are separate, on demand, because they are behind a toggle.
	 */
	import { onMount } from 'svelte';
	import {
		ADMIN_TOKEN_KEY,
		adminErrorMessage,
		adminSignIn,
		getOverview,
		getTimeseries,
		isUnauthorized,
		type Overview,
		type Timeseries
	} from '$lib/admin-api';
	import { LOCALES, initLocale, locale, setLocale, t, type Locale } from '$lib/i18n.svelte';
	import BarChart from '$lib/dashboard/BarChart.svelte';
	import ChartFrame from '$lib/dashboard/ChartFrame.svelte';
	import LineChart from '$lib/dashboard/LineChart.svelte';
	import MatchHistory from '$lib/dashboard/MatchHistory.svelte';
	import ReplayViewer from '$lib/dashboard/ReplayViewer.svelte';
	import SeatChip from '$lib/dashboard/SeatChip.svelte';
	import StatCard from '$lib/dashboard/StatCard.svelte';
	import type { BarDatum, LineSeries } from '$lib/dashboard/charts';
	import {
		formatDuration,
		formatNumber,
		formatPercent,
		rateToPercent,
		sharePercent
	} from '$lib/dashboard/format';
	import { CHART_COLORS, SEAT_COLORS } from '$lib/dashboard/theme';

	/** Where the signed-in admin's NAME is remembered. The token has its own key, owned by the API
	 *  module; this one only exists so a reload can say who is signed in. */
	const ADMIN_NAME_KEY = 'cuarenta-admin-name';
	/** A month of daily points: enough to see a trend, short enough to stay readable on a phone. */
	const TIMESERIES_DAYS = 30;
	const CHART_HEIGHT = 240;

	/** `ready` is what keeps the first render from flashing the sign-in card at an admin who is
	 *  already signed in: the token cannot be read before `onMount` (there is no `localStorage` on
	 *  the server), so the gate has to wait for that read to have happened. */
	let ready = $state(false);
	let token = $state('');
	let adminName = $state('');

	let username = $state('');
	let password = $state('');
	let signingIn = $state(false);
	/** "Your session expired" and "invalid credentials" are different problems, so they are kept
	 *  apart: one is shown after a 401 from a stats call, the other after a failed sign-in. */
	let gateNotice = $state('');
	let signInError = $state('');

	let overview = $state<Overview | null>(null);
	let timeseries = $state<Timeseries | null>(null);
	let loading = $state(false);
	let loadFailed = $state(false);
	let loadError = $state('');

	let showHistory = $state(false);
	let replayGameId = $state('');

	/** localStorage throws in private mode and in a sandboxed frame; the dashboard then works for
	 *  one session instead of falling over. */
	function readStored(key: string): string {
		try {
			return globalThis.localStorage?.getItem(key) ?? '';
		} catch {
			return '';
		}
	}

	function writeStored(key: string, value: string): void {
		try {
			if (value) globalThis.localStorage?.setItem(key, value);
			else globalThis.localStorage?.removeItem(key);
		} catch {
			// Deliberately ignored: the token still works until the tab is closed.
		}
	}

	async function loadStats() {
		if (!token) return;
		loading = true;
		loadFailed = false;
		loadError = '';
		try {
			const [stats, series] = await Promise.all([
				getOverview(token),
				getTimeseries(token, TIMESERIES_DAYS)
			]);
			overview = stats;
			timeseries = series;
		} catch (caught) {
			if (isUnauthorized(caught)) {
				handleUnauthorized();
				return;
			}
			loadFailed = true;
			loadError = adminErrorMessage(caught, '');
		} finally {
			loading = false;
		}
	}

	/** The token is gone (expired, or the backend restarted with a new secret): drop it, forget the
	 *  data it unlocked, and send the operator back to the sign-in card with the reason. */
	function handleUnauthorized() {
		writeStored(ADMIN_TOKEN_KEY, '');
		writeStored(ADMIN_NAME_KEY, '');
		token = '';
		adminName = '';
		overview = null;
		timeseries = null;
		showHistory = false;
		replayGameId = '';
		gateNotice = t('Your admin session expired. Sign in again.');
	}

	async function submitSignIn() {
		if (signingIn) return;
		signingIn = true;
		signInError = '';
		gateNotice = '';
		try {
			const response = await adminSignIn(username.trim(), password);
			writeStored(ADMIN_TOKEN_KEY, response.accessToken);
			const name = response.admin?.username || username.trim();
			writeStored(ADMIN_NAME_KEY, name);
			token = response.accessToken;
			adminName = name;
			password = '';
			await loadStats();
		} catch (caught) {
			// A 401 HERE is the credentials, not an expired session.
			signInError = isUnauthorized(caught)
				? t('Invalid credentials')
				: adminErrorMessage(caught, t('Could not sign in'));
		} finally {
			signingIn = false;
		}
	}

	function signOut() {
		writeStored(ADMIN_TOKEN_KEY, '');
		writeStored(ADMIN_NAME_KEY, '');
		token = '';
		adminName = '';
		overview = null;
		timeseries = null;
		showHistory = false;
		replayGameId = '';
		gateNotice = '';
		signInError = '';
		username = '';
		password = '';
	}

	function openReplay(gameId: string) {
		replayGameId = gameId;
	}

	onMount(() => {
		// The locale module resolves the language in the browser only (the server renders English),
		// so a returning admin reads the dashboard in the language they picked.
		initLocale();
		const stored = readStored(ADMIN_TOKEN_KEY);
		if (stored) {
			token = stored;
			adminName = readStored(ADMIN_NAME_KEY) || t('Admin');
			void loadStats();
		}
		ready = true;
	});

	const totals = $derived(overview?.totals ?? null);
	const points = $derived(timeseries?.points ?? []);
	const hasGames = $derived((totals?.games ?? 0) > 0);

	const gamesSeries: LineSeries[] = $derived([
		{
			name: t('Games'),
			color: CHART_COLORS.games,
			points: points.map((point) => ({ x: point.day, y: point.games })),
			area: true
		},
		{
			// Plies outnumber games by an order of magnitude, so they get their own axis instead of
			// flattening the games line onto the baseline.
			name: t('Plys'),
			color: CHART_COLORS.plies,
			points: points.map((point) => ({ x: point.day, y: point.plies })),
			axis: 'right'
		}
	]);

	const winRateSeries: LineSeries[] = $derived([
		{
			name: t('AI team'),
			color: SEAT_COLORS.ai,
			// A null is a day with no comparable game: `defined` breaks the line there rather than
			// drawing through zero, which is what the null means.
			points: points.map((point) => ({ x: point.day, y: rateToPercent(point.aiTeamWinRate) }))
		},
		{
			name: t('Human team'),
			color: SEAT_COLORS.human,
			points: points.map((point) => ({ x: point.day, y: rateToPercent(point.humanTeamWinRate) }))
		}
	]);
	const hasWinRateData = $derived(
		points.some((point) => point.aiTeamWinRate !== null || point.humanTeamWinRate !== null)
	);

	/** The headline comparison. It is computed from the win COUNTS rather than from the returned
	 *  rates, so it cannot disagree with the "x of y games" line printed under it. */
	const headToHead = $derived.by(() => {
		const stats = overview?.aiVsHuman;
		const games = stats?.games ?? 0;
		const humanWins = stats?.humanTeamWins ?? 0;
		const aiWins = stats?.aiTeamWins ?? 0;
		return [
			{ kind: 'human' as const, wins: humanWins, games, percent: sharePercent(humanWins, games) },
			{ kind: 'ai' as const, wins: aiWins, games, percent: sharePercent(aiWins, games) }
		];
	});

	const seatKindBars: BarDatum[] = $derived(
		(overview?.bySeatKind ?? []).map((entry) => ({
			label: entry.kind === 'human' ? t('H') : t('AI'),
			value: rateToPercent(entry.winRate) ?? 0,
			detail: t('{wins} of {seats} seats won', { wins: entry.wins, seats: entry.seats }),
			color: SEAT_COLORS[entry.kind]
		}))
	);

	const compositionBars: BarDatum[] = $derived(
		(overview?.byComposition ?? []).map((entry) => ({
			label: t('{composition} vs {opponent}', {
				composition: entry.composition,
				opponent: entry.opponent
			}),
			value: rateToPercent(entry.winRate) ?? 0,
			detail: t('{wins} of {games} games won', { wins: entry.wins, games: entry.games }),
			color: CHART_COLORS.info
		}))
	);

	const budgetBars: BarDatum[] = $derived(
		(overview?.bySearchBudget ?? []).map((entry) => ({
			label: formatNumber(entry.iterations),
			value: rateToPercent(entry.winRate) ?? 0,
			detail: t('{wins} of {games} games won by {teams} AI teams', {
				wins: entry.aiWins,
				games: entry.games,
				teams: entry.aiTeams
			}),
			color: CHART_COLORS.accent
		}))
	);
</script>

<svelte:head>
	<title>{t('Admin dashboard')}</title>
</svelte:head>

<div class="min-h-screen px-3 py-4 text-slate-100 sm:px-4 sm:py-6 md:px-6 lg:px-8">
	<div class="mx-auto flex w-full max-w-7xl flex-col gap-3 sm:gap-4 md:gap-6">
		<header
			class="rounded-xl border border-white/10 bg-white/5 p-4 shadow-2xl shadow-black/30 backdrop-blur-xl sm:rounded-2xl sm:p-5"
		>
			<div class="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
				<div class="max-w-2xl">
					<div class="culture-mark" title={t('Cuarenta, the Ecuadorian card game')}>
						<span class="culture-icon lg" role="img" aria-label={t('Ecuador')}></span>
						<div>
							<p
								class="text-xs font-semibold uppercase tracking-[0.2em] text-amber-300 sm:text-sm sm:tracking-[0.3em]"
							>
								{t('Admin dashboard')}
							</p>
							<p class="mt-1 text-[0.7rem] text-slate-400 sm:text-xs">
								{t('Games, win rates and replays recorded by this backend.')}
							</p>
						</div>
					</div>
				</div>

				<!-- Same top-right cluster as the lobby: the actions, then the language picker keeping
				     the corner. The picker is here too because this page is reachable directly, and an
				     admin who cannot read it needs the language switch on it. -->
				<div class="header-actions">
					{#if token}
						<span class="text-xs text-slate-300">
							{t('Signed in as {username}', { username: adminName })}
						</span>
						<button class="dash-button" disabled={loading} onclick={() => void loadStats()}>
							{t('Refresh')}
						</button>
						<button class="dash-button" onclick={signOut}>{t('Sign out')}</button>
					{/if}

					<a class="back-button" href="/" aria-label={t('Back to lobby')} title={t('Back to lobby')}>
						<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
							<path
								d="M15 5l-7 7 7 7"
								fill="none"
								stroke="currentColor"
								stroke-width="2.2"
								stroke-linecap="round"
								stroke-linejoin="round"
							/>
						</svg>
						<span class="back-label">{t('Back to lobby')}</span>
					</a>

					<label class="language-picker">
						<span class="sr-only">{t('Language')}</span>
						<select
							class="language-select"
							aria-label={t('Language')}
							value={locale()}
							onchange={(event) =>
								setLocale((event.currentTarget as HTMLSelectElement).value as Locale)}
						>
							{#each LOCALES as option (option.value)}
								<option value={option.value}>{option.label}</option>
							{/each}
						</select>
					</label>
				</div>
			</div>
		</header>

		{#if !ready}
			<div
				class="mx-auto w-full max-w-md rounded-2xl border border-white/10 bg-[var(--card-bg)] p-4 text-xs text-slate-400 backdrop-blur-xl"
			>
				{t('Loading statistics…')}
			</div>
		{:else if !token}
			<!-- The gate. No credentials are ever prefilled or hardcoded: the admin account lives in the
			     backend's own configuration. -->
			<section
				class="mx-auto w-full max-w-md rounded-xl border border-white/10 bg-[var(--card-bg)] p-4 shadow-2xl shadow-black/30 backdrop-blur-xl sm:rounded-2xl sm:p-6"
			>
				<h1 class="text-sm font-semibold uppercase tracking-[0.2em] text-amber-300">
					{t('Admin sign in')}
				</h1>
				<p class="mt-2 text-xs text-slate-400">{t('Sign in to see the statistics.')}</p>

				<form
					class="mt-4 space-y-4"
					onsubmit={(event) => {
						event.preventDefault();
						void submitSignIn();
					}}
				>
					<label class="block">
						<span class="mb-2 block text-sm font-medium text-slate-300">{t('Username')}</span>
						<input
							bind:value={username}
							autocomplete="username"
							type="text"
							spellcheck="false"
							autocapitalize="none"
							class="w-full rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-white outline-none ring-0 transition placeholder:text-slate-500 focus:border-amber-400/60 focus:bg-slate-900"
						/>
					</label>

					<label class="block">
						<span class="mb-2 block text-sm font-medium text-slate-300">{t('Password')}</span>
						<input
							bind:value={password}
							autocomplete="current-password"
							type="password"
							class="w-full rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-white outline-none ring-0 transition placeholder:text-slate-500 focus:border-amber-400/60 focus:bg-slate-900"
						/>
					</label>

					{#if gateNotice || signInError}
						<div
							role="alert"
							aria-live="assertive"
							class="rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200"
						>
							{gateNotice || signInError}
						</div>
					{/if}

					<button
						type="submit"
						class="inline-flex w-full items-center justify-center rounded-2xl bg-amber-400 px-4 py-3 font-semibold text-slate-950 transition hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-60"
						disabled={signingIn}
					>
						{signingIn ? t('Working...') : t('Sign in')}
					</button>
				</form>
			</section>
		{:else if loading && !overview}
			<div
				class="mx-auto w-full max-w-md rounded-2xl border border-white/10 bg-[var(--card-bg)] p-4 text-xs text-slate-400 backdrop-blur-xl"
			>
				{t('Loading statistics…')}
			</div>
		{:else if loadFailed}
			<div
				role="alert"
				class="mx-auto w-full max-w-lg rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-4 text-sm text-rose-200"
			>
				<p>{loadError || t('Could not load statistics')}</p>
				<button
					class="mt-3 rounded-2xl bg-amber-400 px-4 py-2 font-semibold text-slate-950 transition hover:bg-amber-300"
					onclick={() => void loadStats()}
				>
					{t('Retry')}
				</button>
			</div>
		{:else if overview && totals}
			<!-- 1. The numbers. -->
			<section class="grid grid-cols-2 gap-2 sm:gap-3 lg:grid-cols-3 xl:grid-cols-4">
				<StatCard
					label={t('Total games')}
					value={formatNumber(totals.games)}
					hint={t('{humans} human seats · {ais} AI seats', {
						humans: formatNumber(totals.humans),
						ais: formatNumber(totals.ais)
					})}
					tone="accent"
				/>
				<StatCard label={t('Finished games')} value={formatNumber(totals.finished)} />
				<StatCard label={t('Canceled games')} value={formatNumber(totals.canceled)} />
				<StatCard label={t('Running games')} value={formatNumber(totals.running)} />
				<StatCard label={t('Total plies')} value={formatNumber(totals.plies)} />
				<StatCard
					label={t('Average plies per game')}
					value={formatNumber(totals.avgPliesPerGame, 1)}
				/>
				<StatCard
					label={t('Average duration')}
					value={formatDuration(totals.avgDurationSeconds)}
				/>
				<StatCard
					label={t('Average AI decision time')}
					value={t('{ms} ms', { ms: formatNumber(Math.round(totals.avgAiDecisionMs)) })}
					tone="ai"
				/>
				<StatCard
					label={t('Captures')}
					value={formatNumber(totals.captures)}
					hint={`${t('caída')}: ${formatNumber(totals.caidas)} · ${t('limpia')}: ${formatNumber(
						totals.limpias
					)}`}
				/>
			</section>

			<!-- 2. The headline: AI against humans. -->
			<section
				class="rounded-xl border border-white/10 bg-[var(--card-bg)] p-3 backdrop-blur-xl sm:rounded-2xl sm:p-4"
			>
				<h2 class="text-sm font-semibold text-slate-100">{t('AI vs humans')}</h2>
				<p class="mt-1 text-xs leading-snug text-slate-400">
					{t('Games where one team was all humans and the other all AI.')}
				</p>
				<div class="mt-3 grid gap-2 sm:grid-cols-2 sm:gap-3">
					{#each headToHead as row (row.kind)}
						<div class="rounded-xl border border-white/10 bg-slate-950/40 p-3">
							<div class="flex items-center gap-2 text-xs text-slate-300">
								<SeatChip kind={row.kind} />
								{row.kind === 'human' ? t('Human team') : t('AI team')}
							</div>
							<p
								class="mt-1 text-xl font-bold tabular-nums sm:text-2xl"
								style="color:{SEAT_COLORS[row.kind]}"
							>
								{formatPercent(row.percent)}
							</p>
							<p class="text-xs text-slate-400">
								{t('{wins} of {games} games won', {
									wins: formatNumber(row.wins),
									games: formatNumber(row.games)
								})}
							</p>
							<div class="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
								<div
									class="h-full rounded-full"
									style="width:{row.percent}%; background:{SEAT_COLORS[row.kind]}"
								></div>
							</div>
						</div>
					{/each}
				</div>
			</section>

			<!-- 3. The charts. Each one says what it shows, and says so instead of drawing empty axes when
			     there is nothing to plot. -->
			<section class="grid gap-3 lg:grid-cols-2">
				<div class="lg:col-span-2">
					<ChartFrame
						title={t('Games and plies over time')}
						caption={t(
							'One point per day: how many games were played, and how many plies they took. Games use the left axis, plies the right.'
						)}
						height={CHART_HEIGHT}
						empty={!hasGames || points.length === 0}
					>
						{#snippet children(width)}
							<LineChart
								series={gamesSeries}
								{width}
								height={CHART_HEIGHT}
								leftLabel={t('Games')}
								rightLabel={t('Plys')}
							/>
						{/snippet}
					</ChartFrame>
				</div>

				<div class="lg:col-span-2">
					<ChartFrame
						title={t('Win rate over time')}
						caption={t(
							'Daily win rate of the AI team against the human team. Days with no AI-against-human game are skipped, not counted as a loss.'
						)}
						height={CHART_HEIGHT}
						empty={!hasWinRateData}
					>
						{#snippet children(width)}
							<LineChart
								series={winRateSeries}
								{width}
								height={CHART_HEIGHT}
								leftLabel={t('Win rate')}
								valueFormat={(value) => formatPercent(value, 0)}
							/>
						{/snippet}
					</ChartFrame>
				</div>

				<ChartFrame
					title={t('Win rate by seat kind')}
					caption={t('How often a human seat and an AI seat ended up on the winning team.')}
					height={220}
					empty={seatKindBars.length === 0}
				>
					{#snippet children(width)}
						<BarChart
							bars={seatKindBars}
							{width}
							height={220}
							domainMax={100}
							valueFormat={(value) => formatPercent(value, 0)}
						/>
					{/snippet}
				</ChartFrame>

				<ChartFrame
					title={t('Win rate by team composition')}
					caption={t('How often a team with a given make-up beat the opposing make-up.')}
					height={220}
					empty={compositionBars.length === 0}
				>
					{#snippet children(width)}
						<BarChart
							bars={compositionBars}
							{width}
							height={220}
							domainMax={100}
							valueFormat={(value) => formatPercent(value, 0)}
						/>
					{/snippet}
				</ChartFrame>

				<div class="lg:col-span-2">
					<ChartFrame
						title={t('AI win rate by search budget')}
						caption={t(
							'AI team win rate grouped by how many search simulations each turn was allowed.'
						)}
						height={240}
						empty={budgetBars.length === 0}
					>
						{#snippet children(width)}
							<BarChart
								bars={budgetBars}
								{width}
								height={240}
								yLabel={t('Win rate')}
								xLabel={t('Simulations per turn')}
								domainMax={100}
								valueFormat={(value) => formatPercent(value, 0)}
							/>
						{/snippet}
					</ChartFrame>
				</div>
			</section>

			<!-- 4. The history, hidden until asked for: it is the one section that can be long. -->
			<section class="flex flex-col gap-3">
				<button
					class="self-start rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-200 transition hover:border-amber-300/60 hover:bg-amber-400/10"
					aria-expanded={showHistory}
					onclick={() => (showHistory = !showHistory)}
				>
					{showHistory ? t('Hide match history') : t('Show match history')}
				</button>

				{#if showHistory}
					<MatchHistory
						{token}
						onReplay={openReplay}
						onUnauthorized={handleUnauthorized}
					/>
				{/if}
			</section>
		{/if}
	</div>
</div>

{#if replayGameId && token}
	<ReplayViewer
		{token}
		gameId={replayGameId}
		onClose={() => (replayGameId = '')}
		onUnauthorized={handleUnauthorized}
	/>
{/if}
