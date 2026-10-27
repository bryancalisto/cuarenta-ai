<script lang="ts">
	import { browser } from '$app/environment';
	import { onDestroy, onMount } from 'svelte';
	import {
		createGame,
		getGame,
		getMyGames,
		getWebsocketTicket,
		joinGame,
		openGameSocket,
		signIn,
		signUp,
		startGame,
		cancelGame,
		type AuthResponse,
		type Game,
		type HumanPlayer,
		type PublicUser,
		type WsMessage
	} from '$lib/api';
	import GameBoard from '$lib/GameBoard.svelte';
	import { LOCALES, initLocale, locale, setLocale, t, type Locale } from '$lib/i18n.svelte';

	type AuthTab = 'signin' | 'signup';
	type Screen = 'auth' | 'menu' | 'create' | 'join' | 'lobby' | 'game';

	const sessionKey = 'cuarenta-session';
	const lastGameKey = 'cuarenta-last-game';
	const seats = [1, 2, 3, 4];
	// Localised labels: rebuilt from `t()` so they follow the language dropdown.
	const teams = $derived([
		{ value: 1, label: t('Team 1') },
		{ value: 2, label: t('Team 2') }
	]);

	let authTab = $state<AuthTab>('signin');
	let screen = $state<Screen>('auth');
	let username = $state('');
	let password = $state('');
	let authLoading = $state(false);
	let authError = $state('');
	let token = $state<string | null>(null);
	let currentUser = $state<PublicUser | null>(null);
	let socket = $state<WebSocket | null>(null);
	let socketStatus = $state<'offline' | 'connecting' | 'open' | 'closed'>('offline');
	let socketError = $state('');
	let pendingGameId = $state('');
	let activeGame = $state<Game | null>(null);
	let createSpot = $state(1);
	let createTeam = $state(1);
	let createLoading = $state(false);
	let createMessage = $state('');
	let joinCode = $state('');
	let joinPreview = $state<Game | null>(null);
	let joinPreviewCode = $state('');
	let joinSpot = $state(1);
	let joinTeam = $state(1);
	let joinLoading = $state(false);
	let joinMessage = $state('');
	let startLoading = $state(false);
	let cancelLoading = $state(false);
	let mainMessage = $state('');
	let lastCardPlayed = $state(-1);
	let myGames = $state<Game[]>([]);
	let myGamesLoading = $state(false);

	function safeJsonParse<T>(value: string | null): T | null {
		if (!value) return null;
		try {
			return JSON.parse(value) as T;
		} catch {
			return null;
		}
	}

	function persistSession() {
		if (!browser) return;
		const gameId = pendingGameId || activeGame?.id || '';
		if (token && currentUser) {
			localStorage.setItem(
				sessionKey,
				JSON.stringify({
					token,
					user: currentUser,
					gameId
				})
			);
			if (gameId) {
				localStorage.setItem(lastGameKey, gameId);
			} else {
				localStorage.removeItem(lastGameKey);
			}
		} else {
			localStorage.removeItem(sessionKey);
			localStorage.removeItem(lastGameKey);
		}
	}

	function setNotice(message: string) {
		mainMessage = message;
	}

	function closeSocket() {
		socket?.close();
		socket = null;
		socketStatus = 'offline';
	}

	async function subscribeToGame(gameId: string) {
		pendingGameId = gameId;
		if (!token) return;

		if (!socket || socket.readyState === WebSocket.CLOSED || socket.readyState === WebSocket.CLOSING) {
			await openSocket();
			return;
		}

		if (socket.readyState === WebSocket.OPEN) {
			socket.send(JSON.stringify({ type: 'game.subscribe', payload: { gameId } }));
		}
	}

	async function openSocket() {
		if (!token) return;

		closeSocket();
		socketStatus = 'connecting';
		
		const ticket = await getWebsocketTicket(token);
		const ws = openGameSocket(ticket);
		socket = ws;

		ws.onopen = () => {
			socketStatus = 'open';
			if (pendingGameId) {
				ws.send(JSON.stringify({ type: 'game.subscribe', payload: { gameId: pendingGameId } }));
			}
		};

		ws.onmessage = (event) => {
			const payload = safeJsonParse<WsMessage>(event.data);
			if (!payload) return;

			if (payload.type === 'game.snapshot') {
				if (payload.game.status === 'canceled') {
					closeSocket();
					activeGame = null;
					pendingGameId = '';
					screen = 'menu';
					setNotice(t('Game was canceled'));
					persistSession();
					return;
				}
				activeGame = payload.game;
				if (payload.game.status === 'running' || (payload.game.status === 'finished' && screen !== 'lobby')) {
					screen = 'game';
					mainMessage = '';
				} else {
					screen = 'lobby';
					mainMessage = t('Connected to game {id}', { id: payload.game.id });
				}
				pendingGameId = payload.game.id;
				socketError = '';
				persistSession();
			}

			if (payload.type === 'error') {
				socketError = payload.error.message;
				mainMessage = payload.error.message;
			}
		};

		ws.onclose = () => {
			socketStatus = 'closed';
		};
	}

	function loadStoredSession() {
		if (!browser) return;

		const session = safeJsonParse<{ token: string; user: PublicUser; gameId?: string }>(
			localStorage.getItem(sessionKey)
		);
		if (!session) {
			screen = 'auth';
			return;
		}

		token = session.token;
		currentUser = session.user;
		screen = 'menu';
		void loadMyGames();
		if (session.gameId) {
			pendingGameId = session.gameId;
			void restoreGame(session.gameId);
		}

		openSocket();
	}

	async function restoreGame(gameId: string) {
		if (!token) return;

		try {
			const game = await getGame(token, gameId);
			if (game.status === 'finished' || game.status === 'canceled') {
				// The stored session pointed at a game that is over: keep the player in the menu
				// rather than in a lobby with nothing left to start.
				activeGame = null;
				pendingGameId = '';
				persistSession();
				return;
			}
			activeGame = game;
			screen = 'lobby';
			pendingGameId = game.id;
			mainMessage = '';
			persistSession();
		} catch {
			activeGame = null;
			pendingGameId = '';
			persistSession();
		}
	}

	async function loadMyGames() {
		if (!token) return;

		myGamesLoading = true;
		try {
			myGames = await getMyGames(token);
		} catch (error) {
			console.error('Failed to load games:', error);
			myGames = [];
		} finally {
			myGamesLoading = false;
		}
	}

	async function submitAuth() {
		authLoading = true;
		authError = '';

		try {
			const result: AuthResponse =
				authTab === 'signin' ? await signIn(username, password) : await signUp(username, password);
			token = result.accessToken;
			currentUser = result.user;
			screen = 'menu';
			setNotice(t(authTab === 'signin' ? 'Signed in' : 'Account created'));
			persistSession();
			openSocket();
			await loadMyGames();
			if (pendingGameId) {
				await subscribeToGame(pendingGameId);
			}
		} catch (error) {
			authError = error instanceof Error ? error.message : t('Authentication failed');
		} finally {
			authLoading = false;
		}
	}

	function signOut() {
		closeSocket();
		token = null;
		currentUser = null;
		activeGame = null;
		pendingGameId = '';
		joinPreview = null;
		joinPreviewCode = '';
		mainMessage = '';
		socketError = '';
		createMessage = '';
		joinMessage = '';
		persistSession();
		screen = 'auth';
	}

	function openCurrentSession() {
		if (!activeGame) return;

		pendingGameId = activeGame.id;
		screen = 'lobby';
		setNotice(t('Opened lobby {id}', { id: activeGame.id }));
		persistSession();
		subscribeToGame(activeGame.id);
	}

	async function createLobbyGame() {
		if (!token) return;

		createLoading = true;
		createMessage = '';
		socketError = '';

		try {
			const game = await createGame(token, {
				spotIndex: createSpot,
				team: createTeam
			});
			activeGame = game;
			pendingGameId = game.id;
			screen = 'lobby';
			setNotice(t('Game {id} created', { id: game.id }));
			persistSession();
			await loadMyGames();
			subscribeToGame(game.id);
		} catch (error) {
			createMessage = error instanceof Error ? error.message : t('Could not create game');
		} finally {
			createLoading = false;
		}
	}

	async function openGameFromList(game: Game) {
		activeGame = game;
		pendingGameId = game.id;
		screen = 'lobby';
		setNotice(t('Opened game {id}', { id: game.id }));
		persistSession();
		subscribeToGame(game.id);
	}

	async function handleCancelGameFromList(game: Game, event: Event) {
		event.stopPropagation();

		const confirmed = confirm(t('Are you sure you want to cancel this game? It will be deleted immediately for all players.'));
		if (!confirmed) return;

		try {
			await cancelGame(token ?? '', game.id);
			await loadMyGames();
			setNotice(t('Game canceled successfully'));
		} catch (error) {
			setNotice(error instanceof Error ? error.message : t('Could not cancel game'));
		}
	}

	async function refreshJoinPreview(code: string) {
		if (!token) return;
		if (code.length !== 4) {
			joinPreview = null;
			joinPreviewCode = '';
			joinMessage = '';
			return;
		}
		if (code === joinPreviewCode && joinPreview) return;

		try {
			const game = await getGame(token, code);
			joinPreview = game;
			joinPreviewCode = code;
			joinMessage = '';

			// Autoselect: check Team 1 then Team 2 for the first available spot
			const team1Spots = allowedSeatsForTeam(1);
			const team2Spots = allowedSeatsForTeam(2);
			
			const t1Available = team1Spots.find(s => !isOccupied(game, s));
			if (t1Available) {
				joinTeam = 1;
				joinSpot = t1Available;
			} else {
				const t2Available = team2Spots.find(s => !isOccupied(game, s));
				if (t2Available) {
					joinTeam = 2;
					joinSpot = t2Available;
				}
			}
		} catch (error) {
			joinPreview = null;
			joinPreviewCode = code;
			joinMessage = error instanceof Error ? error.message : t('Could not load game');
		}
	}

	async function enterGame() {
		if (!token || !joinPreview) return;

		joinLoading = true;
		joinMessage = '';
		socketError = '';

		try {
			const game = await joinGame(token, joinPreview.id, {
				spotIndex: joinSpot,
				team: joinTeam
			});
			activeGame = game;
			pendingGameId = game.id;
			screen = 'lobby';
			setNotice(t('Joined game {id}', { id: game.id }));
			persistSession();
			subscribeToGame(game.id);
		} catch (error) {
			joinMessage = error instanceof Error ? error.message : t('Could not join game');
		} finally {
			joinLoading = false;
		}
	}

	async function beginGame() {
		if (!token || !activeGame) return;

		startLoading = true;
		mainMessage = '';

		try {
			const game = await startGame(token, activeGame.id);
			activeGame = game;
			pendingGameId = game.id;
			if (game.status === 'running') {
				screen = 'game';
			}
			await loadMyGames();
			setNotice(t('Game {id} started', { id: game.id }));
			persistSession();
			subscribeToGame(game.id);
		} catch (error) {
			mainMessage = error instanceof Error ? error.message : t('Could not start game');
		} finally {
			startLoading = false;
		}
	}

	async function handleCancelGame() {
		if (!activeGame) return;

		const confirmed = confirm(t('Are you sure you want to cancel this game? It will be deleted immediately for all players.'));
		if (!confirmed) return;

		cancelLoading = true;
		mainMessage = '';
		try {
			await cancelGame(token ?? '', activeGame.id);
			closeSocket();
			activeGame = null;
			pendingGameId = '';
			screen = 'menu';
			await loadMyGames();
			setNotice(t('Game canceled successfully'));
			persistSession();
		} catch (error) {
			mainMessage = error instanceof Error ? error.message : t('Could not cancel game');
		} finally {
			cancelLoading = false;
		}
	}

	function sendGameAction(cardIndex: number, captureIndices?: number[]) {
		if (!socket || socket.readyState !== WebSocket.OPEN) return;
		lastCardPlayed = cardIndex;
		socket.send(JSON.stringify({
			type: 'game.action',
			payload: {
				gameId: activeGame?.id,
				cardIndex: cardIndex,
				captureIndices: captureIndices || []
			}
		}));
	}

	function sendCaptureSelection(captureIndices: number[]) {
		if (lastCardPlayed === -1) return;
		sendGameAction(lastCardPlayed, captureIndices);
	}

	// A finished (or canceled) game is OVER, and its lobby has nothing left to do: there is no
	// game to start, no seats to fill and nothing to cancel, so it is a dead end. Leaving the
	// table therefore lands on the menu, where a new game can be created or joined, and the
	// finished game is dropped from the session instead of being restored into a lobby later.
	const gameOver = $derived(
		activeGame?.status === 'finished' || activeGame?.status === 'canceled'
	);

	function leaveTable() {
		if (!gameOver) {
			screen = 'lobby';
			mainMessage = '';
			return;
		}

		closeSocket();
		activeGame = null;
		pendingGameId = '';
		mainMessage = '';
		socketError = '';
		screen = 'menu';
		persistSession();
		void loadMyGames();
	}

	// ONE back button for the whole app, in the header next to the language dropdown, instead of
	// one per screen. The game screen used to carry its own ⬅️ in a column that was `hidden
	// lg:flex`, which left phones with no way back at all.
	const canGoBack = $derived(
		screen === 'create' || screen === 'join' || screen === 'lobby' || screen === 'game'
	);
	// The label names where the button actually goes: a running game has a lobby to return to, a
	// finished one does not.
	const backLabel = $derived(
		screen === 'game' && !gameOver ? t('Return to lobby') : t('Back to menu')
	);

	function goBack() {
		if (screen === 'game') {
			leaveTable();
			return;
		}
		// Leaving a screen also drops the notice it was showing: a message about the game you just
		// left has no business following you to the menu.
		mainMessage = '';
		createMessage = '';
		joinMessage = '';
		screen = 'menu';
	}

	function isOccupied(game: Game | null, index: number) {
		return Boolean(game?.humanPlayers.some((player) => player.index === index));
	}

	function seatLabel(index: number) {
		return t('Seat {seat}', { seat: index });
	}

	function playerSummary(player: HumanPlayer) {
		return `${seatLabel(player.index)} · ${t('Team {team}', { team: player.team })}`;
	}

	function allowedSeatsForTeam(team: number) {
		return team === 1 ? [1, 3] : [2, 4];
	}

	function isSeatAllowedForTeam(seat: number, team: number) {
		return allowedSeatsForTeam(team).includes(seat);
	}

	function playerForSeat(game: Game | null, index: number) {
		return game?.humanPlayers.find((player) => player.index === index);
	}

	function playerSummaryForSeat(game: Game | null, index: number) {
		const player = playerForSeat(game, index);
		return player ? playerSummary(player) : '';
	}

	function setCreateTeam(team: number) {
		createTeam = team;
		if (!isSeatAllowedForTeam(createSpot, team)) {
			createSpot = allowedSeatsForTeam(team)[0];
		}
	}

	function setJoinTeam(team: number) {
		joinTeam = team;
		const allowed = allowedSeatsForTeam(team);
		const firstAvailable = allowed.find(s => !isOccupied(joinPreview, s));
		if (firstAvailable) {
			joinSpot = firstAvailable;
		} else {
			joinSpot = allowed[0];
		}
	}

	function formatGameStatus(game: Game) {
		switch (game.status) {
			case 'finished':
				return t('Finished');
			case 'running':
				return t('In progress');
			case 'canceled':
				return t('Canceled');
			default:
				return t('Lobby');
		}
	}

	function joinCodeInput(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		joinCode = input.value.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 4);
		void refreshJoinPreview(joinCode);
	}

	onMount(() => {
		initLocale();
		loadStoredSession();
	});

	onDestroy(() => {
		closeSocket();
	});
</script>

<svelte:head>
	<title>{t('Cuarenta Multiplayer')}</title>
</svelte:head>

<div class="min-h-screen flex px-3 py-4 sm:px-4 sm:py-6 md:px-6 lg:px-8 text-slate-100">
	<div class="mx-auto flex w-full flex-col gap-3 sm:gap-4 md:gap-6">
		<header class="rounded-xl sm:rounded-2xl lg:rounded-[2rem] border border-white/10 bg-white/5 p-4 sm:p-5 md:p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
			<div class="flex flex-col gap-3 sm:gap-4 lg:flex-row lg:items-end lg:justify-between">
				<div class="max-w-2xl">
					<div class="culture-mark" title={t('Cuarenta, the Ecuadorian card game')}>
						<!-- The Ecuador/Quito landmark mark: the game is Ecuadorian, and this is
						     the one piece of the UI that says so. -->
						<span class="culture-icon lg" role="img" aria-label={t('Ecuador')}></span>
						<div>
							<p class="text-xs sm:text-sm font-semibold uppercase tracking-[0.2em] sm:tracking-[0.3em] text-amber-300">{t('Cuarenta Multiplayer')}</p>
							<p class="mt-1 text-[0.7rem] sm:text-xs text-slate-400">{t('Ecuador · 40 · caídas & limpias')}</p>
						</div>
					</div>
				</div>

				<!-- Top-right cluster: back, then the language picker. The picker keeps the corner,
				     the back button sits beside it, and on a phone the two wrap rather than squeeze. -->
				<div class="header-actions">
					{#if canGoBack}
						<button class="back-button" onclick={goBack} aria-label={backLabel} title={backLabel}>
							<!-- Arrow drawn inline: one glyph is not worth an icon dependency, and it
							     inherits the button's colour. -->
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
							<span class="back-label">{backLabel}</span>
						</button>
					{/if}

					<!-- Language picker: always present, top right, on every screen. A native
					     <select> because it is keyboard- and screen-reader-friendly for free. -->
					<label class="language-picker">
						<span class="sr-only">{t('Language')}</span>
						<select
							class="language-select"
							aria-label={t('Language')}
							value={locale()}
							onchange={(event) => setLocale((event.currentTarget as HTMLSelectElement).value as Locale)}
						>
							{#each LOCALES as option (option.value)}
								<option value={option.value}>{option.label}</option>
							{/each}
						</select>
					</label>
				</div>
			</div>
		</header>

		{#if screen === 'auth'}
			<section class="grid gap-3 sm:gap-4 md:gap-6 lg:grid-cols-[1.1fr_0.9fr] flex-1">
				<div class="rounded-xl sm:rounded-2xl lg:rounded-[1.75rem] border border-white/10 bg-[var(--card-bg)] p-4 sm:p-5 md:p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					<div class="flex gap-2 rounded-full bg-white/5 p-1">
						<button
							class={`flex-1 rounded-full px-4 py-2 text-sm font-medium transition ${
								authTab === 'signin' ? 'bg-amber-400 text-slate-950' : 'text-slate-300 hover:text-white'
							}`}
							onclick={() => {
								authTab = 'signin';
								authError = '';
							}}
						>
							{t('Sign in')}
						</button>
						<button
							class={`flex-1 rounded-full px-4 py-2 text-sm font-medium transition ${
								authTab === 'signup' ? 'bg-amber-400 text-slate-950' : 'text-slate-300 hover:text-white'
							}`}
							onclick={() => {
								authTab = 'signup';
								authError = '';
							}}
						>
							{t('Sign up')}
						</button>
					</div>

					<div class="mt-6 space-y-4">
						<label class="block">
							<span class="mb-2 block text-sm font-medium text-slate-300">{t('Username')}</span>
							<input
								bind:value={username}
								autocomplete="username"
								type="text"
								spellcheck="false"
								autocapitalize="none"
								class="w-full rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-white outline-none ring-0 transition placeholder:text-slate-500 focus:border-amber-400/60 focus:bg-slate-900"
								placeholder={t('yourname')}
							/>
							<span class="mt-1.5 block text-xs text-slate-500">
								{t('3-20 characters: letters, digits, . _ -')}
							</span>
						</label>

						<label class="block">
							<span class="mb-2 block text-sm font-medium text-slate-300">{t('Password')}</span>
							<input
								bind:value={password}
								autocomplete={authTab === 'signin' ? 'current-password' : 'new-password'}
								type="password"
								class="w-full rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-white outline-none ring-0 transition placeholder:text-slate-500 focus:border-amber-400/60 focus:bg-slate-900"
								placeholder="••••••••"
							/>
						</label>

						{#if authError}
							<div role="alert" aria-live="assertive" class="rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
								{authError}
							</div>
						{/if}

						<button
							class="inline-flex w-full items-center justify-center rounded-2xl bg-amber-400 px-4 py-3 font-semibold text-slate-950 transition hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-60"
							disabled={authLoading}
							onclick={() => void submitAuth()}
						>
							{authLoading
								? t('Working...')
								: authTab === 'signin'
									? t('Sign in')
									: t('Create account')}
						</button>
					</div>
				</div>
			</section>
		{:else if screen === 'menu'}
			<section class="grid gap-3 sm:gap-4 md:gap-6 lg:grid-cols-[0.9fr_1.1fr] flex-1">
				<div class="rounded-xl sm:rounded-2xl lg:rounded-[1.75rem] border border-white/10 bg-[var(--card-bg)] p-4 sm:p-5 md:p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					<div class="text-xs sm:text-sm font-semibold uppercase tracking-[0.2em] sm:tracking-[0.3em] text-amber-300">{t('Menu')}</div>
					<p class="mt-2 sm:mt-3 text-xs sm:text-sm leading-6 text-slate-300">
						{t('You are signed in as {username}. Create a new game or join an existing game.', {
							username: currentUser?.username ?? ''
						})}
					</p>

					<div class="mt-4 sm:mt-6 grid gap-2 sm:gap-3">
						<button
							class="rounded-lg sm:rounded-2xl bg-amber-400 px-4 py-2.5 sm:py-3 font-semibold text-sm sm:text-base text-slate-950 transition hover:bg-amber-300 active:scale-95"
							onclick={() => {
								screen = 'create';
								mainMessage = '';
								createMessage = '';
							}}
						>
							{t('New game')}
						</button>
						<button
							class="rounded-lg sm:rounded-2xl border border-white/10 bg-white/5 px-4 py-2.5 sm:py-3 font-semibold text-sm sm:text-base text-white transition hover:border-sky-400/60 hover:bg-sky-400/10 active:scale-95"
							onclick={() => {
								screen = 'join';
								mainMessage = '';
								joinMessage = '';
							}}
						>
							{t('Join game')}
						</button>
						<button
							class="rounded-lg sm:rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-2.5 sm:py-3 font-semibold text-sm sm:text-base text-slate-200 transition hover:bg-white/5 active:scale-95"
							onclick={() => signOut()}
						>
							{t('Sign out')}
						</button>
					</div>
				</div>

				<div class="rounded-xl sm:rounded-2xl lg:rounded-[1.75rem] flex flex-col max-h-96 sm:max-h-none lg:h-[60%] border border-white/10 bg-white/5 p-4 sm:p-5 md:p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					<div class="text-xs sm:text-sm font-semibold uppercase tracking-[0.2em] sm:tracking-[0.3em] text-sky-300">{t('My games')}</div>
					{#if myGamesLoading}
						<div class="mt-3 sm:mt-4 text-xs sm:text-sm text-slate-400">{t('Loading games...')}</div>
					{:else if myGames.length === 0}
						<div class="mt-3 sm:mt-4 rounded-2xl sm:rounded-3xl border border-dashed border-white/15 bg-slate-950/30 p-3 sm:p-5 text-xs sm:text-sm text-slate-400">
							{t('No games created yet.')}
						</div>
					{:else}
						<div class="mt-3 sm:mt-4 space-y-2 sm:space-y-3 overflow-y-auto">
							{#each myGames as game (game.id)}
								<div class="relative">
									<button
										class="w-full rounded-lg sm:rounded-2xl border border-white/10 bg-slate-950/40 p-3 sm:p-4 text-left transition hover:border-sky-400/50 hover:bg-slate-950/60 active:scale-95"
										onclick={() => openGameFromList(game)}
									>
										<div class="flex flex-wrap items-center justify-between gap-1 sm:gap-2">
											<div>
												<div class="flex items-center gap-2 flex-wrap">
													<div class="text-xs sm:text-sm text-slate-400 font-mono">{game.id}</div>
													{#if game.createdBy === currentUser?.id}
														<div class="rounded-full border border-orange-400/30 bg-orange-500/10 px-2 py-0.5 text-xs font-medium text-orange-200">
															{t('Owner')}
														</div>
													{/if}
												</div>
												<div class="mt-1 text-xs text-slate-500">
													{#if game.startedAt}
														{new Date(game.startedAt).toLocaleDateString()}
													{:else}
														{t('Created')}
													{/if}
												</div>
											</div>
											<div class="rounded-full border px-2 py-1 text-xs font-medium {
												game.status === 'running' ? 'border-green-400/30 bg-green-500/10 text-green-200' :
												game.status === 'finished' ? 'border-blue-400/30 bg-blue-500/10 text-blue-200' :
												game.status === 'canceled' ? 'border-blue-400/30 bg-grey-500/10 text-black-200' :
												'border-amber-400/30 bg-amber-500/10 text-amber-200'
											}">
												{formatGameStatus(game)}
											</div>
										</div>
										<div class="mt-2 sm:mt-3 flex flex-wrap gap-2 sm:gap-4 text-xs sm:text-sm text-slate-400">
											<div class="whitespace-nowrap">{t('Players: {count}/4', { count: game.humanPlayers.length })}</div>
											{#if game.durationSeconds}
												<div class="whitespace-nowrap">{t('Duration: {minutes}m', { minutes: Math.floor(game.durationSeconds / 60) })}</div>
											{/if}
										</div>
									</button>
									{#if game.createdBy === currentUser?.id}
										<button
											class="absolute top-2 right-2 rounded-lg bg-rose-500/20 hover:bg-rose-500/40 text-rose-300 px-2 py-1 text-xs font-medium transition cursor-pointer"
											onclick={(e) => void handleCancelGameFromList(game, e)}
											title={t('Delete game')}
										>
											✕
										</button>
									{/if}
								</div>
							{/each}
						</div>
					{/if}
					{#if mainMessage}
						<div role="status" aria-live="polite" class="mt-4 rounded-2xl border border-sky-400/20 bg-sky-500/10 px-4 py-3 text-sm text-sky-100">
							{mainMessage}
						</div>
					{/if}
				</div>
			</section>
		{:else if screen === 'create'}
			<section class="grid gap-3 sm:gap-4 md:gap-6 lg:grid-cols-[0.95fr_1.05fr] flex-1">
				<div class="rounded-xl sm:rounded-2xl lg:rounded-[1.75rem] border border-white/10 bg-[var(--card-bg)] p-4 sm:p-5 md:p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					<div class="text-xs font-semibold uppercase tracking-[0.3em] text-amber-300">{t('New game')}</div>
					<h2 class="mt-3 text-2xl font-semibold text-white">{t('Pick your seat')}</h2>
					<p class="mt-3 text-sm leading-6 text-slate-300">
						{t('Choose a team and spot index, then create the game.')}
					</p>

					<div class="mt-6 space-y-4">
						<div>
							<div class="mb-2 text-sm font-medium text-slate-300">{t('Team')}</div>
							<div class="grid grid-cols-2 gap-2">
								{#each teams as team}
									<button
										class={`rounded-2xl border px-4 py-3 text-sm font-semibold transition ${
											createTeam === team.value
												? 'border-amber-300 bg-amber-400 text-slate-950'
												: 'border-white/10 bg-slate-950/50 text-slate-200 hover:border-white/20'
										}`}
										onclick={() => setCreateTeam(team.value)}
									>
										{team.label}
									</button>
								{/each}
							</div>
						</div>

						<div>
							<div class="mb-2 text-sm font-medium text-slate-300">{t('Spot index')}</div>
							<div class="grid grid-cols-2 gap-2">
								{#each seats as seat}
										<button
											class={`rounded-2xl border px-4 py-3 text-sm font-semibold transition ${
												createSpot === seat
													? 'border-sky-300 bg-sky-400 text-slate-950'
													: 'border-white/10 bg-slate-950/50 text-slate-200 hover:border-white/20'
											} ${!isSeatAllowedForTeam(seat, createTeam) ? 'cursor-not-allowed opacity-40' : ''}`}
											disabled={!isSeatAllowedForTeam(seat, createTeam)}
											onclick={() => (createSpot = seat)}
										>
											{seatLabel(seat)}
										</button>
									{/each}
								</div>
							</div>

						{#if createMessage}
							<div role="status" aria-live="polite" class="rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
								{createMessage}
							</div>
						{/if}

						<div class="flex gap-3">
							<button
								class="flex-1 rounded-2xl bg-amber-400 px-4 py-3 font-semibold text-slate-950 transition hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-60"
								disabled={createLoading}
								onclick={() => void createLobbyGame()}
							>
								{createLoading ? t('Creating...') : t('Create game')}
							</button>
						</div>
					</div>
				</div>
			</section>
		{:else if screen === 'join'}
			<section class="grid gap-3 sm:gap-4 md:gap-6 lg:grid-cols-[0.95fr_1.05fr] flex-1">
				<div class="rounded-xl sm:rounded-2xl lg:rounded-[1.75rem] border border-white/10 bg-[var(--card-bg)] p-4 sm:p-5 md:p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					<div class="text-xs font-semibold uppercase tracking-[0.3em] text-amber-300">{t('Join game')}</div>
					<h2 class="mt-3 text-2xl font-semibold text-white">{t('Enter the game 4-character code')}</h2>

					<div class="mt-6 space-y-4">
						<label class="block">
							<span class="mb-2 block text-sm font-medium text-slate-300">{t('Game code')}</span>
							<input
								value={joinCode}
								oninput={joinCodeInput}
								maxlength="4"
								autocomplete="off"
								class="w-full rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-center text-2xl font-semibold tracking-[0.45em] uppercase text-white outline-none transition placeholder:text-slate-500 focus:border-sky-400/60 focus:bg-slate-900"
								placeholder={t('Game code')}
							/>
						</label>

						{#if joinMessage}
							<div role="status" aria-live="polite" class="rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
								{joinMessage}
							</div>
						{/if}

						{#if joinPreview}
							<div class="rounded-3xl border border-white/10 bg-slate-950/40 p-5">
								<div class="flex flex-wrap items-center justify-between gap-3">
									<div>
										<div class="text-sm text-slate-400">{t('Lobby preview')}</div>
										<div class="mt-1 text-3xl font-semibold tracking-[0.35em] text-white">{joinPreview.id}</div>
									</div>
									<div class="rounded-full border border-sky-400/30 bg-sky-500/10 px-3 py-1 text-xs font-medium text-sky-200">
										{joinPreview.status}
									</div>
								</div>

								<div class="mt-4 text-sm text-slate-300">
									{t('Human Players: {count}/4', { count: joinPreview.humanPlayers.length })}
								</div>

								<div class="mt-5">
									<div class="mb-2 text-sm font-medium text-slate-300">{t('Team')}</div>
									<div class="grid grid-cols-2 gap-2">
										{#each teams as team}
											<button
												class={`rounded-2xl border px-4 py-3 text-sm font-semibold transition ${
													joinTeam === team.value
														? 'border-amber-300 bg-amber-400 text-slate-950'
														: 'border-white/10 bg-slate-950/50 text-slate-200 hover:border-white/20'
												}`}
										onclick={() => setJoinTeam(team.value)}
											>
												{team.label}
											</button>
										{/each}
									</div>
								</div>

								<div class="mt-5">
									<div class="mb-2 text-sm font-medium text-slate-300">{t('Spot index')}</div>
									<div class="grid grid-cols-2 gap-2">
										{#each seats as seat}
											<button
												class={`rounded-2xl border px-4 py-3 text-sm font-semibold transition ${
													joinSpot === seat
														? 'border-sky-300 bg-sky-400 text-slate-950'
														: 'border-white/10 bg-slate-950/50 text-slate-200 hover:border-white/20'
												} ${isOccupied(joinPreview, seat) ? 'cursor-not-allowed opacity-40' : ''}`}
												disabled={isOccupied(joinPreview, seat) || !isSeatAllowedForTeam(seat, joinTeam)}
												onclick={() => (joinSpot = seat)}
											>
												{seatLabel(seat)}
											</button>
										{/each}
									</div>
								</div>

								<button
									class="mt-5 w-full rounded-2xl bg-amber-400 px-4 py-3 font-semibold text-slate-950 transition hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-60"
									disabled={joinLoading}
									onclick={() => void enterGame()}
								>
									{joinLoading ? t('Entering...') : t('Enter game')}
								</button>
							</div>
						{:else}
							<div class="rounded-3xl border border-dashed border-white/15 bg-slate-950/30 p-5 text-sm text-slate-400">
								{t('Type the game code to enter its lobby.')}
							</div>
						{/if}

						<div class="flex gap-3">
						</div>
					</div>
				</div>

			</section>
		{:else if screen === 'game'}
			<section class="flex flex-1 flex-col gap-3 sm:gap-4 md:gap-6">
				<!-- `min-w-0`: a flex child defaults to `min-width: auto`, so without it the board's
				     content can size this column wider than the screen. -->
				<div class="min-w-0 rounded-xl sm:rounded-2xl lg:rounded-[1.75rem] border border-white/10 bg-[var(--card-bg)] p-4 sm:p-5 md:p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					{#if activeGame}
						<GameBoard game={activeGame} {currentUser} onPlayCard={sendGameAction} onSelectCapture={sendCaptureSelection} onLeaveTable={leaveTable} />
					{/if}
					{#if mainMessage}
						<div role="status" aria-live="polite" class="mt-4 rounded-2xl border border-sky-400/20 bg-sky-500/10 px-4 py-3 text-sm text-sky-100">
							{mainMessage}
						</div>
					{/if}
					{#if socketError}
						<div role="alert" aria-live="assertive" class="mt-4 rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
							{socketError}
						</div>
					{/if}
				</div>

			</section>
		{:else if screen === 'lobby'}
			<section class="grid gap-3 sm:gap-4 md:gap-6 lg:grid-cols-[1.05fr_0.95fr] flex-1">
				<div class="rounded-xl sm:rounded-2xl lg:rounded-[1.75rem] border border-white/10 bg-[var(--card-bg)] p-4 sm:p-5 md:p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					<div class="flex flex-wrap items-start justify-between gap-4">
						<div>
							<div class="text-xs font-semibold uppercase tracking-[0.3em] text-amber-300">{t('Lobby')}</div>
							<h2 class="mt-3 text-2xl font-semibold text-white">{t('Game {id}', { id: activeGame?.id ?? '' })}</h2>
							<p class="mt-3 text-sm leading-6 text-slate-300">
								{t('Players can join until the creator starts the game.')}
							</p>
						</div>
						<div class="rounded-2xl border border-white/10 bg-slate-950/50 px-4 py-3 text-sm text-slate-300">
							<div class="text-xs uppercase tracking-[0.25em] text-slate-500">{t('Status')}</div>
							<div class="mt-1 font-medium text-white">{activeGame ? formatGameStatus(activeGame) : t('Unknown')}</div>
						</div>
					</div>

					{#if activeGame}
						<div class="mt-6 rounded-3xl border border-white/10 bg-slate-950/40 p-5">
							<div class="flex flex-wrap items-center justify-between gap-3">
								<div>
									<div class="text-sm text-slate-400">{t('Game code')}</div>
									<div class="mt-1 text-4xl font-semibold tracking-[0.45em] text-white">{activeGame.id}</div>
								</div>
								<div class="text-sm text-slate-400">
									{t('Created by {who}', { who: activeGame.createdBy === currentUser?.id ? t('you') : activeGame.createdBy })}
								</div>
							</div>

							<div class="mt-6 grid gap-3 sm:grid-cols-2">
								{#each seats as seat}
									<div class={`rounded-2xl border p-4 ${isOccupied(activeGame, seat) ? 'border-emerald-400/25 bg-emerald-500/10' : 'border-white/10 bg-slate-950/30'}`}>
										<div class="text-xs uppercase tracking-[0.25em] text-slate-500">{seatLabel(seat)}</div>
										<div class="mt-2 text-sm font-medium text-white">
											{#if playerForSeat(activeGame, seat)}
												{playerSummaryForSeat(activeGame, seat)}
											{:else}
												{t('Open')}
											{/if}
										</div>
									</div>
								{/each}
							</div>

							<div class="mt-6 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-slate-300">
								{t('Human Players: {count}/4', { count: activeGame.humanPlayers.length })}
							</div>
						</div>

						{#if mainMessage}
							<div class="mt-4 rounded-2xl border border-sky-400/20 bg-sky-500/10 px-4 py-3 text-sm text-sky-100">
								{mainMessage}
							</div>
						{/if}
						{#if socketError}
							<div role="alert" aria-live="assertive" class="mt-4 rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
								{socketError}
							</div>
						{/if}
					{/if}
				</div>

				<div class="rounded-xl sm:rounded-2xl lg:rounded-[1.75rem] border border-white/10 bg-white/5 p-4 sm:p-5 md:p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					<div class="mt-3 sm:mt-4 grid gap-2 sm:gap-3">
						<button
							class="rounded-2xl bg-amber-400 px-4 py-3 font-semibold text-slate-950 transition hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-60"
							disabled={startLoading || (activeGame?.status !== 'running' && activeGame?.status !== 'finished' && activeGame?.createdBy !== currentUser?.id)}
							onclick={() => {
								if (activeGame?.status === 'running' || activeGame?.status === 'finished') {
									screen = 'game';
								} else {
									void beginGame();
								}
							}}
						>
							{startLoading
						? t('Starting...')
						: activeGame?.status === 'running' || activeGame?.status === 'finished'
							? t('Show table')
							: t('Start game')}
						</button>
						{#if activeGame?.createdBy === currentUser?.id}
							<button
								class="rounded-2xl bg-black hover:bg-rose-500 px-4 py-3 font-semibold text-white transition disabled:cursor-not-allowed disabled:opacity-60"
								disabled={cancelLoading}
								onclick={() => handleCancelGame()}
							>
								{cancelLoading ? t('Canceling...') : t('Cancel game')}
							</button>
						{/if}
						<button
							class="rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-3 font-semibold text-slate-200 transition hover:bg-white/5"
							onclick={() => signOut()}
						>
							{t('Sign out')}
						</button>
					</div>
				</div>
			</section>
		{/if}
	</div>
</div>
