<script lang="ts">
	import { browser } from '$app/environment';
	import { onDestroy, onMount } from 'svelte';
	import {
		createGame,
		getGame,
		joinGame,
		openGameSocket,
		signIn,
		signUp,
		startGame,
		type AuthResponse,
		type Game,
		type HumanPlayer,
		type PublicUser,
		type WsMessage
	} from '$lib/api';
	import GameBoard from '$lib/GameBoard.svelte';

	type AuthTab = 'signin' | 'signup';
	type Screen = 'auth' | 'menu' | 'create' | 'join' | 'lobby' | 'game';

	const sessionKey = 'cuarenta-session';
	const lastGameKey = 'cuarenta-last-game';
	const seats = [1, 2, 3, 4];
	const teams = [
		{ value: 1, label: 'Team 1' },
		{ value: 2, label: 'Team 2' }
	];

	let authTab = $state<AuthTab>('signin');
	let screen = $state<Screen>('auth');
	let email = $state('');
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
	let mainMessage = $state('');
	let lastCardPlayed = $state(-1);

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

	function subscribeToGame(gameId: string) {
		pendingGameId = gameId;
		if (!token) return;

		if (!socket || socket.readyState === WebSocket.CLOSED || socket.readyState === WebSocket.CLOSING) {
			openSocket();
			return;
		}

		if (socket.readyState === WebSocket.OPEN) {
			socket.send(JSON.stringify({ type: 'game.subscribe', payload: { gameId } }));
		}
	}

	function openSocket() {
		if (!token) return;

		closeSocket();
		socketStatus = 'connecting';

		const ws = openGameSocket(token);
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
				activeGame = payload.game;
				if (payload.game.status === 'running') {
					screen = 'game';
					mainMessage = 'Game in progress!';
				} else {
					screen = 'lobby';
					mainMessage = `Connected to game ${payload.game.id}`;
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
			activeGame = game;
			screen = 'lobby';
			pendingGameId = game.id;
			mainMessage = `Loaded game ${game.id}`;
			persistSession();
		} catch {
			activeGame = null;
			pendingGameId = '';
			persistSession();
		}
	}

	async function submitAuth() {
		authLoading = true;
		authError = '';

		try {
			const result: AuthResponse =
				authTab === 'signin' ? await signIn(email, password) : await signUp(email, password);
			token = result.accessToken;
			currentUser = result.user;
			screen = 'menu';
			setNotice(authTab === 'signin' ? 'Signed in' : 'Account created');
			persistSession();
			openSocket();
			if (pendingGameId) {
				subscribeToGame(pendingGameId);
			}
		} catch (error) {
			authError = error instanceof Error ? error.message : 'Authentication failed';
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
		setNotice(`Opened lobby ${activeGame.id}`);
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
			setNotice(`Game ${game.id} created`);
			persistSession();
			subscribeToGame(game.id);
		} catch (error) {
			createMessage = error instanceof Error ? error.message : 'Could not create game';
		} finally {
			createLoading = false;
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
		} catch (error) {
			joinPreview = null;
			joinPreviewCode = code;
			joinMessage = error instanceof Error ? error.message : 'Could not load game';
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
			setNotice(`Joined game ${game.id}`);
			persistSession();
			subscribeToGame(game.id);
		} catch (error) {
			joinMessage = error instanceof Error ? error.message : 'Could not join game';
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
			setNotice(`Game ${game.id} started`);
			persistSession();
			subscribeToGame(game.id);
		} catch (error) {
			mainMessage = error instanceof Error ? error.message : 'Could not start game';
		} finally {
			startLoading = false;
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

	function returnToLobby() {
		screen = 'lobby';
		mainMessage = '';
	}

	function isOccupied(game: Game | null, index: number) {
		return Boolean(game?.humanPlayers.some((player) => player.index === index));
	}

	function seatLabel(index: number) {
		return `Seat ${index}`;
	}

	function playerSummary(player: HumanPlayer) {
		return `${seatLabel(player.index)} · Team ${player.team}`;
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
		if (!isSeatAllowedForTeam(joinSpot, team)) {
			joinSpot = allowedSeatsForTeam(team)[0];
		}
	}

	function formatGameStatus(game: Game) {
		return game.status === 'running' ? 'In progress' : 'Lobby';
	}

	function joinCodeInput(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		joinCode = input.value.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 4);
		void refreshJoinPreview(joinCode);
	}

	onMount(() => {
		loadStoredSession();
	});

	onDestroy(() => {
		closeSocket();
	});
</script>

<svelte:head>
	<title>Cuarenta Multiplayer</title>
</svelte:head>

<div class="min-h-screen px-4 py-8 text-slate-100 sm:px-6 lg:px-8">
	<div class="mx-auto flex w-full max-w-6xl flex-col gap-6">
		<header class="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
			<div class="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
				<div class="max-w-2xl">
					<p class="text-xs font-semibold uppercase tracking-[0.3em] text-amber-300">Cuarenta Multiplayer</p>
				</div>
			</div>
		</header>

		{#if screen === 'auth'}
			<section class="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
				<div class="rounded-[1.75rem] border border-white/10 bg-[var(--card-bg)] p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
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
							Sign in
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
							Sign up
						</button>
					</div>

					<div class="mt-6 space-y-4">
						<label class="block">
							<span class="mb-2 block text-sm font-medium text-slate-300">Email</span>
							<input
								bind:value={email}
								autocomplete="email"
								type="email"
								class="w-full rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-white outline-none ring-0 transition placeholder:text-slate-500 focus:border-amber-400/60 focus:bg-slate-900"
								placeholder="you@example.com"
							/>
						</label>

						<label class="block">
							<span class="mb-2 block text-sm font-medium text-slate-300">Password</span>
							<input
								bind:value={password}
								autocomplete={authTab === 'signin' ? 'current-password' : 'new-password'}
								type="password"
								class="w-full rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-white outline-none ring-0 transition placeholder:text-slate-500 focus:border-amber-400/60 focus:bg-slate-900"
								placeholder="••••••••"
							/>
						</label>

						{#if authError}
							<div class="rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
								{authError}
							</div>
						{/if}

						<button
							class="inline-flex w-full items-center justify-center rounded-2xl bg-amber-400 px-4 py-3 font-semibold text-slate-950 transition hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-60"
							disabled={authLoading}
							onclick={() => void submitAuth()}
						>
							{authLoading ? 'Working...' : authTab === 'signin' ? 'Sign in' : 'Create account'}
						</button>
					</div>
				</div>
			</section>
		{:else if screen === 'menu'}
			<section class="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
				<div class="rounded-[1.75rem] border border-white/10 bg-[var(--card-bg)] p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					<div class="text-xs font-semibold uppercase tracking-[0.3em] text-amber-300">Menu</div>
					<p class="mt-3 text-sm leading-6 text-slate-300">
						You are signed in as {currentUser?.email}. Create a new game or join an existing game.
					</p>

					<div class="mt-6 grid gap-3">
						<button
							class="rounded-2xl bg-amber-400 px-4 py-3 font-semibold text-slate-950 transition hover:bg-amber-300"
							onclick={() => {
								screen = 'create';
								mainMessage = '';
								createMessage = '';
							}}
						>
							New game
						</button>
						<button
							class="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 font-semibold text-white transition hover:border-sky-400/60 hover:bg-sky-400/10"
							onclick={() => {
								screen = 'join';
								mainMessage = '';
								joinMessage = '';
							}}
						>
							Join game
						</button>
						<button
							class="rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-3 font-semibold text-slate-200 transition hover:bg-white/5"
							onclick={() => signOut()}
						>
							Sign out
						</button>
					</div>
				</div>

				<div class="rounded-[1.75rem] border border-white/10 bg-white/5 p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					<div class="text-xs font-semibold uppercase tracking-[0.3em] text-sky-300">Current session</div>
					{#if activeGame}
						<button
							class="mt-4 w-full rounded-3xl border border-white/10 bg-slate-950/40 p-5 text-left transition hover:border-sky-400/50 hover:bg-slate-950/60"
							onclick={() => openCurrentSession()}
						>
							<div class="flex flex-wrap items-center justify-between gap-2">
								<div>
									<div class="text-sm text-slate-400">Game code</div>
									<div class="mt-1 text-3xl font-semibold tracking-[0.35em] text-white">{activeGame.id}</div>
								</div>
								<div class="rounded-full border border-emerald-400/30 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-200">
									{formatGameStatus(activeGame)}
								</div>
							</div>
							<div class="mt-4 text-sm text-slate-300">
								Human Players: {activeGame.humanPlayers.length}/4
							</div>
						</button>
					{:else}
						<div class="mt-4 rounded-3xl border border-dashed border-white/15 bg-slate-950/30 p-5 text-sm text-slate-400">
							No game is active yet.
						</div>
					{/if}
					{#if mainMessage}
						<div class="mt-4 rounded-2xl border border-sky-400/20 bg-sky-500/10 px-4 py-3 text-sm text-sky-100">
							{mainMessage}
						</div>
					{/if}
				</div>
			</section>
		{:else if screen === 'create'}
			<section class="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
				<div class="rounded-[1.75rem] border border-white/10 bg-[var(--card-bg)] p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					<div class="text-xs font-semibold uppercase tracking-[0.3em] text-amber-300">New game</div>
					<h2 class="mt-3 text-2xl font-semibold text-white">Pick your seat</h2>
					<p class="mt-3 text-sm leading-6 text-slate-300">
						Choose a team and spot index, then create the game.
					</p>

					<div class="mt-6 space-y-4">
						<div>
							<div class="mb-2 text-sm font-medium text-slate-300">Team</div>
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
							<div class="mb-2 text-sm font-medium text-slate-300">Spot index</div>
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
							<div class="rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
								{createMessage}
							</div>
						{/if}

						<div class="flex gap-3">
							<button
								class="flex-1 rounded-2xl bg-amber-400 px-4 py-3 font-semibold text-slate-950 transition hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-60"
								disabled={createLoading}
								onclick={() => void createLobbyGame()}
							>
								{createLoading ? 'Creating...' : 'Create game'}
							</button>
							<button
								class="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 font-semibold text-white transition hover:bg-white/10"
								onclick={() => (screen = 'menu')}
							>
								Back
							</button>
						</div>
					</div>
				</div>
			</section>
		{:else if screen === 'join'}
			<section class="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
				<div class="rounded-[1.75rem] border border-white/10 bg-[var(--card-bg)] p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					<div class="text-xs font-semibold uppercase tracking-[0.3em] text-amber-300">Join game</div>
					<h2 class="mt-3 text-2xl font-semibold text-white">Enter the game 4-character code</h2>

					<div class="mt-6 space-y-4">
						<label class="block">
							<span class="mb-2 block text-sm font-medium text-slate-300">Game code</span>
							<input
								value={joinCode}
								oninput={joinCodeInput}
								maxlength="4"
								autocomplete="off"
								class="w-full rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-center text-2xl font-semibold tracking-[0.45em] uppercase text-white outline-none transition placeholder:text-slate-500 focus:border-sky-400/60 focus:bg-slate-900"
								placeholder="AB12"
							/>
						</label>

						{#if joinMessage}
							<div class="rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
								{joinMessage}
							</div>
						{/if}

						{#if joinPreview}
							<div class="rounded-3xl border border-white/10 bg-slate-950/40 p-5">
								<div class="flex flex-wrap items-center justify-between gap-3">
									<div>
										<div class="text-sm text-slate-400">Lobby preview</div>
										<div class="mt-1 text-3xl font-semibold tracking-[0.35em] text-white">{joinPreview.id}</div>
									</div>
									<div class="rounded-full border border-sky-400/30 bg-sky-500/10 px-3 py-1 text-xs font-medium text-sky-200">
										{joinPreview.status}
									</div>
								</div>

								<div class="mt-4 text-sm text-slate-300">
									Human Players: {joinPreview.humanPlayers.length}/4
								</div>

								<div class="mt-5">
									<div class="mb-2 text-sm font-medium text-slate-300">Team</div>
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
									<div class="mb-2 text-sm font-medium text-slate-300">Spot index</div>
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
									{joinLoading ? 'Entering...' : 'Enter game'}
								</button>
							</div>
						{:else}
							<div class="rounded-3xl border border-dashed border-white/15 bg-slate-950/30 p-5 text-sm text-slate-400">
								Type the game code to enter its lobby.
							</div>
						{/if}

						<div class="flex gap-3">
							<button
								class="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 font-semibold text-white transition hover:bg-white/10"
								onclick={() => (screen = 'menu')}
							>
								Back
							</button>
						</div>
					</div>
				</div>

			</section>
		{:else if screen === 'game'}
			<section class="grid gap-6 lg:grid-cols-[1fr_120px]">
				<div class="rounded-[1.75rem] border border-white/10 bg-[var(--card-bg)] p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					{#if activeGame}
						<GameBoard game={activeGame} {currentUser} onPlayCard={sendGameAction} onSelectCapture={sendCaptureSelection} />
					{/if}
					{#if mainMessage}
						<div class="mt-4 rounded-2xl border border-sky-400/20 bg-sky-500/10 px-4 py-3 text-sm text-sky-100">
							{mainMessage}
						</div>
					{/if}
					{#if socketError}
						<div class="mt-4 rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
							{socketError}
						</div>
					{/if}
				</div>

				<div class="rounded-[1.75rem] border border-white/10 bg-white/5 p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					<div class="mt-4 grid gap-3">
						<button
							class="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 font-semibold text-white transition hover:bg-white/10"
							onclick={() => returnToLobby()}
						>
							Back
						</button>
						<button
							class="rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-3 font-semibold text-slate-200 transition hover:bg-white/5"
							onclick={() => signOut()}
						>
							Sign out
						</button>
					</div>
				</div>
			</section>
		{:else if screen === 'lobby'}
			<section class="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
				<div class="rounded-[1.75rem] border border-white/10 bg-[var(--card-bg)] p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					<div class="flex flex-wrap items-start justify-between gap-4">
						<div>
							<div class="text-xs font-semibold uppercase tracking-[0.3em] text-amber-300">Lobby</div>
							<h2 class="mt-3 text-2xl font-semibold text-white">Game {activeGame?.id}</h2>
							<p class="mt-3 text-sm leading-6 text-slate-300">
								Players can join until the creator starts the game.
							</p>
						</div>
						<div class="rounded-2xl border border-white/10 bg-slate-950/50 px-4 py-3 text-sm text-slate-300">
							<div class="text-xs uppercase tracking-[0.25em] text-slate-500">Status</div>
							<div class="mt-1 font-medium text-white">{activeGame ? formatGameStatus(activeGame) : 'Unknown'}</div>
						</div>
					</div>

					{#if activeGame}
						<div class="mt-6 rounded-3xl border border-white/10 bg-slate-950/40 p-5">
							<div class="flex flex-wrap items-center justify-between gap-3">
								<div>
									<div class="text-sm text-slate-400">Game code</div>
									<div class="mt-1 text-4xl font-semibold tracking-[0.45em] text-white">{activeGame.id}</div>
								</div>
								<div class="text-sm text-slate-400">
									Created by {activeGame.createdBy === currentUser?.id ? 'you' : activeGame.createdBy}
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
												Open
											{/if}
										</div>
									</div>
								{/each}
							</div>

							<div class="mt-6 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-slate-300">
								Human Players: {activeGame.humanPlayers.length}/4
							</div>
						</div>

						{#if mainMessage}
							<div class="mt-4 rounded-2xl border border-sky-400/20 bg-sky-500/10 px-4 py-3 text-sm text-sky-100">
								{mainMessage}
							</div>
						{/if}
						{#if socketError}
							<div class="mt-4 rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
								{socketError}
							</div>
						{/if}
					{/if}
				</div>

				<div class="rounded-[1.75rem] border border-white/10 bg-white/5 p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
					<div class="mt-4 grid gap-3">
						<button
							class="rounded-2xl bg-amber-400 px-4 py-3 font-semibold text-slate-950 transition hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-60"
							disabled={startLoading || (activeGame?.status !== 'running' && activeGame?.createdBy !== currentUser?.id)}
							onclick={() => {
								if (activeGame?.status === 'running') {
									screen = 'game';
								} else {
									void beginGame();
								}
							}}
						>
							{startLoading ? 'Starting...' : activeGame?.status === 'running' ? 'Show table' : 'Start game'}
						</button>
						<button
							class="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 font-semibold text-white transition hover:bg-white/10"
							onclick={() => {
								screen = 'menu';
								mainMessage = '';
							}}
						>
							Back to menu
						</button>
						<button
							class="rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-3 font-semibold text-slate-200 transition hover:bg-white/5"
							onclick={() => signOut()}
						>
							Sign out
						</button>
					</div>
				</div>
			</section>
		{/if}
	</div>
</div>
