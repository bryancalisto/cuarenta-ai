<script lang="ts">
	import { untrack } from 'svelte';
	import type { Game } from './api';
	import GameResultsDialog from './GameResultsDialog.svelte';

	let { game, currentUser, onPlayCard, onSelectCapture, onReturnToLobby } = $props<{
		game: Game;
		currentUser: { id: string; email: string } | null;
		onPlayCard: (cardIndex: number) => void;
		onSelectCapture: (captureIndices: number[]) => void;
		onReturnToLobby: () => void;
	}>();

	$effect(() => {
		$inspect(game);
	});

	const suits = ['♦️', '♥️', '♠️', '♣️'];
	const ranks = ['A', '2', '3', '4', '5', '6', '7', 'J', 'Q', 'K'];

	function cardToString(cardIndex: number): string {
		const rank = ranks[cardIndex % 10];
		const suit = suits[Math.floor(cardIndex / 10)];
		return `${rank}${suit}`;
	}

	function isHumanPlayer(playerIndex: number): boolean {
		return game.humanPlayers.some((p: any) => p.index - 1 === playerIndex);
	}

	function isCurrentUser(playerIndex: number): boolean {
		return myArrayIndex !== undefined && playerIndex === myArrayIndex;
	}

	function canPlayCard(): boolean {
		return game.currentPlayer === myArrayIndex;
	}

	function getPlayerLabel(index: number): string {
		return `P${index + 1}`;
	}

	const myPlayer = $derived(game.humanPlayers.find((p: any) => currentUser && p.userId === currentUser.id));
	const myArrayIndex = $derived(myPlayer ? myPlayer.index - 1 : undefined);

	const isMyTurn = $derived(myArrayIndex !== undefined && game.currentPlayer === myArrayIndex);

	let isSelectingCapture = $state(false);
	let displayedTable: number[] = $state([]);

	// Reset capture selection when a new capture prompt appears
	$effect(() => {
		if (game.awaitingCapture) {
			isSelectingCapture = false;
		}
	});

	// Get the current player during animation
	const getDisplayedCurrentPlayer = (): number | undefined => {
		return game.currentPlayer
	};

	// Side each player throws from / grabs to, relative to current user (always "down").
	const DIRS = ['down', 'right', 'up', 'left'] as const;
	const OFFSET: Record<string, { tx: string; ty: string }> = {
		down: { tx: '0px', ty: '340px' },
		right: { tx: '560px', ty: '0px' },
		up: { tx: '0px', ty: '-340px' },
		left: { tx: '-560px', ty: '0px' }
	};

	function dirFor(playerIndex: number): string {
		const me = myArrayIndex ?? 0;
		return DIRS[(playerIndex - me + 4) % 4];
	}

	let throwAnim = $state<{ card: number; dir: string; id: number } | null>(null);
	let grabAnim = $state<{ cards: number[]; dir: string; id: number } | null>(null);
	let animId = 0;

	// Animation timings (ms) — must match the CSS animation durations.
	const THROW_MS = 900;
	const GAP_MS = 250;
	const GRAB_MS = 900;

	$effect(() => {
		const table = game.table ?? [];
		const move = game.move;
		const currentPlayer = game.currentPlayer;

		// No move to animate: just mirror the table.
		if (!move || currentPlayer === undefined) {
			displayedTable = table;
			return;
		}

		const before = untrack(() => displayedTable);
		const dir = dirFor((currentPlayer - 1 + 4) % 4);
		const id = ++animId;
		const captures = move.captureIndices ?? [];

		// 1. Throw: keep the old table visible while the played card flies in.
		throwAnim = { card: move.cardIndex, dir, id };
		grabAnim = null;

		const timers: ReturnType<typeof setTimeout>[] = [];

		// 2. Played card lands. Add it to the table; keep captured cards for now.
		timers.push(setTimeout(() => {
			if (animId !== id) return;
			throwAnim = null;
			displayedTable = captures.length > 0 ? [...before, move.cardIndex] : table;
		}, THROW_MS));

		if (captures.length > 0) {
			// 3. Grab: pull the played card + captured cards off the table, laid out
			//    horizontally, toward the player. Remove them from the table at the
			//    same moment so the flying group represents them.
			timers.push(setTimeout(() => {
				if (animId !== id) return;
				grabAnim = { cards: [move.cardIndex, ...captures], dir, id };
				displayedTable = table;
			}, THROW_MS + GAP_MS));
			// 4. Grab done.
			timers.push(setTimeout(() => {
				if (animId === id) grabAnim = null;
			}, THROW_MS + GAP_MS + GRAB_MS));
		}

		return () => timers.forEach(clearTimeout);
	});
</script>

<div class="game-board">
	<div class="board-header">
		<div class="game-info">
			<div class="info-item">
				<span class="label">Game</span>
				<span class="value">{game.id}</span>
			</div>
		</div>

		<div class="scores">
			<div class="score-item">
				<span class="label">Team 1</span>
				<span class="value">{game.scores?.[0] ?? 0}</span>
			</div>
			<div class="score-item">
				<span class="label">Team 2</span>
				<span class="value">{game.scores?.[1] ?? 0}</span>
			</div>
		</div>
	</div>

	<!-- Current player indicator -->
	<div class="current-player-section">
		{#if getDisplayedCurrentPlayer() !== undefined && !game.done}
			{@const currentPlayerIdx = getDisplayedCurrentPlayer() ?? 0}
			<div class="current-player-text" class:your-turn={isCurrentUser(currentPlayerIdx)}>
				{getPlayerLabel(currentPlayerIdx)}
				{#if isCurrentUser(currentPlayerIdx)}
					<span class="turn-label">Your turn</span>
				{:else if isHumanPlayer(currentPlayerIdx)}
					<span class="turn-label">Human</span>
				{:else}
					<span class="turn-label">AI</span>
				{/if}
			</div>
		{/if}
	</div>

	{#if game.done}
		<GameResultsDialog {game} {onReturnToLobby} />
	{/if}

	<!-- Table area -->
	<div class="table-area">
		<div class="table-cards">
			{#if displayedTable.length > 0}
				<div class="cards-display">
					{#each displayedTable as cardIndex}
						<div class="card on-table">
							{cardToString(cardIndex)}
						</div>
					{/each}
				</div>
			{:else}
				<div class="empty-table">Table is empty</div>
			{/if}
		</div>

		<!-- Card throw / grab animations -->
		<div class="anim-overlay">
			{#if throwAnim}
				{#key throwAnim.id}
					<div
						class="fly-card throw"
						style="--tx:{OFFSET[throwAnim.dir].tx}; --ty:{OFFSET[throwAnim.dir].ty}"
					>
						{cardToString(throwAnim.card)}
					</div>
				{/key}
			{/if}
			{#if grabAnim}
				{#key 'g' + grabAnim.id}
					<div class="grab-group">
						{#each grabAnim.cards as cardIndex}
							<div
								class="fly-card grab"
								style="--tx:{OFFSET[grabAnim.dir].tx}; --ty:{OFFSET[grabAnim.dir].ty}"
							>
								{cardToString(cardIndex)}
							</div>
						{/each}
					</div>
				{/key}
			{/if}
		</div>
	</div>

	<!-- Capture selection section -->
	{#if game.awaitingCapture && game.captureOptions && game.captureOptions.length > 0 && !isSelectingCapture}
		<div class="capture-selection">
			<div class="section-title">Choose which cards to capture:</div>
			<div class="captures-grid">
				{#each game.captureOptions as option}
					<button
						class="capture-option"
						onclick={() => {
							isSelectingCapture = true;
							onSelectCapture(option.captureIndices);
						}}
					>
						{#if option.captureIndices && option.captureIndices.length > 0}
							{option.captureIndices.map(cardToString).join(', ')}
						{:else}
							No capture
						{/if}
					</button>
				{/each}
			</div>
		</div>
	{/if}

	<!-- Player's hand at bottom -->
	{#if myArrayIndex !== undefined && game.cards && game.cards[myArrayIndex]}
		<div class="player-hand" class:my-turn={isMyTurn}>
			<div class="hand-header">
				<div class="hand-label">{isMyTurn ? 'Your cards' : 'Your hand'}</div>
			</div>
			<div class="cards-container">
				{#each game.cards[myArrayIndex] as cardIndex}
					{#if cardIndex >= 0}
						<button
							class="card"
							class:playable={canPlayCard()}
							disabled={!canPlayCard()}
							onclick={() => onPlayCard(cardIndex)}
						>
							{cardToString(cardIndex)}
						</button>
					{/if}
				{/each}
			</div>
		</div>
	{/if}
</div>

<style>
	.game-board {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
		height: 100%;
	}

	.board-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		border-bottom: 2px solid white/10;
	}

	.game-info,
	.scores {
		display: flex;
		gap: 2rem;
	}

	.info-item,
	.score-item {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.label {
		font-size: 0.75rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: #cbd5e1;
	}

	.value {
		font-size: 1.5rem;
		font-weight: 600;
		color: white;
		font-family: monospace;
	}

	.current-player-section {
		display: flex;
		justify-content: center;
		align-items: center;
	}

	.current-player-text {
		font-size: clamp(1.5rem, 8vw, 3rem);
		font-weight: 700;
		color: #94a3b8;
		text-align: center;
		transition: all 0.3s ease;
	}

	.current-player-text.your-turn {
		color: #fbbf24;
		font-size: clamp(1.75rem, 9vw, 3.5rem);
		text-shadow: 0 0 20px rgba(251, 191, 36, 0.5);
	}

	.turn-label {
		display: block;
		font-size: clamp(0.875rem, 3vw, 1.5rem);
		margin-top: 0.5rem;
		color: #10b981;
		font-weight: 600;
	}

	.table-area {
		position: relative;
		display: flex;
		flex-direction: column;
		justify-content: center;
		min-height: clamp(6rem, 25vw, 9rem);
		align-items: center;
		border-radius: 1rem;
		background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(20, 30, 50, 0.8));
		padding: clamp(1rem, 4vw, 3rem);
		gap: 1rem;
	}

	.anim-overlay {
		position: absolute;
		inset: 0;
		overflow: visible;
		display: flex;
		align-items: center;
		justify-content: center;
		pointer-events: none;
		z-index: 10;
	}

	.grab-group {
		display: flex;
		flex-wrap: nowrap;
		gap: 0.5rem;
	}

	.fly-card {
		padding: clamp(0.5rem, 2vw, 1rem);
		border-radius: 0.5rem;
		background: white;
		color: black;
		font-weight: 600;
		font-size: clamp(1rem, 4vw, 1.5rem);
		min-width: clamp(2rem, 8vw, 3rem);
		min-height: clamp(2.5rem, 8vw, 3.5rem);
		display: flex;
		align-items: center;
		justify-content: center;
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
		will-change: transform, opacity;
	}

	.fly-card.throw {
		animation: throw-in 0.9s cubic-bezier(0.22, 1, 0.36, 1);
	}

	.fly-card.grab {
		animation: grab-out 0.9s cubic-bezier(0.55, 0, 0.55, 0.2) forwards;
	}

	@keyframes throw-in {
		from {
			transform: translate(var(--tx), var(--ty)) scale(0.6) rotate(8deg);
			opacity: 0;
		}
		to {
			transform: translate(0, 0) scale(1) rotate(0deg);
			opacity: 1;
		}
	}

	@keyframes grab-out {
		from {
			transform: translate(0, 0) scale(1);
			opacity: 1;
		}
		to {
			transform: translate(var(--tx), var(--ty)) scale(0.6);
			opacity: 0;
		}
	}

	.table-cards {
		display: flex;
		justify-content: center;
		align-items: center;
	}

	.cards-display {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
		justify-content: center;
	}

	.card {
		padding: clamp(0.5rem, 2vw, 1rem) clamp(0.5rem, 2vw, 1rem);
		border-radius: 0.5rem;
		border: 2px solid white/30;
		background: white;
		color: black;
		font-weight: 600;
		font-size: clamp(1rem, 4vw, 1.5rem);
		cursor: pointer;
		transition: all 0.2s;
		min-width: clamp(2rem, 8vw, 3rem);
		min-height: clamp(2.5rem, 8vw, 3.5rem);
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.card.on-table {
		cursor: default;
		pointer-events: none;
	}


	.card.playable:hover {
		transform: scale(1.05);
		box-shadow: 0 4px 12px rgba(251, 191, 36, 0.4);
		border-color: #fbbf24;
	}

	.card:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.empty-table {
		color: #94a3b8;
		font-style: italic;
		font-size: 1.125rem;
	}

	.player-hand {
		border: 2px solid white/10;
		border-radius: 1rem;
		padding: 1rem;
		background: rgba(15, 23, 42, 0.6);
		transition: all 0.3s ease;
	}

	.player-hand.my-turn {
		border-color: #fbbf24;
		background: rgba(251, 191, 36, 0.1);
		box-shadow: 0 0 20px rgba(251, 191, 36, 0.2);
	}

	.hand-header {
		margin-bottom: 0.75rem;
		border-bottom: 1px solid white/10;
	}

	.hand-label {
		font-size: 0.875rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: #cbd5e1;
	}

	.player-hand.my-turn .hand-label {
		color: #fbbf24;
	}

	.cards-container {
		display: flex;
		gap: 0.5rem;
		overflow-x: auto;
		padding: 0.5rem 0;
	}

	.section-title {
		font-size: 0.875rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: #cbd5e1;
		margin-bottom: 0.75rem;
	}

	.capture-selection {
		border: 2px solid #fbbf24;
		border-radius: 1rem;
		padding: 1rem;
		background: rgba(251, 191, 36, 0.1);
	}

	.captures-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
		gap: 0.75rem;
	}

	.capture-option {
		padding: 0.75rem;
		border: 2px solid #fbbf24;
		border-radius: 0.5rem;
		background: transparent;
		color: #fbbf24;
		font-weight: 600;
		cursor: pointer;
		transition: all 0.2s;
	}

	.capture-option:hover {
		background: rgba(251, 191, 36, 0.2);
		box-shadow: 0 0 12px rgba(251, 191, 36, 0.3);
	}

</style>
