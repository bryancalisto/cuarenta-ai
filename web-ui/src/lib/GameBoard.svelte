<script lang="ts">
	import type { Game } from './api';
	import GameResultsDialog from './GameResultsDialog.svelte';

	let { game, currentUser, onPlayCard, onSelectCapture, onReturnToLobby } = $props<{
		game: Game;
		currentUser: { id: string; email: string } | null;
		onPlayCard: (cardIndex: number) => void;
		onSelectCapture: (captureIndices: number[]) => void;
		onReturnToLobby: () => void;
	}>();

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

	function canPlayCard(cardIndex: number): boolean {
		return game.currentPlayer === myArrayIndex;
	}

	function getPlayerLabel(index: number): string {
		return `P${index + 1}`;
	}

	const myPlayer = $derived(game.humanPlayers.find((p: any) => currentUser && p.userId === currentUser.id));
	const myArrayIndex = $derived(myPlayer ? myPlayer.index - 1 : undefined);

	const isMyTurn = $derived(myArrayIndex !== undefined && game.currentPlayer === myArrayIndex);

	let displayedMoveCount = $state(0);
	let isSelectingCapture = $state(false);

	// Reset capture selection when a new capture prompt appears
	$effect(() => {
		if (game.awaitingCapture) {
			isSelectingCapture = false;
		}
	});

	// Determine if we're still animating pending moves
	const isAnimatingMoves = (): boolean => {
		return game.pendingMoves ? displayedMoveCount < game.pendingMoves.length : false;
	};

	// Get the current player during animation (before pending moves)
	const getDisplayedCurrentPlayer = (): number | undefined => {
		if (!isAnimatingMoves() || !game.pendingMoves) {
			return game.currentPlayer;
		}

		// During animation, show the player before the pending moves started
		// The pending moves start from the player after the displayed moves
		const totalMoves = game.pendingMoves.length;
		const remainingMoves = totalMoves - displayedMoveCount;
		const playerOffset = (remainingMoves % 4);

		if (game.currentPlayer === undefined) return undefined;
		return (game.currentPlayer - playerOffset + 400) % 4; // +400 to avoid negative
	};

	// Calculate the table state progressively as moves are displayed
	const getDisplayedTable = (): number[] => {
		if (!game.pendingMoves || displayedMoveCount === 0) {
			return game.table || [];
		}

		// Reconstruct initial table before pending moves
		const initialTable = new Set<number>(game.table || []);
		for (let i = game.pendingMoves.length - 1; i >= 0; i--) {
			const move = game.pendingMoves[i];
			// If it was a capture, the cardIndex was never left on the table
			if (!move.captureIndices || move.captureIndices.length === 0) {
				initialTable.delete(move.cardIndex);
			}
			if (move.captureIndices) {
				for (const captureIdx of move.captureIndices) {
					initialTable.add(captureIdx);
				}
			}
		}

		// Replay moves up to displayedMoveCount
		const displayedTable = new Set(initialTable);
		for (let i = 0; i < displayedMoveCount && i < game.pendingMoves.length; i++) {
			const move = game.pendingMoves[i];
			if (move.captureIndices && move.captureIndices.length > 0) {
				for (const captureIdx of move.captureIndices) {
					displayedTable.delete(captureIdx);
				}
			} else {
				displayedTable.add(move.cardIndex);
			}
		}

		return Array.from(displayedTable);
	};

	$effect(() => {
		if (!game.pendingMoves) return;

		const totalMoves = game.pendingMoves.length;
		if (displayedMoveCount >= totalMoves) return;

		const timer = setTimeout(() => {
			displayedMoveCount++;
		}, 3000);

		return () => clearTimeout(timer);
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
			{#if getDisplayedTable().length > 0}
				<div class="cards-display">
					{#each getDisplayedTable() as cardIndex, idx}
						{@const displayedTable = getDisplayedTable()}
						<div class="card on-table" class:last-placed={idx === displayedTable.length - 1 && game.lastPlaced !== null}>
							{cardToString(cardIndex)}
						</div>
					{/each}
				</div>
			{:else}
				<div class="empty-table">Table is empty</div>
			{/if}
		</div>
	</div>

	<!-- Capture selection section -->
	{#if game.awaitingCapture && game.captureOptions && game.captureOptions.length > 0 && !isSelectingCapture}
		<div class="capture-selection">
			<div class="section-title">Choose which cards to capture:</div>
			<div class="captures-grid">
				{#each game.captureOptions as option, idx}
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
							class:playable={canPlayCard(cardIndex)}
							disabled={!canPlayCard(cardIndex)}
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
		font-size: 3rem;
		font-weight: 700;
		color: #94a3b8;
		text-align: center;
		transition: all 0.3s ease;
	}

	.current-player-text.your-turn {
		color: #fbbf24;
		font-size: 3.5rem;
		text-shadow: 0 0 20px rgba(251, 191, 36, 0.5);
	}

	.turn-label {
		display: block;
		font-size: 1.5rem;
		margin-top: 0.5rem;
		color: #10b981;
		font-weight: 600;
	}

	.game-over {
		font-size: 3rem;
		font-weight: 700;
		color: #4ade80;
		text-align: center;
	}

	.table-area {
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		border: 2px solid white/20;
		border-radius: 1rem;
		background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(20, 30, 50, 0.8));
		padding: 3rem;
		gap: 1rem;
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
		padding: 1rem 1rem;
		border-radius: 0.5rem;
		border: 2px solid white/30;
		background: white;
		color: black;
		font-weight: 600;
		font-size: 1.5rem;
		cursor: pointer;
		transition: all 0.2s;
	}

	.card.on-table {
		cursor: default;
		pointer-events: none;
	}

	.card.on-table.last-placed {
		background: linear-gradient(135deg, #fbbf24 0%, #fcd34d 100%);
		box-shadow: 0 0 16px rgba(251, 191, 36, 0.5);
		border-color: #f59e0b;
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
