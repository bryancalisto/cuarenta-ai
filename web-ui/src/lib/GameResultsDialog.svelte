<script lang="ts">
	import { onMount } from 'svelte';
	import type { Game } from './api';
	import { t } from './i18n.svelte';

	let { game, onLeaveTable } = $props<{
		game: Game;
		onLeaveTable: () => void;
	}>();

	let duration = $state('');

	function formatDuration(durationSeconds: number) {
		const minutes = Math.floor(durationSeconds / 60);
		const seconds = Math.floor((durationSeconds % 60));

		if (minutes > 0) {
			return `${minutes}m ${seconds}s`;
		}
		return `${seconds}s`;
	}

	onMount(() => {
		if (game.durationSeconds) {
			duration = formatDuration(game.durationSeconds);
		}
	});

	const winnerTeam = $derived(game.winner !== undefined ? game.winner + 1 : null);
</script>

<div class="dialog-overlay">
	<div class="dialog-content">
		<div class="results-header">
			<div class="winner-label">{t('Winner')}</div>
			<div class="winner-team">{t('Team {team}', { team: winnerTeam ?? '' })}</div>
		</div>

		<div class="scores-grid">
			<div class="score-card">
				<div class="team-label">{t('Team 1')}</div>
				<div class="team-score">{game.scores?.[0] ?? 0}</div>
			</div>
			<div class="score-card">
				<div class="team-label">{t('Team 2')}</div>
				<div class="team-score">{game.scores?.[1] ?? 0}</div>
			</div>
		</div>

		<div class="game-meta">
			<div class="meta-item">
				<span class="meta-label">{t('Time played:')}</span>
				<span class="meta-value">{duration}</span>
			</div>
		</div>

		<!-- The game is over, so there is no lobby to return to: the button leaves the table. -->
		<button class="return-button" onclick={onLeaveTable}>
			{t('Back to menu')}
		</button>
	</div>
</div>

<style>
	.dialog-overlay {
		position: fixed;
		inset: 0;
		background: rgba(15, 23, 42, 0.9);
		backdrop-filter: blur(8px);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 1000;
	}

	.dialog-content {
		background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
		border: 2px solid #fbbf24;
		border-radius: 2rem;
		padding: 3rem;
		width: 100%;
		max-width: 500px;
		text-align: center;
		box-shadow: 0 0 50px rgba(251, 191, 36, 0.2);
		animation: dialogAppear 0.4s cubic-bezier(0.16, 1, 0.3, 1);
	}

	@keyframes dialogAppear {
		from {
			opacity: 0;
			transform: scale(0.9) translateY(20px);
		}
		to {
			opacity: 1;
			transform: scale(1) translateY(0);
		}
	}

	.results-header {
		margin-bottom: 2.5rem;
	}

	.winner-label {
		font-size: 0.875rem;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.2em;
		color: #fbbf24;
		margin-bottom: 0.5rem;
	}

	.winner-team {
		font-size: 4rem;
		font-weight: 800;
		color: white;
		text-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
	}

	.scores-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1.5rem;
		margin-bottom: 2.5rem;
	}

	.score-card {
		background: rgba(255, 255, 255, 0.05);
		border: 1px solid rgba(255, 255, 255, 0.1);
		border-radius: 1.5rem;
		padding: 1.5rem;
	}

	.team-label {
		font-size: 0.75rem;
		font-weight: 600;
		text-transform: uppercase;
		color: #94a3b8;
		margin-bottom: 0.5rem;
	}

	.team-score {
		font-size: 2.5rem;
		font-weight: 700;
		color: white;
	}

	.game-meta {
		padding-top: 1.5rem;
		border-top: 1px solid rgba(255, 255, 255, 0.1);
		margin-bottom: 2.5rem;
	}

	.meta-item {
		display: flex;
		justify-content: center;
		gap: 0.75rem;
		font-size: 1rem;
	}

	.meta-label {
		color: #94a3b8;
	}

	.meta-value {
		color: white;
		font-weight: 600;
	}

	.return-button {
		width: 100%;
		background: #fbbf24;
		color: #0f172a;
		border: none;
		border-radius: 1rem;
		padding: 1.25rem;
		font-size: 1.125rem;
		font-weight: 700;
		cursor: pointer;
		transition: all 0.2s;
	}

	.return-button:hover {
		background: #fcd34d;
		transform: translateY(-2px);
		box-shadow: 0 10px 20px rgba(251, 191, 36, 0.2);
	}

	.return-button:active {
		transform: translateY(0);
	}
</style>
