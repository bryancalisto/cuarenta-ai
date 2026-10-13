<script lang="ts">
	/**
	 * A card face, drawn the way a real card is drawn: PORTRAIT, with the rank over the suit in
	 * two opposite corners (the lower pair upside down, as on a real card) and the suit pips
	 * arranged in the middle in their traditional positions -- which is what lets a 6 be read as
	 * a 6 at a glance instead of counting a row of identical symbols.
	 *
	 * Everything inside scales from the card's own width (`cqw`, container query units), so one
	 * component draws a hand card and a table card: the PARENT decides the size, this decides the
	 * face. Nothing here is sized in `rem`, or the corners would be unreadable on a small card.
	 *
	 * The card index is the shared one: `suit * 10 + rankPosition` (see src/ppo/env.py), the same
	 * mapping the backend and the training code use.
	 */
	import { t } from './i18n.svelte';

	let { cardIndex, class: className = '' } = $props<{ cardIndex: number; class?: string }>();

	const SUIT_GLYPHS = ['♦', '♥', '♠', '♣'];
	const RANKS = ['A', '2', '3', '4', '5', '6', '7', 'J', 'Q', 'K'];

	/**
	 * The suit's name, for the accessible label (a screen reader cannot read a pip layout). The
	 * `t()` calls are written out as literals so `npm run check:i18n` can see the keys: a lookup
	 * table passed to `t()` would leave the translations unverifiable.
	 */
	function suitName(index: number): string {
		switch (index) {
			case 0:
				return t('Diamonds');
			case 1:
				return t('Hearts');
			case 2:
				return t('Spades');
			default:
				return t('Clubs');
		}
	}
	const COURT_RANKS = ['J', 'Q', 'K'];

	/**
	 * Where the pips go, as [x%, y%, upsideDown]. The positions are the classic ones: two columns
	 * for 4-7, a centre column for 1-3, the lower half mirrored so the card reads from whichever
	 * end it was picked up. `A` gets one big pip in the middle.
	 */
	const PIP_LAYOUT: Record<string, [number, number, boolean][]> = {
		A: [[50, 50, false]],
		'2': [
			[50, 18, false],
			[50, 82, true]
		],
		'3': [
			[50, 18, false],
			[50, 50, false],
			[50, 82, true]
		],
		'4': [
			[26, 18, false],
			[74, 18, false],
			[26, 82, true],
			[74, 82, true]
		],
		'5': [
			[26, 18, false],
			[74, 18, false],
			[50, 50, false],
			[26, 82, true],
			[74, 82, true]
		],
		'6': [
			[26, 16, false],
			[74, 16, false],
			[26, 50, false],
			[74, 50, false],
			[26, 84, true],
			[74, 84, true]
		],
		'7': [
			[26, 14, false],
			[74, 14, false],
			[50, 33, false],
			[26, 52, false],
			[74, 52, false],
			[26, 86, true],
			[74, 86, true]
		]
	};

	const suitIndex = $derived(Math.floor(cardIndex / 10));
	const rank = $derived(RANKS[cardIndex % 10] ?? '?');
	const suit = $derived(SUIT_GLYPHS[suitIndex] ?? '?');
	// Diamonds and hearts are red in the deck this UI draws.
	const isRed = $derived(suitIndex <= 1);
	const isCourt = $derived(COURT_RANKS.includes(rank));
	const pips = $derived(PIP_LAYOUT[rank] ?? []);
	const label = $derived(t('{rank} of {suit}', { rank, suit: suitName(suitIndex) }));
</script>

<span
	class="playing-card {isRed ? 'red' : 'black'} {className}"
	role="img"
	aria-label={label}
	data-card={cardIndex}
>
	<!-- Corner index: rank over suit, top-left, and the same turned upside down bottom-right. -->
	<span class="corner top" aria-hidden="true">
		<span class="corner-rank">{rank}</span>
		<span class="corner-suit">{suit}</span>
	</span>
	<span class="corner bottom" aria-hidden="true">
		<span class="corner-rank">{rank}</span>
		<span class="corner-suit">{suit}</span>
	</span>

	{#if isCourt}
		<!-- Court cards carry a figure rather than pips: two halves facing each other, the way a
		     real double-headed court card is drawn. -->
		<span class="court" aria-hidden="true">
			<span class="court-figure">
				<span class="court-rank">{rank}</span>
				<span class="court-suit">{suit}</span>
			</span>
			<span class="court-figure flipped">
				<span class="court-rank">{rank}</span>
				<span class="court-suit">{suit}</span>
			</span>
		</span>
	{:else}
		<span class="pips" aria-hidden="true">
			{#each pips as [x, y, upsideDown] (x + ':' + y)}
				<span
					class="pip"
					class:flipped={upsideDown}
					class:big={pips.length === 1}
					style="left:{x}%; top:{y}%"
					>{suit}</span
				>
			{/each}
		</span>
	{/if}
</span>

<style>
	.playing-card {
		position: relative;
		display: block;
		width: 100%;
		height: 100%;
		/* Percentages inside are relative to THIS box, and `cqw` lets the corner and pip sizes
		   follow the card's width instead of the page's. */
		container-type: size;
		border-radius: 0.35rem;
		background: linear-gradient(160deg, #fffdf7, #f3efe4);
		box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.25);
		overflow: hidden;
		font-family: 'Helvetica Neue', Arial, sans-serif;
		font-weight: 700;
		user-select: none;
	}

	.red {
		color: #be123c;
	}

	.black {
		color: #0f172a;
	}

	/* ---------------- corner indices ---------------- */

	.corner {
		position: absolute;
		display: flex;
		flex-direction: column;
		align-items: center;
		line-height: 0.95;
		font-size: 23cqw;
	}

	.corner.top {
		left: 7cqw;
		top: 5cqw;
	}

	/* The lower index of a real card is the same mark upside down: hold the card either way and
	   one of the two is the right way up. */
	.corner.bottom {
		right: 7cqw;
		bottom: 5cqw;
		transform: rotate(180deg);
	}

	.corner-suit {
		font-size: 19cqw;
	}

	/* ---------------- pips ---------------- */

	.pips {
		position: absolute;
		/* Inset so no pip collides with a corner index. */
		inset: 12cqw 20cqw;
	}

	.pip {
		position: absolute;
		transform: translate(-50%, -50%);
		font-size: 19cqw;
		line-height: 1;
	}

	.pip.flipped {
		transform: translate(-50%, -50%) rotate(180deg);
	}

	.pip.big {
		font-size: 34cqw;
	}

	/* ---------------- court cards ---------------- */

	.court {
		position: absolute;
		inset: 11cqw 18cqw;
		display: flex;
		flex-direction: column;
		justify-content: space-between;
		border: 1px solid currentColor;
		border-radius: 0.3rem;
		background:
			repeating-linear-gradient(45deg, rgba(15, 23, 42, 0.05) 0 2px, transparent 2px 4px),
			linear-gradient(160deg, rgba(255, 255, 255, 0.6), rgba(15, 23, 42, 0.06));
		overflow: hidden;
	}

	.court-figure {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		line-height: 0.9;
		flex: 1;
	}

	.court-figure.flipped {
		transform: rotate(180deg);
	}

	.court-rank {
		font-size: 26cqw;
	}

	.court-suit {
		font-size: 16cqw;
	}
</style>
