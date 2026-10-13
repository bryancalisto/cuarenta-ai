<script lang="ts">
	import { onDestroy, untrack } from 'svelte';
	import { flip } from 'svelte/animate';
	import { scale } from 'svelte/transition';
	import { cubicIn } from 'svelte/easing';
	import type { Game, RoundEnd } from './api';
	import { t } from './i18n.svelte';
	import GameResultsDialog from './GameResultsDialog.svelte';
	import PlayingCard from './PlayingCard.svelte';

	let { game, currentUser, onPlayCard, onSelectCapture, onReturnToLobby } = $props<{
		game: Game;
		currentUser: { id: string; username: string } | null;
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

	// The suit alone: at the size of a card in the counting pile's fan, a rank would be a smudge.
	function suitGlyph(cardIndex: number): string {
		return suits[Math.floor(cardIndex / 10)] ?? '';
	}

	function isCurrentUser(playerIndex: number): boolean {
		return myArrayIndex !== undefined && playerIndex === myArrayIndex;
	}

	// One thing at a time: input is frozen and incoming moves are not animated while the board
	// is playing a cinematic. `boardBusy` itself is declared with the choreography state below,
	// next to `phase`, so it cannot be read before `phase` exists.
	function canPlayCard(): boolean {
		return !boardBusy && game.currentPlayer === myArrayIndex;
	}

	function getPlayerLabel(index: number): string {
		return `P${index + 1}`;
	}

	const myPlayer = $derived(game.humanPlayers.find((p: any) => currentUser && p.userId === currentUser.id));
	const myArrayIndex = $derived(myPlayer ? myPlayer.index - 1 : undefined);

	const isMyTurn = $derived(myArrayIndex !== undefined && game.currentPlayer === myArrayIndex);

	function getCartonCount(capturedCards: number[] | undefined): number {
		if (!capturedCards || capturedCards.length === 0) return 0;
		const count = capturedCards.length;
		if (count <= 19) return 0;
		let points = count - 19 + 5;
		if (points % 2 !== 0) points++;
		return points;
	}

	let isSelectingCapture = $state(false);
	let displayedTable: number[] = $state([]);

	// ---------------------------------------------------------------------------
	// Deal / round-end choreography
	// ---------------------------------------------------------------------------
	// Two events drive the effects, both published by the backend:
	//   * `dealId` changes exactly when every seat has been dealt a fresh hand (the opening
	//     deal, a redeal from the deck, and the deal after a cartón).
	//   * `roundEnd` rides along with the ONE snapshot of the move that ended a deal and paid
	//     the cartón: the piles as they were counted, the cards left on the table, and the
	//     points. The state clears those piles in the same step, so this payload is the only
	//     chance to *show* the counting.
	// Because both can arrive together (the cartón ends a deal and the next one is already
	// dealt server-side), the sequence is explicit: wait -> count -> cartón paid -> deal.
	//
	// `waiting` is the first step: the sequence has been scheduled and the move that triggered it
	// is still animating. It shows nothing, but it is NOT idle -- input stays frozen and incoming
	// moves are not animated, so the cinematic cannot be interrupted halfway through.
	type Phase = 'idle' | 'waiting' | 'counting' | 'carton-paid' | 'dealing';
	let phase = $state<Phase>('idle');
	// True from the moment a sequence is scheduled until it returns the board to idle.
	const boardBusy = $derived(phase !== 'idle');
	// Whether the sequence in flight will deal a new hand. A hand that arrived with the sequence
	// stays face down until the deal animation turns it over, so the deal is not spoiled by the
	// new cards appearing the instant the snapshot lands.
	let sequenceDeals = $state(false);
	let cartonEvent = $state<RoundEnd | null>(null);
	let counted = $state<[number, number]>([0, 0]);
	let dealtCards = $state(0);
	// While the counting plays, the table must keep showing the cards that were there when
	// the deal ended (the live table is already empty).
	let heldTable = $state<number[] | null>(null);
	// Bookkeeping the markup never reads: plain variables on purpose. They are written
	// inside the effect that also reads them, and a $state would re-trigger it forever.
	let seenDealId = 0;
	let seenRoundEndId = 0;
	let seqId = 0;

	const SEAT_COUNT = 4;
	const HAND_SIZE = 5;
	// Timings. The backend paces AI turns 1.5-3 s apart, so the whole cartón sequence
	// (count -> paid -> deal) is kept near 3 s: long enough to read, short enough that the
	// next move rarely lands on top of it.
	const HAND_DEAL_MS = 250; // per card
	const HAND_GAP_MS = 180; // beat between one player's five cards and the next player's
	const COUNT_MS = 1000; // total time to count each pile
	const PAID_MS = 2500; // how long "cartón paid" stays on screen
	// When the last card of the deal lands. The deal hands out five cards per player (see
	// `cardsDealtTo`), so it is five cards plus one beat, four times over.
	const DEAL_TOTAL_MS = SEAT_COUNT * HAND_SIZE * HAND_DEAL_MS + (SEAT_COUNT - 1) * HAND_GAP_MS;
	const DEAL_HOLD_MS = 1000; // settle time after the last card lands
	// Hard deadline for one cartón+deal sequence, in case a step is ever missed. The
	// cinematic must never be able to leave the board non-interactive.
	const SEQUENCE_CAP_MS = COUNT_MS + PAID_MS + DEAL_TOTAL_MS + DEAL_HOLD_MS + 1200;

	// Five cards to one player, THEN the next: a hand lands as a group rather than a card at a
	// time around the table. This is what the backend does too -- `deal`/`redeal` in
	// backend/internal/game/cuarenta.go copy `HandSize` cards per seat in seat order, and only
	// redeal when a full 20 cards are left -- so the animation is a faithful replay.

	// How many cards `seat` holds once `dealt` cards have left the deck.
	function cardsDealtTo(seat: number, dealt: number): number {
		return Math.min(HAND_SIZE, Math.max(0, dealt - seat * HAND_SIZE));
	}

	// The seat the card that just landed went to (`dealt` counts cards dealt so far, 1-based).
	function seatOfCard(dealt: number): number {
		return Math.floor((dealt - 1) / HAND_SIZE);
	}

	// When the `k`-th card (0-based) of the deal lands: its five-card group's turn, plus a
	// small beat before each new player starts receiving.
	function dealDelayMs(k: number): number {
		return k * HAND_DEAL_MS + Math.floor(k / HAND_SIZE) * HAND_GAP_MS;
	}

	// ---------------------------------------------------------------------------------------
	// ONE ANIMATION AT A TIME
	// ---------------------------------------------------------------------------------------
	// The snapshot that ends a deal also carries the move that ended it (the cartón is paid by
	// a capture, and a deal from the deck follows the card that emptied the hands). Starting the
	// counting or the deal on that same snapshot plays two cinematics on top of each other: the
	// captured cards are still flying to the player while the piles are already being counted.
	// So the choreography waits for the move animation to land first.
	// A capture has to be READABLE: a card lands, there is a beat in which the cards it takes are
	// highlighted, and they then leave one after another -- not all at once, which is what made it
	// feel like a scuffle. The numbers below are the pacing of that story; the CSS animations
	// (`.fly-card.throw`, `flyToSide`) are kept in step with them.
	const THROW_MS = 1150; // the played card flies in from its player's side
	const CAPTURE_HOLD_MS = 500; // beat: the cards it will take sit highlighted on the table
	const CAPTURE_STAGGER_MS = 150; // each captured card leaves after the one before it
	const GRAB_MS = 1250; // a card flies out to the player who grabbed it

	// How long the animation for `move` runs: the card flies in, and on a capture the played and
	// captured cards then fly out to the player, LAST one first. The board is busy for the whole
	// of it, so this has to include the staggered departures.
	function moveAnimationMs(move: { captureIndices?: number[] } | undefined): number {
		if (!move) return 0;
		const captured = move.captureIndices?.length ?? 0;
		if (captured === 0) return THROW_MS;
		return THROW_MS + CAPTURE_HOLD_MS + captured * CAPTURE_STAGGER_MS + GRAB_MS;
	}

	// The card that was played leaves first, then the cards it took, one after another: the eye
	// can follow each card to the player instead of watching a pile jump.
	function departureDelayMs(cardIndex: number): number {
		const move = game.move;
		if (!move) return 0;
		const position = (move.captureIndices ?? []).indexOf(cardIndex);
		return position < 0 ? 0 : (position + 1) * CAPTURE_STAGGER_MS;
	}

	// True while the cards a capture is about to take are highlighted, so the beat between the
	// played card landing and those cards leaving says WHICH cards are being taken.
	let captureHold = $state(false);
	const heldCapture = $derived((game.move?.captureIndices ?? []) as number[]);
	function isBeingCaptured(cardIndex: number): boolean {
		return captureHold && heldCapture.includes(cardIndex);
	}

	// The move the running sequence waits for, and the move the board has already dealt with.
	// Plain variables on purpose: both are written by one effect and read by the other, and a
	// `$state` would make each one re-run the other.
	let sequenceMoveStep = -1;
	let handledMove = -1;

	// When the move animation currently on screen ends. Written by the move effect, read here:
	// a deal can also arrive on its own snapshot, with the previous move still in flight, and
	// then `game.move` is already gone. Deliberately NOT `$state`: this effect reads it while
	// also writing state, and reacting to it would re-run the effect mid-sequence.
	let moveAnimUntil = 0;

	$effect(() => {
		const dealId = game.dealId ?? 0;
		const roundEnd = game.roundEnd;
		const roundEndKey = roundEnd ? game.timeSteps ?? 0 : 0;
		// Both reads must happen before the guards so the effect re-runs on either change.
		if (dealId === 0 && !roundEnd) return;

		const pendingDeal = dealId > seenDealId;
		const pendingCarton = roundEnd && roundEndKey !== seenRoundEndId && roundEnd !== undefined;
		if (!pendingDeal && !pendingCarton) return;

		seenDealId = Math.max(seenDealId, dealId);
		if (pendingCarton && roundEndKey) seenRoundEndId = roundEndKey;

		// SEQUENCE OWNERSHIP. Each sequence gets an id and every callback checks it, so a
		// NEWER sequence orphans an older one. What it must NOT do is cancel the running
		// sequence's timers on every re-run: snapshots arrive per move (including the AI's,
		// mid-animation), each one re-runs this effect, and the guard above then returns
		// early — so a teardown that cleared the timers would leave the board stuck in
		// 'dealing'/'carton-paid' with a full-screen overlay up. That was the freeze.
		const id = ++seqId;
		const alive = () => seqId === id;

		// Let the move that triggered all this land first. On the cartón snapshot the move is
		// right here in the payload, so its duration is known exactly; if this snapshot has no
		// move, the previous one may still be in flight, and `moveAnimUntil` says until when.
		// The offset is applied inside `schedule`, so every step (and the deadline) shifts with
		// it instead of racing it.
		const settleMs = game.move
			? moveAnimationMs(game.move)
			: Math.max(0, moveAnimUntil - performance.now());
		const schedule = (fn: () => void, delay: number) => {
			setTimeout(() => {
				if (alive()) fn();
			}, settleMs + delay);
		};

		// From here the board is busy: input is frozen and moves that arrive are not animated
		// until the sequence ends. `waiting` shows nothing -- it is the move animation playing
		// out -- but it keeps the player from starting a turn the deal is about to overwrite.
		// (This effect never returns a teardown; see the note at its end.)
		sequenceDeals = Boolean(pendingDeal);
		phase = 'waiting';
		// The move this sequence waits for. It is the one move that still animates while the
		// board is busy; anything later is mirrored without animation (see the move effect).
		sequenceMoveStep = game.move ? game.timeSteps ?? 0 : -1;

		// Belt and braces: whatever happens, the cinematic cannot hold the UI hostage. A
		// stuck overlay is indistinguishable from a frozen app, so every sequence has a hard
		// deadline that returns the board to idle.
		schedule(() => {
			phase = 'idle';
			dealtCards = 0;
			cartonEvent = null;
			heldTable = null;
			sequenceDeals = false;
		}, SEQUENCE_CAP_MS);

		const startDeal = () => {
			phase = 'dealing';
			dealtCards = 0;
			for (let k = 0; k < SEAT_COUNT * HAND_SIZE; k++) {
				schedule(() => (dealtCards = k + 1), dealDelayMs(k));
			}
			schedule(() => {
				phase = 'idle';
				dealtCards = 0;
				sequenceDeals = false;
			}, DEAL_TOTAL_MS + DEAL_HOLD_MS);
		};

		if (pendingCarton && roundEnd) {
			// 1. Count the piles, card by card, while the table keeps its cards. This starts on
			//    delay 0, i.e. right after the move animation: the cards that were on the table
			//    are put back (heldTable) only then, because until that moment the move effect is
			//    still animating them off it.
			schedule(() => {
				cartonEvent = roundEnd;
				counted = [0, 0];
				heldTable = roundEnd.tableCards ?? [];
				phase = 'counting';
			}, 0);

			// ~26 updates reads as "counting", whatever the pile sizes are.
			const steps = 26;
			const targets: [number, number] = [roundEnd.counted?.[0] ?? 0, roundEnd.counted?.[1] ?? 0];
			for (let s = 1; s <= steps; s++) {
				schedule(() => {
					const t = s / steps;
					counted = [Math.round(targets[0] * t), Math.round(targets[1] * t)];
				}, (s * COUNT_MS) / steps);
			}
			// 2. Cartón paid: show the points, and let the table empty out.
			schedule(() => {
				counted = [targets[0], targets[1]];
				phase = 'carton-paid';
				// Dropping the held cards is all it takes: the move effect has already put
				// the live (empty) table in `displayedTable`, which takes over again.
				heldTable = null;
			}, COUNT_MS);
			// 3. Then the new deal.
			schedule(() => {
				cartonEvent = null;
				if (pendingDeal) startDeal();
				else phase = 'idle';
			}, COUNT_MS + PAID_MS);
		} else {
			schedule(startDeal, 0);
		}

		// No teardown: the sequence above owns its timers and always terminates (its last
		// step is idle, and the cap guarantees one). Cancelling them here is what wedged the
		// board; orphaning on unmount is handled by `onDestroy`, which bumps `seqId`.
	});

	// Leaving the board (or the game) orphans whatever sequence is in flight, so its
	// remaining callbacks become no-ops instead of writing to a dead component.
	onDestroy(() => {
		seqId++;
	});

	// NUMBER KEYS PLAY A CARD. A hand is five cards and the board is mouse-driven otherwise,
	// so on a desktop this is the whole game: 1-5 picks the n-th card of your hand. Guarded
	// on it being your turn, no capture prompt being open (that choice belongs to the
	// prompt), and the focus not sitting in a form field.
	function handleKeydown(event: KeyboardEvent) {
		// `boardBusy` covers it being your turn during a cinematic: the cards are still being
		// counted or dealt, so the move would land on top of an animation.
		if (boardBusy || !isMyTurn || isSelectingCapture || game.awaitingCapture) return;
		if (event.metaKey || event.ctrlKey || event.altKey) return;
		const target = event.target as HTMLElement | null;
		if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
		const index = Number.parseInt(event.key, 10);
		if (!Number.isInteger(index) || index < 1 || index > 9) return;
		const hand = myArrayIndex !== undefined ? game.cards?.[myArrayIndex] : undefined;
		const cardIndex = hand?.[index - 1];
		if (cardIndex === undefined || cardIndex < 0) return;
		event.preventDefault();
		onPlayCard(cardIndex);
	}

	// The hand is revealed in the same order the cards are dealt, so the animation matches
	// what actually happened: your five cards turn up together, when your seat's turn in the
	// deal comes round. A hand that arrived with the deal in flight stays face down until then
	// -- showing the new cards the instant the snapshot lands would spoil the deal, and those
	// cards are unplayable during it anyway.
	const myDealtCount = $derived(
		myArrayIndex === undefined
			? Number.POSITIVE_INFINITY
			: phase === 'dealing'
				? cardsDealtTo(myArrayIndex, dealtCards)
				: sequenceDeals && boardBusy
					? 0
					: Number.POSITIVE_INFINITY
	);

	// Game-end dialog, 2 s after the final snapshot so the last move's animation can finish.
	// `game.done` is the source of truth (a derived value, read-only); the delay is local
	// state. Writing to the derived directly -- as this used to -- makes the pause a no-op,
	// because the derived already reads the very flag the timer was meant to lag behind.
	const gameEnded = $derived(Boolean(game.done));
	let showGameEndDialog = $state(false);
	$effect(() => {
		if (!gameEnded) {
			showGameEndDialog = false;
			return;
		}
		const timer = setTimeout(() => {
			showGameEndDialog = true;
		}, 2000);
		return () => clearTimeout(timer);
	});

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

	// Tighter offsets for the dealing overlay: the throw/grab animations travel the whole
	// board, but four card piles have to fit on a phone screen. These are capped against the
	// viewport, so on a narrow screen the piles close in rather than sliding off the edge (a
	// pile positioned past the viewport edge is what drags the page sideways). They are only
	// ever read as CSS custom properties, so `min()`/`calc()` are fine here -- unlike OFFSET
	// below, which `flyToSide` parses as a number.
	const FX_OFFSET: Record<string, { tx: string; ty: string }> = {
		down: { tx: '0px', ty: 'min(150px, 20vh)' },
		right: { tx: 'min(210px, 26vw)', ty: '0px' },
		up: { tx: '0px', ty: 'calc(-1 * min(150px, 20vh))' },
		left: { tx: 'calc(-1 * min(210px, 26vw))', ty: '0px' }
	};

	// While a cartón is being counted the live table is already empty, so the cards that
	// were on it come from the round-end payload; `shownTable` keeps them on screen.
	const shownTable = $derived(heldTable ?? displayedTable);

	let throwAnim = $state<{ card: number; dir: string; id: number } | null>(null);
	// Direction the currently-leaving table cards fly toward (read by the out transition).
	let removeDir = $state('down');
	let animId = 0;

	// Animation timings (ms) — must match the CSS animation durations. Declared with the
	// choreography above, because the deal/cartón sequence has to wait for them.

	// Out transition: a table card flies off toward the player who grabbed it. `delay` staggers
	// the captured cards so they leave in sequence (see departureDelayMs).
	function flyToSide(_node: Element, { dir, delay = 0 }: { dir: string; delay?: number }) {
		const tx = parseFloat(OFFSET[dir].tx);
		const ty = parseFloat(OFFSET[dir].ty);
		return {
			duration: GRAB_MS,
			delay,
			easing: cubicIn,
			// t goes 1 -> 0; start centered/opaque, end translated/faded.
			css: (t: number) =>
				`transform: translate(${tx * (1 - t)}px, ${ty * (1 - t)}px) scale(${0.5 + 0.5 * t}); opacity: ${t};`
		};
	}

	$effect(() => {
		const table = game.table ?? [];
		const move = game.move;
		const currentPlayer = game.currentPlayer;
		// One move per snapshot: the backend bumps `stepCount` on every turn.
		const step = game.timeSteps ?? 0;

		// A cinematic owns the board. Only the move the sequence is waiting for still animates
		// (the sequence's own first step hangs off it); any move after that is mirrored without
		// animation, and marked as seen so it is not replayed when the board goes idle again.
		// Animating here is what put a throw/grab on top of the counting or the deal.
		if (boardBusy && step !== sequenceMoveStep) {
			displayedTable = table;
			throwAnim = null;
			captureHold = false;
			moveAnimUntil = 0;
			if (move && currentPlayer !== undefined) handledMove = step;
			return;
		}

		// No move to animate: just mirror the table.
		if (!move || currentPlayer === undefined) {
			displayedTable = table;
			// Re-running this effect cancels the timers of an in-flight throw (Svelte runs
			// the previous teardown first), so a snapshot that carries no move -- the
			// opening deal, for instance -- would otherwise leave the thrown card stuck on
			// screen forever. Drop it explicitly.
			throwAnim = null;
			captureHold = false;
			// Nothing is animating any more, so the deal/cartón choreography is free to run.
			moveAnimUntil = 0;
			return;
		}

		// The same move can be seen twice: this effect re-runs whenever one of its dependencies
		// changes (the phase ending a sequence, for instance) and a reconnect can resend the same
		// snapshot. Animate each move once.
		if (step === handledMove) return;
		handledMove = step;

		const before = untrack(() => displayedTable);
		const dir = dirFor((currentPlayer - 1 + 4) % 4);
		const id = ++animId;
		const captures = move.captureIndices ?? [];
		// Publish when this animation is over, for the choreography effect above.
		moveAnimUntil = performance.now() + moveAnimationMs(move);

		// 1. Throw: keep the old table visible while the played card flies in.
		throwAnim = { card: move.cardIndex, dir, id };

		const timers: ReturnType<typeof setTimeout>[] = [];

		// 2. Played card lands -> inserted into the table (scale-in, neighbors reflow).
		//    On a capture, also keep the captured cards so they can be grabbed next.
		timers.push(setTimeout(() => {
			if (animId !== id) return;
			throwAnim = null;
			displayedTable = captures.length > 0 ? [...before, move.cardIndex] : table;
			// 2b. A capture holds here: the cards the played card takes are lit up for a beat
			//     before any of them moves, so the move can be seen rather than guessed.
			captureHold = captures.length > 0;
		}, THROW_MS));

		if (captures.length > 0) {
			// 3. Grab: remove the played + captured cards from the table. Each flies off toward
			//    the player (out transition, staggered by `departureDelayMs`); remaining cards
			//    reflow (flip).
			timers.push(setTimeout(() => {
				if (animId !== id) return;
				captureHold = false;
				removeDir = dir;
				displayedTable = table;
			}, THROW_MS + CAPTURE_HOLD_MS));
			// The captured cards are still travelling to their player until the out transition
			// ends, so release the board only then -- this is what the deal waits for.
			timers.push(setTimeout(() => {
				if (animId !== id) return;
				moveAnimUntil = 0;
			}, moveAnimationMs(move)));
		}

		return () => timers.forEach(clearTimeout);
	});
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="game-board">
	{#if phase === 'counting'}
		<div class="round-indicator counting">{t('🃏 End of the deal — counting the cartón…')}</div>
	{:else if phase === 'carton-paid'}
		<div class="round-indicator paid">
			{t('Cartón paid: Team 1 +{first} · Team 2 +{second}', {
				first: cartonEvent?.carton?.[0] ?? 0,
				second: cartonEvent?.carton?.[1] ?? 0
			})}
		</div>
	{/if}

	<div class="board-header">
		<div class="game-info">
			<div class="info-item">
				<span class="label">{t('Game')}</span>
				<span class="value">
					<span class="culture-mark" title={t('Cuarenta, the Ecuadorian card game')}>
						<span class="culture-icon sm" role="img" aria-label={t('Ecuador')}></span>
						{game.id}
					</span>
				</span>
			</div>
		</div>

		<div class="scores">
			<div class="score-item">
				<span class="label">{t('Team 1')}</span>
				<span class="value">{game.scores?.[0] ?? 0}</span>
				{#if game.cardsCaptured}
					<span class="carton-info">
						<span class="carton-label">{t('Cartón:')}</span>
						<span class="carton-count">{getCartonCount(game.cardsCaptured[0])}</span>
						<span class="captured-label">({game.cardsCaptured[0]?.length ?? 0})</span>
					</span>
				{/if}
			</div>
			<div class="score-item">
				<span class="label">{t('Team 2')}</span>
				<span class="value">{game.scores?.[1] ?? 0}</span>
				{#if game.cardsCaptured}
					<span class="carton-info">
						<span class="carton-label">{t('Cartón:')}</span>
						<span class="carton-count">{getCartonCount(game.cardsCaptured[1])}</span>
						<span class="captured-label">({game.cardsCaptured[1]?.length ?? 0})</span>
					</span>
				{/if}
			</div>
		</div>
	</div>

	<!-- Current player indicator. Only YOUR turn is announced: for anyone else the seat label
	     alone says whose turn it is, and labelling a seat "Human"/"AI's turn" was noise. -->
	<div class="current-player-section">
		{#if getDisplayedCurrentPlayer() !== undefined && !game.done}
			{@const currentPlayerIdx = getDisplayedCurrentPlayer() ?? 0}
			<div class="current-player-text" class:your-turn={isCurrentUser(currentPlayerIdx)}>
				{getPlayerLabel(currentPlayerIdx)}
				{#if isCurrentUser(currentPlayerIdx)}
					<span class="turn-label" aria-live="polite">{t('Your turn')}</span>
				{/if}
			</div>
		{/if}
	</div>

	{#if showGameEndDialog}
		<GameResultsDialog {game} {onReturnToLobby} />
	{/if}

	<!-- Table area -->
	<div class="table-area">
		<div class="table-cards">
			{#if shownTable.length > 0}
				<div class="cards-display">
					{#each shownTable as cardIndex (cardIndex)}
						<div
							class="card on-table"
							class:captured={isBeingCaptured(cardIndex)}
							animate:flip={{ duration: 350 }}
							in:scale={{ duration: 300, start: 0.4 }}
							out:flyToSide={{ dir: removeDir, delay: departureDelayMs(cardIndex) }}
						>
							<PlayingCard {cardIndex} />
						</div>
					{/each}
				</div>
			{:else}
				<div class="empty-table">{t('Table is empty')}</div>
			{/if}
		</div>

		<!-- Played card flying in from the player's side toward the table -->
		<div class="anim-overlay">
			{#if throwAnim}
				{#key throwAnim.id}
					<div
						class="fly-card throw"
						style="--tx:{OFFSET[throwAnim.dir].tx}; --ty:{OFFSET[throwAnim.dir].ty}"
					>
						<PlayingCard cardIndex={throwAnim.card} />
					</div>
				{/key}
			{/if}
		</div>

		<!-- Deal: the deck empties into four piles, five cards to a player before the next -->
		{#if phase === 'dealing'}
			<div class="fx-overlay">
				<div class="deck-pile">
					{#each Array(Math.max(0, SEAT_COUNT * HAND_SIZE - dealtCards)) as _, i}
						<span class="card-back" style="--i:{i}"></span>
					{/each}
				</div>
				{#each [0, 1, 2, 3] as seat (seat)}
					{@const dir = dirFor(seat)}
					<div
						class="seat-pile"
						class:mine={seat === myArrayIndex}
						style="--tx:{FX_OFFSET[dir].tx}; --ty:{FX_OFFSET[dir].ty}"
					>
						{#each Array(cardsDealtTo(seat, dealtCards)) as _, i}
							<span class="card-back landed" style="--i:{i}"></span>
						{/each}
						{#if seat === myArrayIndex}<span class="seat-label">{t('You')}</span>{/if}
					</div>
				{/each}
				{#if dealtCards > 0 && dealtCards <= SEAT_COUNT * HAND_SIZE}
					{@const dir = dirFor(seatOfCard(dealtCards))}
					{#key dealtCards}
						<span
							class="deal-fly card-back"
							style="--tx:{FX_OFFSET[dir].tx}; --ty:{FX_OFFSET[dir].ty}"
						></span>
					{/key}
				{/if}
			</div>
		{/if}

		<!-- Round end: count the two captured piles, card by card, then pay the cartón -->
		{#if (phase === 'counting' || phase === 'carton-paid') && cartonEvent}
			<div class="fx-overlay carton-overlay">
				{#each [0, 1] as team (team)}
					<div class="carton-pile" class:paid={phase === 'carton-paid'}>
						<div class="carton-pile-head">{t('Team {team}', { team: team + 1 })}</div>
						<div class="carton-stack">
							{#each cartonEvent.captured?.[team] ?? [] as cardIndex, i (i)}
								<!-- Face-down pile: cards turn face-up one at a time as they are
								     counted, so the animation reads as counting rather than as a
								     slide show of the final numbers. -->
								<span
									class="card-back"
									class:flipped={i < counted[team]}
									class:red={Math.floor(cardIndex / 10) <= 1}
									style="--i:{i}"
									title={i < counted[team] ? cardToString(cardIndex) : undefined}
									>{#if i < counted[team]}{suitGlyph(cardIndex)}{/if}</span
								>
							{/each}
						</div>
						<!-- Just the running count: the total was a spoiler, and the cards turning
						     over one by one already show the pile being counted. -->
						<div class="carton-tally">
							<span class="tally-number">{counted[team]}</span>
						</div>
						{#if phase === 'carton-paid'}
							<div class="carton-paid-badge" class:zero={(cartonEvent.carton?.[team] ?? 0) === 0}>
								{(cartonEvent.carton?.[team] ?? 0) > 0
									? t('+{points} points', { points: cartonEvent.carton[team] })
									: t('no cartón')}
							</div>
						{/if}
					</div>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Capture selection section -->
	{#if game.awaitingCapture && game.captureOptions && game.captureOptions.length > 0 && !isSelectingCapture && !boardBusy}
		<div class="capture-selection">
			<div class="section-title">{t('Choose which cards to capture:')}</div>
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
							{t('No capture')}
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
				<div class="hand-label">
					{isMyTurn ? t('Your cards') : t('Your hand')}
					{#if isMyTurn}<span class="hand-hint">{t('Press 1-5 to play a card')}</span>{/if}
				</div>
			</div>
			<div class="cards-container">
				{#each game.cards[myArrayIndex] as cardIndex, i}
					{#if cardIndex >= 0}
						{#if i < myDealtCount}
							<button
								class="card"
								class:playable={canPlayCard()}
								class:just-dealt={phase === 'dealing'}
								style="--i:{i}"
								disabled={!canPlayCard()}
								onclick={() => onPlayCard(cardIndex)}
							>
								<PlayingCard {cardIndex} />
							</button>
						{:else}
							<span class="card card-back placeholder" style="--i:{i}"></span>
						{/if}
					{/if}
				{/each}
			</div>
		</div>
	{/if}
</div>

<style>
	/* Fits the screen: `min-width: 0` lets this shrink inside a grid/flex parent (which
	   otherwise sizes to its content's min-content width and pushes the page sideways), and the
	   clip is the last line of defence against anything that still travels past the edge. */
	.game-board {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
		height: 100%;
		width: 100%;
		max-width: 100%;
		min-width: 0;
		overflow-x: clip;
	}

	.board-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		row-gap: 0.4rem;
		min-width: 0;
		border-bottom: 2px solid white/10;
	}

	.game-info,
	.scores {
		display: flex;
		flex-wrap: wrap;
		gap: clamp(0.75rem, 3vw, 2rem);
		min-width: 0;
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
		min-width: 0;
		max-width: 100%;
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
		/* The throw/grab travels 340-560px toward a player's side, which is off-screen on a
		   phone: clip it here so the card leaves the table instead of widening the page. */
		overflow: hidden;
		display: flex;
		align-items: center;
		justify-content: center;
		pointer-events: none;
		z-index: 10;
	}

	.fly-card {
		padding: 0;
		border-radius: 0.45rem;
		width: clamp(2.9rem, 12vw, 3.9rem);
		aspect-ratio: 5 / 7;
		display: block;
		overflow: hidden;
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
		will-change: transform, opacity;
	}

	.fly-card.throw {
		/* Keep in step with THROW_MS. */
		animation: throw-in 1150ms cubic-bezier(0.22, 1, 0.36, 1);
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

	/* The frame only: the face itself is PlayingCard.svelte. Portrait, because a card is taller
	   than it is wide -- the ratio matches the card backs so a face and a back are the same
	   shape. */
	.card {
		padding: 0;
		border-radius: 0.45rem;
		border: 2px solid rgba(255, 255, 255, 0.3);
		background: transparent;
		cursor: pointer;
		transition: all 0.2s;
		width: clamp(4.2rem, 12vw, 4.9rem);
		aspect-ratio: 5 / 7;
		display: block;
		overflow: hidden;
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
	}

	.card.on-table {
		cursor: default;
		pointer-events: none;
		width: clamp(4.5rem, 9.5vw, 4.4rem);
	}


	/* The cards a capture is about to take, lit up for a beat before they leave. No transform: the
	   FLIP animation writes `transform` inline, and fighting it would make the cards jump. */
	.card.captured {
		border-color: #fbbf24;
		box-shadow:
			0 0 0 3px rgba(251, 191, 36, 0.6),
			0 8px 22px rgba(251, 191, 36, 0.35);
		filter: brightness(1.12);
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
		min-width: 0;
		max-width: 100%;
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

	/* Wrap, never scroll sideways: the hand is at most five cards, and a horizontally
	   scrollable row hides a card behind the screen edge. */
	.cards-container {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
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

	.round-indicator {
		max-width: 100%;
		overflow-wrap: anywhere;
		padding: 0.75rem 1rem;
		border-radius: 0.5rem;
		font-size: 0.875rem;
		font-weight: 600;
		color: #86efac;
		text-align: center;
		animation: pulse 0.5s ease-in-out;
		background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(34, 197, 94, 0.1));
		border: 2px solid #22c55e;
	}

	.round-indicator.counting {
		color: #fde68a;
		background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(251, 191, 36, 0.08));
		border-color: #fbbf24;
	}

	.round-indicator.paid {
		color: #fbbf24;
		font-size: 1rem;
		background: linear-gradient(135deg, rgba(251, 191, 36, 0.3), rgba(251, 191, 36, 0.12));
		border-color: #fbbf24;
		box-shadow: 0 0 24px rgba(251, 191, 36, 0.35);
	}

	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.7; }
	}

	/* ---------------- dealing / cartón effects ---------------- */

	.fx-overlay {
		position: fixed;
		inset: 0;
		overflow: hidden;
		display: flex;
		align-items: center;
		justify-content: center;
		pointer-events: none;
		z-index: 40;
	}

	.carton-overlay {
		background: radial-gradient(circle at center, rgba(2, 6, 23, 0.82), rgba(2, 6, 23, 0.62));
		backdrop-filter: blur(2px);
	}

	/* The same 5/7 shape as a face, so the cards dealt onto the table are the shape of the hand
	   they turn into. */
	.card-back {
		display: inline-block;
		width: clamp(1.6rem, 6vw, 2.4rem);
		aspect-ratio: 5 / 7;
		border-radius: 0.375rem;
		border: 1px solid rgba(255, 255, 255, 0.35);
		background:
			repeating-linear-gradient(45deg, rgba(255, 255, 255, 0.16) 0 4px, transparent 4px 8px),
			linear-gradient(135deg, #1e3a8a, #0f172a);
		box-shadow: 0 6px 16px rgba(0, 0, 0, 0.45);
	}

	.deck-pile {
		position: absolute;
		display: grid;
		place-items: center;
	}

	.deck-pile .card-back {
		grid-area: 1 / 1;
		transform: translate(calc(var(--i) * 0.5px), calc(var(--i) * -1.1px)) rotate(calc(var(--i) * 0.4deg));
		opacity: 0.95;
	}

	.seat-pile {
		position: absolute;
		display: grid;
		grid-template-columns: repeat(2, auto);
		gap: 0.25rem;
		transform: translate(var(--tx), var(--ty));
		place-items: center;
	}

	.seat-pile.mine {
		filter: drop-shadow(0 0 12px rgba(251, 191, 36, 0.5));
	}

	.seat-pile .card-back.landed {
		animation: land 220ms cubic-bezier(0.22, 1, 0.36, 1) both;
	}

	.seat-label {
		grid-column: 1 / -1;
		margin-top: 0.25rem;
		font-size: 0.7rem;
		font-weight: 700;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: #fbbf24;
	}

	@keyframes land {
		from { transform: scale(0.6) rotate(-6deg); opacity: 0; }
		to { transform: scale(1) rotate(0deg); opacity: 1; }
	}

	.deal-fly {
		position: absolute;
		animation: deal-fly 300ms cubic-bezier(0.2, 0.9, 0.3, 1) both;
	}

	@keyframes deal-fly {
		from { transform: translate(0, 0) scale(0.9) rotate(12deg); opacity: 0; }
		35% { opacity: 1; }
		to { transform: translate(var(--tx), var(--ty)) scale(1) rotate(0deg); opacity: 0.9; }
	}

	.carton-pile {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.5rem;
		margin: 0 clamp(0.75rem, 6vw, 4rem);
		padding: 1rem clamp(0.85rem, 3vw, 1.5rem);
		border-radius: 1rem;
		background: rgba(15, 23, 42, 0.75);
		border: 2px solid rgba(251, 191, 36, 0.35);
		transition: border-color 0.3s, box-shadow 0.3s;
	}

	.carton-pile.paid {
		border-color: #fbbf24;
		box-shadow: 0 0 28px rgba(251, 191, 36, 0.35);
	}

	.carton-pile-head {
		font-size: 0.75rem;
		font-weight: 700;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		color: #cbd5e1;
	}

	.carton-stack {
		display: grid;
		grid-template-columns: repeat(6, auto);
		gap: 0.15rem;
		justify-content: center;
		max-width: 18rem;
	}

	.carton-stack .card-back {
		width: clamp(0.7rem, 2.4vw, 1.05rem);
		height: clamp(1rem, 3.4vw, 1.5rem);
		border-radius: 0.2rem;
		transition: transform 0.18s ease, opacity 0.18s ease, background 0.18s ease;
		opacity: 0.55;
	}

	.carton-stack .card-back.flipped {
		opacity: 1;
		background: linear-gradient(160deg, #fffdf7, #f3efe4);
		border-color: rgba(15, 23, 42, 0.35);
		color: #0f172a;
		font-size: clamp(0.5rem, 1.6vw, 0.7rem);
		font-weight: 700;
		line-height: 1.2;
		text-align: center;
		transform: translateY(-3px);
	}

	.carton-stack .card-back.flipped.red {
		color: #be123c;
	}

	.carton-tally {
		display: flex;
		align-items: baseline;
		gap: 0.3rem;
		font-family: monospace;
	}

	.tally-number {
		font-size: clamp(1.5rem, 5vw, 2.25rem);
		font-weight: 700;
		color: #fbbf24;
	}

	.carton-paid-badge {
		padding: 0.25rem 0.6rem;
		border-radius: 999px;
		font-size: 0.85rem;
		font-weight: 700;
		color: #052e16;
		background: #fbbf24;
		animation: land 260ms ease both;
	}

	.carton-paid-badge.zero {
		color: #cbd5e1;
		background: rgba(148, 163, 184, 0.25);
	}

	.hand-hint {
		margin-left: 0.6rem;
		font-size: 0.7rem;
		font-weight: 500;
		letter-spacing: 0.02em;
		text-transform: none;
		color: #94a3b8;
	}

	.card.just-dealt {
		animation: land 240ms cubic-bezier(0.22, 1, 0.36, 1) both;
	}

	.card.placeholder {
		border: 1px dashed rgba(255, 255, 255, 0.25);
		background: transparent;
		box-shadow: none;
		cursor: default;
	}

	/* The culture mark's styles are global (see routes/layout.css) so the header of this
	   board and the app header can share one definition. */

	.carton-info {
		display: flex;
		gap: 0.5rem;
		align-items: center;
		margin-top: 0.25rem;
		font-size: 0.75rem;
	}

	.carton-label {
		color: #94a3b8;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.carton-count {
		color: #fbbf24;
		font-weight: 700;
		font-size: 1rem;
	}

	.captured-label {
		color: #64748b;
		font-size: 0.7rem;
	}

</style>
