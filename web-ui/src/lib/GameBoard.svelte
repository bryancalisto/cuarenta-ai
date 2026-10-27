<script lang="ts">
	import { onDestroy, tick, untrack } from 'svelte';
	import { fade } from 'svelte/transition';
	import type { Game, Move, RoundEnd } from './api';
	import { t } from './i18n.svelte';
	import GameResultsDialog from './GameResultsDialog.svelte';
	import PlayingCard from './PlayingCard.svelte';
	import { slotOffsetUnits, TableSlots } from './tableLayout';

	let { game, currentUser, onPlayCard, onSelectCapture, onLeaveTable } = $props<{
		game: Game;
		currentUser: { id: string; username: string } | null;
		onPlayCard: (cardIndex: number) => void;
		onSelectCapture: (captureIndices: number[]) => void;
		onLeaveTable: () => void;
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

	// ---------------------------------------------------------------------------------------
	// ONE SNAPSHOT AT A TIME
	// ---------------------------------------------------------------------------------------
	// The backend publishes a snapshot per move and paces the AI 1.5-3 s apart, which is shorter
	// than a capture takes to show. Reacting to `game` directly is what made animations overlap:
	// the next move started while the previous capture was still flying, and the cards of one
	// capture ended up split between two players.
	//
	// So the board does NOT render `game`. Every snapshot goes into a queue, and the board renders
	// `view`: the last snapshot whose story has been told. `present()` takes the next snapshot,
	// plays its move (the card flies in from its player's side; on a capture the taken cards light
	// up and then fly out to that player), then the cartón counting and the deal if the snapshot
	// carries them, and only THEN moves `view` on and takes the next one. Nothing can overlap,
	// because there is only ever one `present()` running.
	let view = $state<Game>(untrack(() => game));
	const queue: Game[] = [];
	// Snapshots waiting their turn. Mirrors `queue.length` so the markup can react to it.
	let queued = $state(0);
	let draining = false;
	let presentedAny = false;
	let destroyed = false;

	// A backlog this deep is a reconnect or a tab that was asleep, not a game being played: it is
	// caught up without the show, or the board would replay a minute of moves.
	const BACKLOG_LIMIT = 4;

	$effect(() => {
		const next = game;
		untrack(() => {
			queue.push(next);
			queued = queue.length;
			if (!draining) void drain();
		});
	});

	async function drain() {
		draining = true;
		try {
			while (queue.length > 0 && !destroyed) {
				const next = queue.shift() as Game;
				queued = queue.length;
				try {
					await present(next);
				} catch (error) {
					// A broken animation must never wedge the board: show the snapshot as it is.
					console.error('board animation failed', error);
					resetChoreography();
					view = next;
				}
			}
		} finally {
			draining = false;
		}
	}

	onDestroy(() => {
		destroyed = true;
	});

	function sleep(ms: number): Promise<void> {
		return new Promise((resolve) => setTimeout(resolve, ms));
	}

	function myIndexIn(g: Game): number | undefined {
		const me = g.humanPlayers.find((p: any) => currentUser && p.userId === currentUser.id);
		return me ? me.index - 1 : undefined;
	}

	const myArrayIndex = $derived(myIndexIn(view));
	const isMyTurn = $derived(myArrayIndex !== undefined && view.currentPlayer === myArrayIndex);

	function isCurrentUser(playerIndex: number): boolean {
		return myArrayIndex !== undefined && playerIndex === myArrayIndex;
	}

	// One thing at a time: input is frozen while anything is animating or waiting to be animated.
	// A move made on a board that has not caught up would be made against cards the player has not
	// seen yet.
	function canPlayCard(): boolean {
		return !boardBusy && view.currentPlayer === myArrayIndex;
	}

	// SEATS. Each player has a side of the screen, relative to you (always "down"). Humans are
	// shown by name, the AI seats as AI 1, AI 2, ... in seat order.
	const DIRS = ['down', 'right', 'up', 'left'] as const;
	type Dir = (typeof DIRS)[number];

	function dirFor(playerIndex: number): Dir {
		const me = myArrayIndex ?? 0;
		return DIRS[(playerIndex - me + 4) % 4];
	}

	function seatAt(dir: Dir): number {
		return ((myArrayIndex ?? 0) + DIRS.indexOf(dir)) % 4;
	}

	function humanAt(seat: number) {
		return view.humanPlayers.find((p: any) => p.index - 1 === seat);
	}

	function seatName(seat: number): string {
		const human = humanAt(seat);
		if (human) return human.username || t('Player {n}', { n: seat + 1 });
		const aiSeats = [0, 1, 2, 3].filter((s) => !humanAt(s));
		return `AI ${aiSeats.indexOf(seat) + 1}`;
	}

	// Cards a seat holds, as the board shows them: a card on its way to the table has left the hand.
	function handCount(seat: number): number {
		const held = (view.cards?.[seat] ?? []).filter((c: number) => c >= 0).length;
		const dealt = phase === 'dealing' ? Math.min(held, cardsDealtTo(seat, dealtCards)) : sequenceDeals ? 0 : held;
		return Math.max(0, dealt - (throwing?.seat === seat ? 1 : 0));
	}

	function getCartonCount(capturedCards: number[] | undefined): number {
		if (!capturedCards || capturedCards.length === 0) return 0;
		const count = capturedCards.length;
		if (count <= 19) return 0;
		let points = count - 19 + 5;
		if (points % 2 !== 0) points++;
		return points;
	}

	let isSelectingCapture = $state(false);

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
	// Both can arrive on the snapshot of the move that ended the deal, so `present()` plays them
	// in order: move -> count -> cartón paid -> deal.
	type Phase = 'idle' | 'counting' | 'carton-paid' | 'dealing';
	let phase = $state<Phase>('idle');
	// True while a move is being animated.
	let animating = $state(false);
	// The board is busy while anything is on screen or still waiting in the queue.
	const boardBusy = $derived(phase !== 'idle' || animating || queued > 0);
	// Whether the snapshot being presented deals a new hand. That hand stays face down until the
	// deal animation turns it over, so the deal is not spoiled by the new cards appearing first.
	let sequenceDeals = $state(false);
	// The carton points the counting is about to explain, held back from the score board until it
	// has been counted on screen. The backend adds them to the score in the SAME step that
	// publishes `roundEnd`, so the snapshot already carries the final score.
	let heldCartonPoints = $state<[number, number] | null>(null);
	// ...and the points that just landed, for the "+N" next to the score. Not a timer: the badge
	// animates itself away (see .score-gain) and the next sequence clears it.
	let cartonGain = $state<[number, number] | null>(null);
	let cartonEvent = $state<RoundEnd | null>(null);
	let counted = $state<[number, number]>([0, 0]);
	let dealtCards = $state(0);
	// While the counting plays, the table must keep showing the cards that were there when
	// the deal ended (the live table is already empty).
	let heldTable = $state<number[] | null>(null);

	const SEAT_COUNT = 4;
	const HAND_SIZE = 5;
	const HAND_DEAL_MS = 250; // per card
	const HAND_GAP_MS = 180; // beat between one player's five cards and the next player's
	const COUNT_MS = 1000; // total time to count each pile
	const PAID_MS = 2500; // how long "cartón paid" stays on screen
	// When the last card of the deal lands. The deal hands out five cards per player (see
	// `cardsDealtTo`), so it is five cards plus one beat, four times over.
	const DEAL_TOTAL_MS = SEAT_COUNT * HAND_SIZE * HAND_DEAL_MS + (SEAT_COUNT - 1) * HAND_GAP_MS;
	const DEAL_HOLD_MS = 1000; // settle time after the last card lands

	// Five cards to one player, THEN the next: a hand lands as a group rather than a card at a
	// time around the table. This is what the backend does too -- `deal`/`redeal` in
	// backend/internal/game/cuarenta.go copy `HandSize` cards per seat in seat order, and only
	// redeal when a full 20 cards are left -- so the animation is a faithful replay.

	// The score to show for `team`: the real score, minus any carton the counting is still
	// explaining. It is the same number the score board showed before the deal ended, so nothing
	// jumps until the points have been counted out.
	function displayedScore(team: number): number {
		return Math.max(0, (view.scores?.[team] ?? 0) - (heldCartonPoints?.[team] ?? 0));
	}

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

	// A capture has to be READABLE: a card lands, there is a beat in which the cards it takes are
	// highlighted, and they then leave one after another, all of them towards the player who took
	// them.
	const THROW_MS = 900; // the played card flies in from its player's side
	const CAPTURE_HOLD_MS = 550; // beat: the cards it will take sit highlighted on the table
	const CAPTURE_STAGGER_MS = 160; // each captured card leaves after the one before it
	const GRAB_MS = 850; // a card flies out to the player who grabbed it

	// True while the cards a capture is about to take are highlighted.
	let heldCapture = $state<number[]>([]);
	function isBeingCaptured(cardIndex: number): boolean {
		return heldCapture.includes(cardIndex);
	}

	function resetChoreography() {
		phase = 'idle';
		animating = false;
		dealtCards = 0;
		cartonEvent = null;
		heldTable = null;
		tableOverride = null;
		sequenceDeals = false;
		heldCartonPoints = null;
		heldCapture = [];
		incomingCard = null;
		throwing = null;
		flights = [];
	}

	// Tells the story of one snapshot, then makes it the one on screen.
	async function present(next: Game) {
		const prev = view;
		const first = !presentedAny;
		presentedAny = true;

		if (first || next.id !== prev.id) {
			resetChoreography();
			view = next;
			// The opening deal is animated when the board opens on it; a board opened mid-game
			// (a reload) just shows the hands.
			if ((next.dealId ?? 0) > 0 && (next.timeSteps ?? 0) === 0) await playDeal();
			return;
		}

		const isNewStep = (next.timeSteps ?? 0) > (prev.timeSteps ?? 0);
		const hurry = queue.length >= BACKLOG_LIMIT || document.hidden;
		if (!isNewStep || hurry) {
			// The same step again (a resend, a capture prompt), or a backlog: just show it.
			view = next;
			return;
		}

		if (next.move) await playMove(prev, next.move);

		const roundEnd = next.roundEnd;
		const deals = (next.dealId ?? 0) > (prev.dealId ?? 0);
		if (roundEnd || deals) cartonGain = null;
		// Everything the next snapshot would give away before its time is held back BEFORE it goes
		// on screen: the cartón points (already in its score), the cards that were on the table when
		// the deal ended (its table is already empty), and the new hand (not dealt yet).
		sequenceDeals = deals;
		if (roundEnd) {
			heldCartonPoints = [roundEnd.carton?.[0] ?? 0, roundEnd.carton?.[1] ?? 0];
			heldTable = roundEnd.tableCards ?? [];
		}
		view = next;
		tableOverride = null;

		if (roundEnd) await playCounting(roundEnd);
		if (deals) await playDeal();
		sequenceDeals = false;
	}

	async function playCounting(roundEnd: RoundEnd) {
		cartonEvent = roundEnd;
		counted = [0, 0];
		phase = 'counting';
		// ~26 updates reads as "counting", whatever the pile sizes are.
		const steps = 26;
		const targets: [number, number] = [roundEnd.counted?.[0] ?? 0, roundEnd.counted?.[1] ?? 0];
		for (let s = 1; s <= steps; s++) {
			await sleep(COUNT_MS / steps);
			if (destroyed) return;
			counted = [Math.round((targets[0] * s) / steps), Math.round((targets[1] * s) / steps)];
		}
		// Cartón paid: show the points, and let the table empty out.
		counted = [targets[0], targets[1]];
		phase = 'carton-paid';
		heldTable = null;
		await sleep(PAID_MS);
		// The counting is over: pay it on the score board. The points land here rather than with
		// the snapshot, which is what makes the counter agree with what was just counted.
		const paid: [number, number] = [roundEnd.carton?.[0] ?? 0, roundEnd.carton?.[1] ?? 0];
		cartonEvent = null;
		heldCartonPoints = null;
		cartonGain = paid.some((points) => points > 0) ? paid : null;
		phase = 'idle';
	}

	async function playDeal() {
		sequenceDeals = true;
		phase = 'dealing';
		dealtCards = 0;
		const start = performance.now();
		const until = (at: number) => sleep(Math.max(0, start + at - performance.now()));
		for (let k = 0; k < SEAT_COUNT * HAND_SIZE; k++) {
			await until(dealDelayMs(k));
			if (destroyed) return;
			dealtCards = k + 1;
		}
		await until(DEAL_TOTAL_MS + DEAL_HOLD_MS);
		phase = 'idle';
		dealtCards = 0;
		sequenceDeals = false;
	}

	// ---------------------------------------------------------------------------------------
	// CARDS IN FLIGHT
	// ---------------------------------------------------------------------------------------
	// A card travelling between a player and the table is drawn in `.fly-layer`, a layer fixed to
	// the viewport, so it is visible the whole way: from the edge of the screen on its player's
	// side to its place on the table, and back out. (Drawn inside the table it was clipped to it.)
	type Box = { cx: number; cy: number; w: number; h: number };
	type Flight = {
		key: number;
		card: number;
		// The box the element is laid out in; the keyframes move it from/to the other boxes.
		base: Box;
		frames: Keyframe[];
		options: KeyframeAnimationOptions;
		done: () => void;
	};
	let flights = $state<Flight[]>([]);
	let flightKey = 0;
	let flyLayer: HTMLElement | undefined = $state();
	let boardEl: HTMLElement | undefined = $state();
	const seatEls: Partial<Record<Dir, HTMLElement>> = $state({});

	function boxOf(el: Element): Box {
		const r = el.getBoundingClientRect();
		return { cx: r.left + r.width / 2, cy: r.top + r.height / 2, w: r.width, h: r.height };
	}

	// Just past the edge of the screen on `dir`'s side, level with that player's seat.
	function edgeBox(dir: Dir, like: Box): Box {
		const seat = seatEls[dir];
		const anchor = seat ? boxOf(seat) : { cx: window.innerWidth / 2, cy: window.innerHeight / 2, w: 0, h: 0 };
		const margin = Math.max(like.w, like.h);
		const box = { ...like };
		if (dir === 'left') Object.assign(box, { cx: -margin, cy: anchor.cy });
		if (dir === 'right') Object.assign(box, { cx: window.innerWidth + margin, cy: anchor.cy });
		if (dir === 'up') Object.assign(box, { cx: anchor.cx, cy: -margin });
		if (dir === 'down') Object.assign(box, { cx: anchor.cx, cy: window.innerHeight + margin });
		return box;
	}

	function transformTo(box: Box, base: Box, rotate = 0): string {
		const s = box.w / base.w;
		return `translate(${box.cx - base.cx}px, ${box.cy - base.cy}px) scale(${s}) rotate(${rotate}deg)`;
	}

	// Flies `card` from `from` to `to`. `base` is the box the card is laid out in (the table box),
	// in viewport coordinates. Resolves when the card has arrived.
	function fly(card: number, base: Box, from: Box, to: Box, options: KeyframeAnimationOptions, spin = 0) {
		// The layer is `position: fixed`, but an ancestor with a transform would make it relative to
		// that ancestor instead of the viewport: measure where it really is.
		const origin = flyLayer?.getBoundingClientRect() ?? { left: 0, top: 0 };
		const local = { ...base, cx: base.cx - origin.left, cy: base.cy - origin.top };
		return new Promise<void>((resolve) => {
			const key = ++flightKey;
			flights = [
				...flights,
				{
					key,
					card,
					base: local,
					frames: [{ transform: transformTo(from, base, spin) }, { transform: transformTo(to, base) }],
					options,
					done: () => {
						flights = flights.filter((f) => f.key !== key);
						resolve();
					}
				}
			];
		});
	}

	// Svelte action: runs one flight's animation and reports when it is over (or torn down).
	function flight(node: HTMLElement, f: Flight) {
		const animation = node.animate(f.frames, { ...f.options, fill: 'both' });
		let finished = false;
		const finish = () => {
			if (finished) return;
			finished = true;
			f.done();
		};
		animation.finished.then(finish, finish);
		return {
			destroy() {
				animation.cancel();
				finish();
			}
		};
	}

	// What the table shows while a move is being told (null: the snapshot's own table).
	let tableOverride = $state<number[] | null>(null);
	// The card that is flying in: it already holds its place on the table, invisibly, so the
	// flight knows exactly where it lands.
	let incomingCard = $state<number | null>(null);
	// The card being thrown and whose hand it left.
	let throwing = $state<{ card: number; seat: number } | null>(null);

	function tableCardEl(card: number): HTMLElement | null {
		return boardEl?.querySelector<HTMLElement>(`[data-table-card="${card}"]`) ?? null;
	}

	// The move in `move`, told against the board as it was before it (`prev`).
	async function playMove(prev: Game, move: Move) {
		animating = true;
		try {
			// The move names its player: that is where the card comes from and where the capture
			// goes. Never inferred from whose turn it is now.
			const seat = move.playerIndex;
			const dir = dirFor(seat);
			const card = move.cardIndex;
			const captures = move.captureIndices ?? [];
			const before = (prev.table ?? []).filter((c: number) => c !== card);

			// 1. Throw. Your own card leaves from your hand; anyone else's from their edge.
			const handCard = boardEl?.querySelector<HTMLElement>(`[data-hand-card="${card}"]`);
			const handBox = handCard && seat === myArrayIndex ? boxOf(handCard) : null;
			throwing = { card, seat };
			incomingCard = card;
			tableOverride = [...before, card];
			await tick();
			const target = tableCardEl(card);
			if (target && !destroyed) {
				const base = boxOf(target);
				await fly(card, base, handBox ?? edgeBox(dir, base), base, {
					duration: THROW_MS,
					easing: 'cubic-bezier(0.22, 1, 0.36, 1)'
				}, handBox ? 0 : 10);
			}
			incomingCard = null;
			if (captures.length === 0 || destroyed) return;

			// 2. The cards it takes light up for a beat, so the capture can be seen.
			heldCapture = captures;
			await sleep(CAPTURE_HOLD_MS);
			heldCapture = [];
			if (destroyed) return;

			// 3. Grab: the played card, then each captured card, leave one after another, ALL of
			//    them to the player who made the move. They are measured where they sit, and handed
			//    to the fly layer in the same update that takes them off the table.
			const leaving = [card, ...captures];
			const grabs: Promise<void>[] = [];
			leaving.forEach((c, i) => {
				const el = tableCardEl(c);
				if (!el) return;
				const base = boxOf(el);
				grabs.push(
					fly(c, base, base, { ...edgeBox(dir, base), w: base.w * 0.8 }, {
						duration: GRAB_MS,
						delay: i * CAPTURE_STAGGER_MS,
						easing: 'cubic-bezier(0.55, 0, 0.75, 0.3)'
					})
				);
			});
			tableOverride = [...before, card].filter((c) => !leaving.includes(c));
			await Promise.all(grabs);
		} finally {
			// The hand keeps the card hidden until the snapshot that no longer has it goes on
			// screen, which `present()` does right after this.
			throwing = null;
			animating = false;
		}
	}

	// NUMBER KEYS PLAY A CARD. A hand is five cards and the board is mouse-driven otherwise,
	// so on a desktop this is the whole game: 1-5 picks the n-th card of your hand. Guarded
	// on it being your turn, no capture prompt being open (that choice belongs to the
	// prompt), and the focus not sitting in a form field.
	function handleKeydown(event: KeyboardEvent) {
		// `boardBusy` covers it being your turn during an animation: the move would land on top
		// of it.
		if (boardBusy || !isMyTurn || isSelectingCapture || view.awaitingCapture) return;
		if (event.metaKey || event.ctrlKey || event.altKey) return;
		const target = event.target as HTMLElement | null;
		if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
		const index = Number.parseInt(event.key, 10);
		if (!Number.isInteger(index) || index < 1 || index > 9) return;
		const hand = myArrayIndex !== undefined ? view.cards?.[myArrayIndex] : undefined;
		const cardIndex = hand?.[index - 1];
		if (cardIndex === undefined || cardIndex < 0) return;
		event.preventDefault();
		onPlayCard(cardIndex);
	}

	// The hand is revealed in the same order the cards are dealt, so the animation matches
	// what actually happened: your five cards turn up together, when your seat's turn in the
	// deal comes round. A hand that arrived with the deal in flight stays face down until then.
	const myDealtCount = $derived(
		myArrayIndex === undefined
			? Number.POSITIVE_INFINITY
			: phase === 'dealing'
				? cardsDealtTo(myArrayIndex, dealtCards)
				: sequenceDeals
					? 0
					: Number.POSITIVE_INFINITY
	);

	// Game-end dialog, 2 s after the last animation has played out.
	const gameEnded = $derived(Boolean(view.done) && !boardBusy);
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
		if (view.awaitingCapture) {
			isSelectingCapture = false;
		}
	});

	// Tighter offsets for the dealing overlay: four card piles have to fit on a phone screen.
	// These are capped against the viewport, so on a narrow screen the piles close in rather
	// than sliding off the edge. Only ever read as CSS custom properties.
	const FX_OFFSET: Record<string, { tx: string; ty: string }> = {
		down: { tx: '0px', ty: 'min(150px, 20vh)' },
		right: { tx: 'min(210px, 26vw)', ty: '0px' },
		up: { tx: '0px', ty: 'calc(-1 * min(150px, 20vh))' },
		left: { tx: 'calc(-1 * min(210px, 26vw))', ty: '0px' }
	};

	// While a cartón is being counted the live table is already empty, so the cards that were on
	// it come from the round-end payload; while a move is being told, from `tableOverride`.
	const shownTable = $derived<number[]>(heldTable ?? tableOverride ?? view.table ?? []);

	// WHERE A CARD SITS ON THE TABLE. See tableLayout.ts: the first card takes the middle and the
	// rest are laid out around it, each card keeping its place while it is on the table. A capture
	// therefore moves only the cards being taken, and the row never re-centres itself.
	const tableSlots = new TableSlots();

	function slotFor(cardIndex: number): number {
		return slotOffsetUnits(tableSlots.slotFor(cardIndex));
	}

	// The furthest slot in use, in steps from the middle: the row is `2 * half + 1` cards wide, and
	// that is the width the card size has to fit into (see .cards-display).
	const halfSlots = $derived(
		shownTable.reduce((widest, cardIndex) => Math.max(widest, Math.abs(slotFor(cardIndex))), 0)
	);

	// Cards that have left free their slots for the next ones.
	$effect(() => {
		tableSlots.retain(shownTable);
	});
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="game-board" bind:this={boardEl}>
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
						{view.id}
					</span>
				</span>
			</div>
		</div>

		<div class="scores">
			<div class="score-item">
				<span class="label">{t('Team 1')}</span>
				<span class="value">
					{displayedScore(0)}
					{#if cartonGain && cartonGain[0] > 0}
						<span class="score-gain">+{cartonGain[0]}</span>
					{/if}
				</span>
				{#if view.cardsCaptured}
					<span class="carton-info">
						<span class="carton-label">{t('Cartón:')}</span>
						<span class="carton-count">{getCartonCount(view.cardsCaptured[0])}</span>
						<span class="captured-label">({view.cardsCaptured[0]?.length ?? 0})</span>
					</span>
				{/if}
			</div>
			<div class="score-item">
				<span class="label">{t('Team 2')}</span>
				<span class="value">
					{displayedScore(1)}
					{#if cartonGain && cartonGain[1] > 0}
						<span class="score-gain">+{cartonGain[1]}</span>
					{/if}
				</span>
				{#if view.cardsCaptured}
					<span class="carton-info">
						<span class="carton-label">{t('Cartón:')}</span>
						<span class="carton-count">{getCartonCount(view.cardsCaptured[1])}</span>
						<span class="captured-label">({view.cardsCaptured[1]?.length ?? 0})</span>
					</span>
				{/if}
			</div>
		</div>
	</div>

	<!-- Only YOUR turn is announced here: whose turn it is otherwise is lit up on their seat. -->
	<div class="current-player-section">
		{#if isMyTurn && !view.done}
			<div class="current-player-text your-turn">
				<span class="turn-label" aria-live="polite">{t('Your turn')}</span>
			</div>
		{/if}
	</div>

	{#if showGameEndDialog}
		<GameResultsDialog game={view} {onLeaveTable} />
	{/if}

	<!-- The table with the four players around it: you at the bottom, the others left, up and
	     right in turn order. Cards fly between these sides and the table. -->
	<div class="table-ring">
		{#each DIRS as dir (dir)}
			{@const seat = seatAt(dir)}
			<div
				class="seat seat-{dir}"
				class:active={!view.done && view.currentPlayer === seat}
				class:team-a={seat % 2 === 0}
				class:team-b={seat % 2 === 1}
				bind:this={seatEls[dir]}
			>
				<span class="seat-name" title={seatName(seat)}>{seatName(seat)}</span>
				{#if isCurrentUser(seat)}<span class="seat-you">{t('You')}</span>{/if}
				<span class="seat-hand" aria-hidden="true">
					{#each Array(handCount(seat)) as _, i (i)}
						<span class="mini-back"></span>
					{/each}
				</span>
			</div>
		{/each}

		<div class="table-area">
			<div class="table-cards">
				{#if shownTable.length > 0}
					<div class="cards-display" style="--half-slots:{halfSlots}">
						{#each shownTable as cardIndex (cardIndex)}
							<div
								class="card on-table"
								style="--slot:{slotFor(cardIndex)}"
								data-table-card={cardIndex}
								class:captured={isBeingCaptured(cardIndex)}
								class:incoming={incomingCard === cardIndex}
								in:fade={{ duration: 200 }}
							>
								<PlayingCard {cardIndex} />
							</div>
						{/each}
					</div>
				{:else}
					<div class="empty-table">{t('Table is empty')}</div>
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
							<span class="seat-label">{seat === myArrayIndex ? t('You') : seatName(seat)}</span>
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
	</div>

	<!-- Cards travelling between a player's side of the screen and the table. Fixed to the
	     viewport, so a card is visible the whole way, not only once it is over the table. -->
	<div class="fly-layer" bind:this={flyLayer} aria-hidden="true">
		{#each flights as f (f.key)}
			<div
				class="fly-card"
				style="left:{f.base.cx - f.base.w / 2}px; top:{f.base.cy - f.base.h / 2}px; width:{f.base.w}px; height:{f.base.h}px"
				use:flight={f}
			>
				<PlayingCard cardIndex={f.card} />
			</div>
		{/each}
	</div>

	<!-- Capture selection section -->
	{#if view.awaitingCapture && view.captureOptions && view.captureOptions.length > 0 && !isSelectingCapture && !boardBusy}
		<div class="capture-selection">
			<div class="section-title">{t('Choose which cards to capture:')}</div>
			<div class="captures-grid">
				{#each view.captureOptions as option}
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
	{#if myArrayIndex !== undefined && view.cards && view.cards[myArrayIndex]}
		<div class="player-hand" class:my-turn={isMyTurn}>
			<div class="hand-header">
				<div class="hand-label">
					{isMyTurn ? t('Your cards') : t('Your hand')}
					{#if isMyTurn}<span class="hand-hint">{t('Press 1-5 to play a card')}</span>{/if}
				</div>
			</div>
			<div class="cards-container">
				{#each view.cards[myArrayIndex] as cardIndex, i}
					{#if cardIndex >= 0}
						{#if i < myDealtCount}
							<button
								class="card"
								class:playable={canPlayCard()}
								class:just-dealt={phase === 'dealing'}
								class:in-flight={throwing?.card === cardIndex}
								data-hand-card={cardIndex}
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
		display: inline-flex;
		align-items: baseline;
		gap: 0.35rem;
		font-size: 1.5rem;
		font-weight: 600;
		color: white;
		font-family: monospace;
	}

	/* The carton points arriving: a pop and a fade, so the counter says "+6" rather than silently
	   becoming a bigger number. No timer to clear it -- it animates itself out and the next
	   sequence drops it from the DOM. */
	.score-gain {
		font-size: 0.95rem;
		font-weight: 700;
		color: #fbbf24;
		animation: score-gain 2600ms ease-out forwards;
	}

	@keyframes score-gain {
		0% { opacity: 0; transform: translateY(0.4rem) scale(0.8); }
		12% { opacity: 1; transform: translateY(0) scale(1.1); }
		24% { transform: translateY(0) scale(1); }
		70% { opacity: 1; }
		100% { opacity: 0; }
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

	/* Cards in flight between a player's side and the table. Fixed to the viewport so the card is
	   visible the whole way, outside the table too; the clip at the viewport edge is where a card
	   comes from and goes to, and it keeps the travel from widening the page. */
	.fly-layer {
		position: fixed;
		inset: 0;
		overflow: hidden;
		pointer-events: none;
		z-index: 30;
	}

	.fly-card {
		position: absolute;
		border-radius: 0.45rem;
		border: 2px solid rgba(255, 255, 255, 0.3);
		overflow: hidden;
		box-shadow: 0 10px 28px rgba(0, 0, 0, 0.55);
		will-change: transform;
	}

	/* The table and the four players around it: you at the bottom, the others on their sides. */
	.table-ring {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr) auto;
		grid-template-areas:
			'.    up    .'
			'left table right'
			'.    down  .';
		align-items: center;
		justify-items: center;
		gap: 0.5rem;
		min-width: 0;
	}

	.seat {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.2rem;
		min-width: 0;
		max-width: clamp(3.5rem, 18vw, 8rem);
		padding: 0.35rem 0.6rem;
		border-radius: 0.6rem;
		border: 2px solid rgba(255, 255, 255, 0.12);
		background: rgba(15, 23, 42, 0.6);
		transition: border-color 0.3s, box-shadow 0.3s, background 0.3s;
	}

	.seat-up { grid-area: up; }
	.seat-left { grid-area: left; }
	.seat-right { grid-area: right; }
	.seat-down { grid-area: down; }

	.seat.active {
		border-color: #fbbf24;
		background: rgba(251, 191, 36, 0.12);
		box-shadow: 0 0 16px rgba(251, 191, 36, 0.35);
	}

	.seat-name {
		max-width: 100%;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: 0.8rem;
		font-weight: 700;
		color: #e2e8f0;
	}

	.seat.team-a .seat-name { color: #93c5fd; }
	.seat.team-b .seat-name { color: #fca5a5; }
	.seat.active .seat-name { color: #fbbf24; }

	.seat-you {
		font-size: 0.6rem;
		font-weight: 700;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: #94a3b8;
	}

	.seat-hand {
		display: flex;
		gap: 2px;
		min-height: 0.9rem;
	}

	.mini-back {
		width: 0.6rem;
		height: 0.85rem;
		border-radius: 0.15rem;
		border: 1px solid rgba(255, 255, 255, 0.35);
		background:
			repeating-linear-gradient(45deg, rgba(255, 255, 255, 0.16) 0 2px, transparent 2px 4px),
			linear-gradient(135deg, #1e3a8a, #0f172a);
	}

	.table-area {
		grid-area: table;
		width: 100%;
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

	/* `width: 100%` is load-bearing: `.cards-display` sizes its cards against a PERCENTAGE of this
	   box, and the cards inside it are absolutely positioned, so without an explicit width this box
	   shrink-wraps to nothing and the card size collapses with it. */
	.table-cards {
		display: flex;
		justify-content: center;
		align-items: center;
		width: 100%;
		min-width: 0;
	}

	/* The table's own grid: no flex, no wrapping, no centring. Cards are placed from the middle
	   outwards (see tableLayout.ts), so adding or taking one leaves the others exactly where they
	   were -- which is the point, and why this is not a centred flex row. */
	.cards-display {
		position: relative;
		width: 100%;
		--gap: 0.4rem;
		--half-slots: 0;
		/* The card is as large as it can be while still fitting `2 * half-slots + 1` of them (plus a
		   gap between each) inside the table: the SIZE gives way, never the spacing, so cards are
		   never drawn over each other and all of them stay visible. The `max()` is the floor: a card
		   must never shrink into a dot, whatever the arithmetic does. */
		--card-w: max(
			2.1rem,
			min(
				clamp(2.5rem, 9.5vw, 3.4rem),
				calc((100% - 2 * var(--half-slots) * var(--gap)) / (2 * var(--half-slots) + 1))
			)
		);
		--card-h: calc(var(--card-w) * 7 / 5);
		/* Always a whole card plus a gap: cards get smaller as the table fills, they never close in
		   on each other. */
		--step: calc(var(--card-w) + var(--gap));
		height: var(--card-h);
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

	/* Centred by `left/top: 50%` plus a negative margin, and then moved by its slot as a MARGIN --
	   not a transform, which the in/out transitions own. Sharing it would make a card jump to the
	   middle the moment it started leaving the table. */
	.card.on-table {
		position: absolute;
		top: 50%;
		left: 50%;
		width: var(--card-w);
		margin-top: calc(var(--card-h) / -2);
		margin-left: calc((var(--card-w) / -2) + var(--slot) * var(--step));
		cursor: default;
		pointer-events: none;
	}


	/* The cards a capture is about to take, lit up for a beat before they leave. No transform: the
	   in/out transitions animate `transform`, and sharing the property would make the highlight
	   fight the flight. */
	.card.captured {
		border-color: #fbbf24;
		box-shadow:
			0 0 0 3px rgba(251, 191, 36, 0.6),
			0 8px 22px rgba(251, 191, 36, 0.35);
		filter: brightness(1.12);
	}

	/* Holding its place on the table (or in the hand) while its flying copy is on the way. */
	.card.incoming,
	.card.in-flight {
		visibility: hidden;
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
