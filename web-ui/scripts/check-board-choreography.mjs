#!/usr/bin/env node
/**
 * Static invariants for the board's animations in `src/lib/GameBoard.svelte`.
 *
 * WHY THIS EXISTS
 * ---------------
 * The backend publishes one snapshot per move and paces the AI faster than a capture takes to
 * show. When the board reacted to each snapshot directly, animations overlapped: a throw landed
 * on top of the previous capture, and the cards of ONE capture flew partly to one player and
 * partly to another (each departure read whatever direction was current when it started). Before
 * that, an effect that cancelled its own timers wedged the board in "dealing" for good.
 *
 * The fix is structural: snapshots are QUEUED and presented one at a time by a single async loop,
 * and the board renders `view` (the snapshot whose story has been told), never `game`. These
 * checks pin that structure. There is no DOM test harness in this repo (no jsdom, no browser in
 * CI), so this is a cheap source-level guard; it is a lint for known hazards, not a substitute for
 * opening the page.
 *
 * Run with `npm run check`.
 */
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
// Optional path argument so the guard itself can be tested against a mutated copy.
const file = process.argv[2] ?? resolve(here, '../src/lib/GameBoard.svelte');
const source = readFileSync(file, 'utf8');

const failures = [];
const check = (ok, message) => {
	if (!ok) failures.push(message);
};

/**
 * Drop comments before matching, so a comment that merely *describes* an invariant cannot
 * satisfy a check for its code. `//` is left alone when it follows a colon, which keeps URLs
 * (`https://...`) intact.
 */
function stripComments(code) {
	return code
		.replace(/<!--[\s\S]*?-->/g, '')
		.replace(/\/\*[\s\S]*?\*\//g, '')
		.replace(/(^|\s)\/\/[^\n]*/g, '$1');
}

const code = stripComments(source);
const scriptEnd = code.indexOf('</script>');
const script = code.slice(0, scriptEnd);
const markup = code.slice(scriptEnd);

/** The body of `function name(...) {...}` (or `async function`), by brace matching. */
function functionBody(name) {
	const at = script.search(new RegExp(`(?:async\\s+)?function ${name}\\(`));
	if (at < 0) return '';
	let depth = 0;
	for (let i = script.indexOf('{', script.indexOf(')', at)); i < script.length; i++) {
		if (script[i] === '{') depth++;
		else if (script[i] === '}' && --depth === 0) return script.slice(at, i + 1);
	}
	return '';
}

// 1. ONE SNAPSHOT AT A TIME. Every snapshot is queued; a single loop presents them in order and
//    awaits each one before the next, and a failure shows the snapshot rather than wedging.
check(
	/queue\.push\(next\)/.test(script) && /if \(!draining\) void drain\(\)/.test(script),
	'snapshots must be queued (`queue.push(next)`) and drained by a single loop (`if (!draining) void drain()`)'
);
const drain = functionBody('drain');
check(
	/while \(queue\.length > 0[\s\S]*?await present\(next\)/.test(drain),
	'`drain()` must await `present(next)` for each snapshot in turn: that await is what keeps two moves from animating at once'
);
check(
	/catch[\s\S]*?resetChoreography\(\);[\s\S]*?view = next/.test(drain),
	'`drain()` must catch a failed presentation, reset the choreography and show the snapshot, or one broken animation freezes the board'
);
check(
	/finally \{\s*draining = false;/.test(drain),
	'`drain()` must clear `draining` in a finally block, or the queue stops for good after an error'
);

// 2. The board renders `view`, never `game`: reading `game` in the markup shows a snapshot before
//    its move has been animated.
check(
	!/\bgame\./.test(markup) && !/\{game\}/.test(markup),
	'the markup reads `game` directly; it must read `view`, the snapshot whose animation has played'
);

// 3. A move's card comes from, and its capture goes to, the player NAMED BY THE MOVE, and every
//    card of a capture goes the same way.
const playMove = functionBody('playMove');
check(
	/const seat = move\.playerIndex;/.test(playMove) && /const dir = dirFor\(seat\);/.test(playMove),
	'`playMove` must take the direction from `move.playerIndex`, not from whose turn it is now'
);
check(
	!/currentPlayer/.test(playMove),
	'`playMove` must not look at `currentPlayer`: it has already moved on to the next player'
);
check(
	/leaving\.forEach[\s\S]*?fly\(c, base, base, \{ \.\.\.edgeBox\(dir, base\)/.test(playMove),
	'every card a capture takes must fly to `edgeBox(dir, ...)` of the one direction of the move'
);
check(
	/delay: i \* CAPTURE_STAGGER_MS/.test(playMove),
	'the captured cards must leave one after another (`delay: i * CAPTURE_STAGGER_MS`), not all at once'
);
check(
	/heldCapture = captures;[\s\S]*?await sleep\(CAPTURE_HOLD_MS\);[\s\S]*?heldCapture = \[\];/.test(playMove),
	'a capture must light up the cards it takes for CAPTURE_HOLD_MS before they leave'
);
check(
	/await Promise\.all\(grabs\)/.test(playMove),
	'`playMove` must wait for every captured card to arrive before the next snapshot is presented'
);
const captureHold = source.match(/const CAPTURE_HOLD_MS = (\d+)/)?.[1];
const captureStagger = source.match(/const CAPTURE_STAGGER_MS = (\d+)/)?.[1];
check(
	Number(captureHold) > 0 && Number(captureStagger) > 0,
	'the capture needs a hold beat (CAPTURE_HOLD_MS) and a per-card stagger (CAPTURE_STAGGER_MS)'
);
check(
	/captured=\{isBeingCaptured\(cardIndex\)\}/.test(markup) && /\.card\.captured \{/.test(code),
	'the cards a capture takes must be highlighted on the table (class + a .card.captured rule)'
);

// 4. Order inside one snapshot: move, then counting, then deal.
const present = functionBody('present');
const moveAt = present.indexOf('await playMove(');
const countAt = present.indexOf('await playCounting(');
const dealAt = present.lastIndexOf('await playDeal(');
check(
	moveAt > 0 && countAt > moveAt && dealAt > countAt,
	'`present()` must play the move, then the counting, then the deal, each awaited'
);

/**
 * Pull the deal helpers out of the component and evaluate them. They are pure functions of
 * their arguments, so the deal order can be checked by *running* the real code rather than by
 * re-reading the markup. Type annotations are stripped; if a rename breaks the extraction,
 * that is reported as a failure instead of silently skipping the checks.
 */
function loadDealHelpers() {
	const constants = source.match(/const SEAT_COUNT = 4;[\s\S]*?const DEAL_HOLD_MS = [^;]+;/);
	const functions = ['cardsDealtTo', 'seatOfCard', 'dealDelayMs'].map((name) =>
		source.match(new RegExp(`function ${name}[\\s\\S]*?\\n\\t\\}`))
	);
	if (!constants || functions.some((fn) => !fn)) return null;
	const strip = (code) => code.replace(/:\s*(number|string|boolean|void|Phase)\b/g, '');
	const body = `${strip(constants[0])}\n${strip(functions.join('\n'))}
		return { SEAT_COUNT, HAND_SIZE, HAND_DEAL_MS, HAND_GAP_MS, DEAL_TOTAL_MS, cardsDealtTo, seatOfCard, dealDelayMs };`;
	try {
		return new Function(body)();
	} catch {
		return null;
	}
}

// 4. The deal order: five cards to one player, then the next, not one card at a time around
//    the table. This is a product decision, so it is pinned here rather than left to the
//    animation to reveal.
const deal = loadDealHelpers();
check(
	deal !== null,
	'could not evaluate the deal helpers (cardsDealtTo / seatOfCard / dealDelayMs). If they were ' +
		'renamed or given non-primitive annotations, update this check to match.'
);
if (deal) {
	const total = deal.SEAT_COUNT * deal.HAND_SIZE;

	// The animation replays a deal the backend performed, so its idea of the deal must be the
	// backend's: read the constants straight out of the Go source instead of trusting a copy.
	const rules = readFileSync(resolve(here, '../../backend/internal/game/cuarenta.go'), 'utf8');
	const goSeats = Number(rules.match(/\bNumPlayers\s*=\s*(\d+)/)?.[1]);
	const goHand = Number(rules.match(/\bHandSize\s*=\s*(\d+)/)?.[1]);
	check(
		goSeats > 0 && goHand > 0,
		'could not read NumPlayers/HandSize from backend/internal/game/cuarenta.go, so the board cannot ' +
			'be checked against the deal it animates.'
	);
	check(
		deal.SEAT_COUNT === goSeats && deal.HAND_SIZE === goHand,
		`the board deals ${deal.HAND_SIZE} cards to each of ${deal.SEAT_COUNT} seats, but the backend ` +
			`deals ${goHand} to each of ${goSeats} (backend/internal/game/cuarenta.go).`
	);

	const order = [];
	for (let dealt = 1; dealt <= total; dealt++) {
		order.push(deal.seatOfCard(dealt));
		const before = Array.from({ length: deal.SEAT_COUNT }, (_, s) => deal.cardsDealtTo(s, dealt - 1));
		const after = Array.from({ length: deal.SEAT_COUNT }, (_, s) => deal.cardsDealtTo(s, dealt));
		const gained = after.map((n, s) => n - before[s]);
		const served = gained.flatMap((n, s) => (n === 1 ? [s] : []));
		check(
			served.length === 1 && served[0] === order[dealt - 1] && gained.every((n) => n === 0 || n === 1),
			`the card dealt as number ${dealt} must go to the one seat whose count grows, and ` +
				'`seatOfCard` must name that seat.'
		);
		check(
			after[order[dealt - 1]] <= deal.HAND_SIZE,
			`seat ${order[dealt - 1]} ends up with more than ${deal.HAND_SIZE} cards (dealt ${dealt}).`
		);
	}
	const expected = Array.from({ length: deal.SEAT_COUNT }, (_, s) => Array(deal.HAND_SIZE).fill(s)).flat();
	check(
		order.join(',') === expected.join(','),
		`the deal must hand ${deal.HAND_SIZE} cards to one player before moving to the next, in seat ` +
			`order (expected ${expected.join(',')}, got ${order.join(',')}).`
	);
	// The beat between players is what makes the groups read as hands; without it the deal is
	// just a fast round of singles again. Cards inside a group stay HAND_DEAL_MS apart, and the
	// step across a group boundary is that plus the gap.
	const lastCard = total - 1;
	check(deal.HAND_GAP_MS > 0, 'HAND_GAP_MS must be positive: the beat between players is what groups the hands.');
	for (let k = 1; k < total; k++) {
		const step = deal.dealDelayMs(k) - deal.dealDelayMs(k - 1);
		const crossesGroup = k % deal.HAND_SIZE === 0;
		check(
			step === deal.HAND_DEAL_MS + (crossesGroup ? deal.HAND_GAP_MS : 0),
			`the delay before card ${k + 1} is ${step}ms; expected ${deal.HAND_DEAL_MS}ms within a ` +
				`player's five cards${crossesGroup ? `, or ${deal.HAND_DEAL_MS + deal.HAND_GAP_MS}ms when a new player starts` : ''}.`
		);
	}
	check(
		deal.dealDelayMs(lastCard) <= deal.DEAL_TOTAL_MS,
		'DEAL_TOTAL_MS must not elapse before the last card lands: the settle time and the idle flip hang off it.'
	);
}


// 5. Input is frozen while anything is animating OR still queued, through every entry point.
check(
	/const boardBusy = \$derived\(phase !== 'idle' \|\| animating \|\| queued > 0\)/.test(script),
	'`boardBusy` must cover the cinematics, a move being animated, and snapshots still waiting in the queue'
);
check(
	/return\s+!boardBusy\s*&&\s*view\.currentPlayer\s*===\s*myArrayIndex/.test(script),
	'`canPlayCard()` must be false while the board is busy, so the hand cannot be played mid-animation.'
);
const keydown = functionBody('handleKeydown');
check(/boardBusy/.test(keydown), 'the keyboard handler must ignore number keys while the board is busy.');
check(
	/view\.awaitingCapture[\s\S]{0,160}!boardBusy/.test(markup),
	'the capture prompt must not be offered while the board is busy.'
);

// 6. The carton points land on the score board when the COUNTING CLOSES, not when the snapshot
//    arrives (the backend already added them). They are held back before the snapshot goes on
//    screen and released with the overlay.
check(
	/function displayedScore\(team: number\): number \{[\s\S]*?Math\.max\(0, \(view\.scores\?\.\[team\] \?\? 0\) - \(heldCartonPoints\?\.\[team\] \?\? 0\)\)/.test(script),
	'the score board must show `view.scores` MINUS the pending carton points (and never below zero)'
);
check(
	/class="value">[\s\S]{0,200}?displayedScore\(0\)/.test(markup) && /class="value">[\s\S]{0,200}?displayedScore\(1\)/.test(markup),
	'both score counters must render displayedScore()'
);
check(
	/heldCartonPoints = \[roundEnd\.carton[\s\S]*?view = next;/.test(present),
	'the carton points must be held back BEFORE the snapshot that carries them goes on screen'
);
check(
	/cartonEvent = null;\s*\n\s*heldCartonPoints = null;\s*\n\s*cartonGain = paid/.test(functionBody('playCounting')),
	'the points must be released in the same step that closes the counting overlay'
);
check(
	/heldCartonPoints = null;/.test(functionBody('resetChoreography')),
	'resetting the choreography must release the held points too, or the score stays short for good'
);
check(
	/cartonGain = null;/.test(present) && /\.score-gain \{[\s\S]*?animation: score-gain/.test(code),
	'the "+N" must be cleared when the next sequence starts and animate itself away'
);

// 7. Flying cards live in a layer fixed to the viewport, so they are visible outside the table.
check(
	/\.fly-layer \{[\s\S]*?position: fixed;/.test(code) && /class="fly-layer"/.test(markup),
	'flying cards must be drawn in `.fly-layer` (position: fixed), or they are clipped to the table'
);

if (failures.length > 0) {
	console.error('check-board-choreography: FAILED');
	for (const failure of failures) console.error(`  - ${failure}`);
	process.exit(1);
}
console.log(
	'check-board-choreography: OK (snapshots are presented one at a time, captures fly to the mover, ' +
		'deals five cards per player before moving on, and play is frozen while anything animates)'
);
