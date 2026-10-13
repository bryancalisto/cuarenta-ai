#!/usr/bin/env node
/**
 * Static invariants for the deal / cartón choreography in `src/lib/GameBoard.svelte`.
 *
 * WHY THIS EXISTS
 * ---------------
 * The board's cinematic (count the cartón -> pay it -> deal the next hands) must never be
 * able to leave the UI stuck. It already did once: the effect returned a teardown that
 * cancelled its own timers, while websocket snapshots arrive once per move (the AI's
 * included) and re-run the effect; the re-run hit the "nothing pending" guard and returned
 * early, so the teardown fired and the sequence was orphaned mid-flight. `phase` stayed
 * 'dealing' forever, which renders every not-yet-dealt card as a placeholder -- no hand, no
 * clickable cards, a full-screen overlay. That reads as a frozen app.
 *
 * These are the three properties that prevent it. There is no DOM test harness in this repo
 * (no jsdom, no browser in CI), so this is a cheap source-level guard; it is a lint for a
 * known hazard, not a substitute for opening the page.
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
const lines = source.split('\n');

const failures = [];
const check = (ok, message) => {
	if (!ok) failures.push(message);
};

/**
 * Slice out one `$effect(() => { ... })` body by brace matching, starting at the first line
 * matching `marker`. Returns the body text plus the line it starts on.
 */
function effectBody(marker) {
	const hit = lines.findIndex((line) => marker.test(line));
	if (hit === -1) return null;
	// The marker sits *inside* the effect, so walk back to its `$effect(() => {` opener:
	// brace counting has to start there to span the whole body.
	let start = hit;
	while (start > 0 && !/\$effect\s*\(/.test(lines[start])) start--;
	let depth = 0;
	for (let i = start; i < lines.length; i++) {
		for (const char of lines[i]) {
			if (char === '{') depth++;
			else if (char === '}') depth--;
		}
		// The effect's own arrow-function body closes when depth returns to 0.
		if (depth === 0 && i > start) {
			return { body: stripComments(lines.slice(start, i + 1).join('\n')), line: start + 1 };
		}
	}
	return null;
}

/**
 * Drop comments before matching, so a comment that merely *describes* an invariant cannot
 * satisfy a check for its code. `//` is left alone when it follows a colon, which keeps URLs
 * (`https://...`) intact.
 */
function stripComments(code) {
	return code.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|\s)\/\/[^\n]*/g, '$1');
}

// 1. The choreography effect must NOT cancel its timers on teardown. It owns them until the
//    sequence ends (its last step is `phase = 'idle'`), and orphaning on unmount is handled
//    by bumping `seqId` in `onDestroy` instead.
const choreography = effectBody(/const\s+dealId\s*=\s*game\.dealId/);
check(choreography !== null, 'could not find the choreography effect (looked for `const dealId = game.dealId`)');
if (choreography) {
	const { body, line } = choreography;
	check(
		!/return\s*\(\s*\)\s*=>/.test(body) && !/return\s+\(\)\s*=>\s*\{?\s*(timers|clearTimeout)/.test(body),
		`GameBoard.svelte:${line}: the choreography effect returns a teardown. Snapshots re-run this ` +
			'effect mid-animation and the pending-work guard returns early, so a teardown that cancels ' +
			'timers orphans the sequence and wedges the board in "dealing"/"carton-paid".'
	);
	// 2. Ownership + a hard deadline: a new sequence orphans the old one, and every sequence
	//    guarantees a return to idle even if a step is somehow missed.
	check(
		/const\s+alive\s*=[^;]*seqId\s*===\s*id/.test(body) && /if\s*\(\s*alive\(\)\s*\)/.test(body),
		`GameBoard.svelte:${line}: the choreography effect lost its sequence-ownership guard: it must define ` +
			'`alive` in terms of `seqId === id` AND consult it before every callback, so a newer sequence ' +
			'orphans an older one and unmounting stops dead callbacks from writing state.'
	);
	check(
		/SEQUENCE_CAP_MS/.test(body),
		`GameBoard.svelte:${line}: the choreography effect no longer schedules the SEQUENCE_CAP_MS deadline, ` +
			'so a missed step can leave the overlay up indefinitely.'
	);
}

// 3. The move animation mirrors the table on snapshots without a move, and re-running it
//    cancels the previous throw's timers -- so it must also clear the thrown card.
const moveEffect = effectBody(/const\s+table\s*=\s*game\.table/);
check(moveEffect !== null, 'could not find the move-animation effect (looked for `const table = game.table`)');
if (moveEffect) {
	const earlyReturn = moveEffect.body.split('if (!move || currentPlayer === undefined)')[1] ?? '';
	check(
		/throwAnim\s*=\s*null/.test(earlyReturn.slice(0, 400)),
		`GameBoard.svelte:${moveEffect.line}: the no-move branch of the move effect must set \`throwAnim = null\`, ` +
			'otherwise a cancelled throw leaves the card stuck on the board.'
	);
}

/**
 * Pull the deal helpers out of the component and evaluate them. They are pure functions of
 * their arguments, so the deal order can be checked by *running* the real code rather than by
 * re-reading the markup. Type annotations are stripped; if a rename breaks the extraction,
 * that is reported as a failure instead of silently skipping the checks.
 */
function loadDealHelpers() {
	const constants = source.match(/const SEAT_COUNT = 4;[\s\S]*?const SEQUENCE_CAP_MS = [^;]+;/);
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

// 5. One animation at a time. The snapshot that ends a deal also carries the move that ended
//    it, so the counting/dealing must wait for that move's throw/grab to land -- otherwise two
//    cinematics play on top of each other. The choreography derives the wait and folds it into
//    every scheduled step; the move effect publishes when its animation ends.
if (choreography) {
	const { body, line } = choreography;
	check(
		/moveAnimationMs\s*\(\s*game\.move\s*\)/.test(body) && /moveAnimUntil/.test(body),
		`GameBoard.svelte:${line}: the choreography effect must wait for the move animation (it reads ` +
			'`moveAnimationMs(game.move)` and falls back to `moveAnimUntil`) before it starts counting.'
	);
	check(
		/setTimeout\s*\([\s\S]{0,120}?settleMs\s*\+/.test(body),
		`GameBoard.svelte:${line}: the wait must be applied inside the scheduler (\`settleMs + delay\`), ` +
			'so every step and the deadline shift together instead of racing the move animation.'
	);
}
if (moveEffect) {
	check(
		/moveAnimUntil\s*=\s*performance\.now\(\)\s*\+/.test(moveEffect.body),
		`GameBoard.svelte:${moveEffect.line}: the move effect must publish when its animation ends ` +
			'(`moveAnimUntil = performance.now() + ...`), or the choreography cannot know when to start.'
	);
	check(
		/moveAnimUntil\s*=\s*0/.test(moveEffect.body),
		`GameBoard.svelte:${moveEffect.line}: the move effect must release the board (\`moveAnimUntil = 0\`) ` +
			'both when its animation finishes and when a snapshot brings no move.'
	);
}

// 6. One animation at a time, and nothing playable during a cinematic. The sequence starts by
//    waiting for the move that triggered it, freezes input from that first moment (not only once
//    an overlay is up), and the move effect animates only the move the sequence waits for --
//    anything later is mirrored, so a throw cannot land on top of the counting or the deal.
if (choreography) {
	const { body, line } = choreography;
	check(
		/phase\s*=\s*'waiting'/.test(body),
		`GameBoard.svelte:${line}: the sequence must start in the 'waiting' phase, which freezes input ` +
			'and holds moves back while the triggering move animation finishes.'
	);
	check(
		/sequenceMoveStep\s*=/.test(body),
		`GameBoard.svelte:${line}: the sequence must record the move step it is waiting for ` +
			'(`sequenceMoveStep`), so the move effect can tell it apart from a move that arrives mid-cinematic.'
	);
}
if (moveEffect) {
	check(
		/boardBusy\s*&&\s*step\s*!==\s*sequenceMoveStep/.test(moveEffect.body),
		`GameBoard.svelte:${moveEffect.line}: while the board is busy the move effect may animate ONLY the ` +
			'move the sequence is waiting for; every other move must be mirrored, or two animations overlap.'
	);
	check(
		/handledMove\s*=\s*step/.test(moveEffect.body) && /step\s*===\s*handledMove/.test(moveEffect.body),
		`GameBoard.svelte:${moveEffect.line}: the move effect must animate each move once ('handledMove'), so a ` +
			'mirrored or already-shown move is not replayed when the cinematic ends.'
	);
}

// 7. Input is frozen for the whole cinematic, through every entry point. These look at the whole
//    file, comments stripped: a comment that merely mentions `boardBusy` must not pass.
const code = stripComments(source);
check(
	/return\s+!boardBusy\s*&&\s*game\.currentPlayer\s*===\s*myArrayIndex/.test(code),
	'`canPlayCard()` must be false while the board is busy, so the hand cannot be played mid-animation.'
);
const keydown = code.match(/function handleKeydown\(event: KeyboardEvent\) \{[\s\S]*?\n\t\}/)?.[0] ?? '';
check(
	/boardBusy/.test(keydown),
	'the keyboard handler must ignore number keys while the board is busy (the mouse path is gated by canPlayCard).'
);
check(
	/game\.awaitingCapture[\s\S]{0,160}!boardBusy/.test(code),
	'the capture prompt must not be offered while a cinematic is playing: the choice is sent to the server ' +
		'and would land on top of the animation.'
);

// 8. A capture has to be readable. It is a story: the played card lands, the cards it takes are
//    lit up for a beat, then they leave one after another. All at once at 900ms read as a
//    scuffle, which is what this pins against.
const captureHold = source.match(/const CAPTURE_HOLD_MS = (\d+)/)?.[1];
const captureStagger = source.match(/const CAPTURE_STAGGER_MS = (\d+)/)?.[1];
const throwMs = source.match(/const THROW_MS = (\d+)/)?.[1];
check(
	Number(captureHold) > 0 && Number(captureStagger) > 0,
	'the capture needs a hold beat (CAPTURE_HOLD_MS) and a per-card stagger (CAPTURE_STAGGER_MS): ' +
		'without them every captured card leaves in the same instant and the move cannot be followed'
);
check(
	/captureHold = captures\.length > 0/.test(moveEffect?.body ?? ''),
	`GameBoard.svelte:${moveEffect?.line}: the move effect must raise the capture highlight when the played ` +
		'card lands, before anything leaves the table'
);
// The clear has to be in the GRAB step specifically: the busy/no-move branches clear it too, and
// a check for the bare assignment would be satisfied by those while the highlight stuck on screen.
check(
	/captureHold = false;[\s\S]{0,80}?removeDir = dir;/.test(moveEffect?.body ?? ''),
	'the capture highlight must be dropped in the step that removes the cards (next to `removeDir = dir`), ' +
		'or the lit-up cards stay lit after they have gone'
);
check(
	/captured=\{isBeingCaptured\(cardIndex\)\}/.test(code) && /\.card\.captured \{/.test(code),
	'the cards a capture takes must be highlighted on the table (class + a .card.captured rule)'
);
check(
	/out:flyToSide=\{\{ dir: removeDir, delay: departureDelayMs\(cardIndex\) \}\}/.test(code),
	'the captured cards must leave with a per-card delay (departureDelayMs), not all at once'
);
check(
	/delay,\n\t\t\teasing: cubicIn/.test(code),
	'flyToSide must pass its delay to the transition, or the stagger is silently ignored'
);
// The staggering is part of the animation, so the board has to stay busy for all of it.
check(
	/(?:captured|captures)\s*\*\s*CAPTURE_STAGGER_MS|CAPTURE_STAGGER_MS\s*\*\s*(?:captured|captures)/.test(code),
	'moveAnimationMs must include the staggered departures: the deal would otherwise start on top of them'
);
// The CSS throw has to last exactly as long as the constant the state machine waits for.
const cssThrow = source.match(/animation: throw-in (\d+)ms/)?.[1];
check(
	cssThrow !== undefined && cssThrow === throwMs,
	`the throw-in animation is ${cssThrow}ms but THROW_MS is ${throwMs}ms: the card would land before (or after) the state says it did`
);

if (failures.length > 0) {
	console.error('check-board-choreography: FAILED');
	for (const failure of failures) console.error(`  - ${failure}`);
	process.exit(1);
}
console.log(
	'check-board-choreography: OK (choreography owns its timers, waits for the move animation, deals ' +
		'five cards per player before moving on, and freezes play + moves while a cinematic runs)'
);
