# sv

Everything you need to build a Svelte project, powered by [`sv`](https://github.com/sveltejs/cli).

## Creating a project

If you're seeing this, you've probably already done this step. Congrats!

```sh
# create a new project
npx sv create my-app
```

To recreate this project with the same configuration:

```sh
# recreate this project
bun x sv@0.16.1 create --template minimal --types ts --add tailwindcss="plugins:none" --install bun web-ui
```

## Developing

Once you've created a project and installed dependencies with `npm install` (or `pnpm install` or `yarn`), start a development server:

```sh
npm run dev

# or start the server and open the app in a new browser tab
npm run dev -- --open
```

Checks (all of them run in one go with `npm run check`):

| command | what it protects |
|---|---|
| `npm run check` | `svelte-check` + both checks below |
| `npm run check:i18n` | every `t()` key has a translation, none unused, placeholders match |
| `npm run check:choreography` | the deal/cartón effects stay unable to wedge the board (see below) |

## Building

To create a production version of your app:

```sh
npm run build
```

You can preview the production build with `npm run preview`.

> To deploy your app, you may need to install an [adapter](https://svelte.dev/docs/kit/adapters) for your target environment.

## Game board effects

`src/lib/GameBoard.svelte` choreographs the table from the snapshots the backend publishes.
Everything is driven by two server events rather than by guessing from the state:

| event | when | what the client does |
|---|---|---|
| `dealId` changes | the opening deal, a redeal from the deck, or the deal after a cartón | deal animation: the deck empties into four piles **five cards to one player, then the next** (matching `deal`/`redeal` in `backend/internal/game/cuarenta.go`, which copy `HandSize` cards per seat in seat order), with a short beat between players; your own hand turns up as its group lands |
| `roundEnd` present | the ONE snapshot of the move that ended a deal and paid the cartón | count animation: the two captured piles are counted card by card (face-down cards turn up), the table keeps showing the cards that were there, then the cartón points are shown before the next deal |

Both are needed because the state moves on inside the same step: by the time a client sees a
snapshot the piles are cleared, so `roundEnd` carries them (and the cards that were on the
table). When a cartón ends a deal the two events arrive together, and the board sequences
them: **count → cartón paid → deal**.

The deal itself is only the animation — there is no banner announcing it, because the cards
flying to each seat already say it. The *counting* does keep its banner: it is the only place
the points are explained.

Timings live in the `Timings` constants at the top of the component; the backend paces AI
turns 1.5-3 s apart, so the whole sequence is kept near 3 s.

### The one rule for these effects: a sequence owns its timers

A snapshot arrives per move — the AI's included, i.e. *during* an animation — and each one
re-runs the effect, which then hits the "nothing pending" guard and returns early. So a
teardown that cancelled the sequence's timers would kill it mid-flight while `phase` stayed
`dealing`: every not-yet-dealt card renders as a non-interactive placeholder, the overlay
stays up, and the board is unplayable. That is the freeze this code has already shipped once.

Three things keep it out, and `scripts/check-board-choreography.mjs` (wired into
`npm run check`) fails the build if any of them is dropped:

1. the choreography `$effect` has **no teardown** — it owns its timers until the sequence's
   last step sets `phase = 'idle'`;
2. every callback checks `alive()` (`seqId === id`), so a newer sequence orphans an older one
   and `onDestroy` stops otherwise-dead callbacks from writing state;
3. each sequence schedules a `SEQUENCE_CAP_MS` deadline that forces `phase = 'idle'`, so even
   a missed step cannot hold the UI hostage.

The move/throw effect is the mirror image: it *does* have a teardown (a re-run should restart
the throw), so its no-move branch must drop `throwAnim`, or a cancelled throw leaves the card
stuck on the board forever.

### Capture pacing

A capture is the one move with a lot happening at once, so it is paced as a story rather than as
one gesture (`THROW_MS`, `CAPTURE_HOLD_MS`, `CAPTURE_STAGGER_MS`, `GRAB_MS` in the component):

1. **the played card flies in** (`THROW_MS`, 1150ms) — kept in step with the `throw-in` animation
   in the CSS, which the check enforces;
2. **a beat, with the cards it takes lit up** (`CAPTURE_HOLD_MS`, 500ms): `.card.captured` rings
   them in amber before anything moves, so the move says *which* cards it takes;
3. **they leave in sequence** — the played card first, then each captured card
   `CAPTURE_STAGGER_MS` (150ms) behind the last, each taking `GRAB_MS` (1250ms) to reach the
   player. `departureDelayMs` feeds that stagger into the `out:flyToSide` transition.

A capture is therefore ~3.2s (a two-card one) instead of the ~2s it was, and no two cards start
moving in the same frame. `moveAnimationMs` includes the stagger, because the board is busy for
all of it: a deal or cartón sequence scheduled behind a capture waits for the last card to land.

**One animation at a time.** The snapshot that ends a deal also carries the move that ended it,
so the sequence waits for that move's throw/grab to land: it takes `moveAnimationMs(game.move)`
when the payload has the move, and otherwise the remaining time until `moveAnimUntil` (published
by the move effect), and folds that wait into every scheduled step. Without it the captured
cards are still flying to their player while the piles are already being counted.

The board is *busy* for that whole stretch, not just while an overlay is up: the sequence starts
in the `waiting` phase (nothing shown, the triggering move is still animating) and only then
moves to `counting`/`carton-paid`/`dealing`. While busy:

* `canPlayCard()`, the keyboard handler and the capture prompt are all off — a turn started now
  would be overwritten by the deal that is already on its way;
* the move effect mirrors incoming moves instead of animating them, so an AI card cannot fly
  across a deal in progress. The one exception is the move the sequence is waiting for
  (`sequenceMoveStep`), which *must* animate because the sequence's first step hangs off it;
* a hand that arrived with the deal stays face down (`myDealtCount` is 0) until the deal
  animation turns it over.

The server cooperates through `AI_DEAL_GRACE_MS` (`backend/.env`): the first AI move of a deal is
held back so the cards have landed before anyone plays. Keep the two in step — that value is
`DEAL_TOTAL_MS + DEAL_HOLD_MS` here. The client gating is what guarantees the rule; the grace is
what keeps the AI's move visible and animated instead of silently applied during the animation.

### The cartón reaches the score board when the counting closes

The backend pays the cartón **in the same step** that publishes `roundEnd` (`cuarenta.go` adds it to
`Scores` and then reports it, pinned by `TestDealCounterAndRoundEndEvent`). So the snapshot the board
receives already carries the final score, and rendering it straight away gave the counting away
before the overlay performed it.

The board therefore holds the points back for the whole sequence, `waiting` included:

* `heldCartonPoints` is set from `roundEnd.carton` when the sequence is scheduled;
* `displayedScore(team)` = `game.scores[team] - heldCartonPoints[team]`, floored at zero, and both
  counters render *that* rather than `game.scores`;
* the hold is released in the same step that clears `cartonEvent` — the moment the counting overlay
  closes — where the points also get a "+N" burst next to the score (`.score-gain`, which animates
  itself out and is dropped by the next sequence);
* the `SEQUENCE_CAP_MS` deadline releases it too, so a missed step cannot leave the score short for
  good.

### Deal order

`HAND_SIZE` cards to one player before moving to the next: your five cards turn up as a group,
not one card at a time around the table. The grouping is `seatOfCard(dealt)` /
`cardsDealtTo(seat, dealt)` in the component, plus `HAND_GAP_MS`, the beat that makes one
player's five read as a hand. `check-board-choreography.mjs` evaluates those helpers and pins
the resulting order, and reads `NumPlayers`/`HandSize` out of the Go source so the animation
cannot drift from the deal it is replaying.

The Ecuador/Quito mark in the headers is `static/images/ecuador-quito-svgrepo-com.svg`, used
as a CSS **mask** painted with the theme's amber (`.culture-icon` in `src/routes/layout.css`)
so one flat-black asset stays crisp and follows the palette.

## Languages

The UI ships in English (default) and Spanish, switchable from the dropdown that the header
shows on every screen. It is a ~150-line module, `src/lib/i18n.svelte.ts`, with no dependency:
a locale, a dictionary, and `t()`.

**Keys are the English text** (`t('Sign in')` → `'Iniciar sesión'`). Consequences worth
knowing:

* a missing translation renders the English text, never a raw identifier;
* the English locale needs no data — it is the identity map;
### Persistence

The choice lives in `localStorage` under `cuarenta-locale` (exported as `LOCALE_STORAGE_KEY`), so it
survives a reload:

* `setLocale()` (what the dropdown calls) switches, writes the key, and keeps `<html lang>` in step;
* `initLocale()` runs on mount, restores the stored value, and — when there is nothing stored —
  falls back to `navigator.language` and *stores that too*, so a browser-detected choice is
  remembered like one made in the dropdown;
* everything else stays English: `initLocale` is client-only (the server renders English, which is
  what hydration expects), junk in storage is ignored, and storage that throws (private mode, a
  sandboxed frame) is survived with a one-off console warning instead of failing silently — a
  language that will not stay put is otherwise indistinguishable from a bug in the UI.

To check it by hand: pick Español, reload, and read `localStorage.getItem('cuarenta-locale')` in
DevTools. `scripts/check-locale-persistence.mjs` (part of `npm run check`) does the same thing
automatically: it compiles the module and runs it against a fake browser, covering a fresh visit,
a stored choice surviving a reload, junk in storage, a throwing `localStorage`, and the server.

* renaming an English string orphans its translation, so `npm run check:i18n` (also part of
  `npm run check`) verifies that every `t()` key has a Spanish entry, that no entry is
  unused, and that `{placeholders}` match on both sides.

Placeholders: `t('Cartón paid: Team 1 +{first} · Team 2 +{second}', { first: 6, second: 0 })`.
Game vocabulary (caída, limpia, cartón) stays in Spanish in both locales — that is the game's
own language. The choice persists in `localStorage` and the initial guess comes from
`navigator.language`; `<html lang>` follows it, because screen readers and hyphenation use it.

Adding a language is: add it to `LOCALES`, add its dictionary next to `ES`, and extend the
checker to walk both dictionaries.

## Cards

`src/lib/PlayingCard.svelte` draws a card face the way a real card is drawn:

* **portrait** — the frame in `GameBoard.svelte` is a `5 / 7` box and the face fills it, so a card
  is taller than it is wide at every size (hand, table, flying card);
* the **rank over the suit in two opposite corners**, the lower pair rotated 180°: hold the card
  either way up and one of the two indices reads correctly;
* the **suit pips in their traditional positions per rank** (`PIP_LAYOUT`): one oversized pip for
  the A, a centre column for 2-3, two columns for 4-7, with the lower half mirrored. Counting
  identical symbols is what the layout saves you from;
* **court cards** (J/Q/K) carry a figure instead of pips — two halves facing each other, like a
  double-headed court card;
* every card carries an `aria-label` ("6 of Diamonds", via `{rank} of {suit}`), because a pip
  layout says nothing out loud.

Everything inside the face is sized in `cqw` (container query units) off the card's own width, so
one component draws a hand card and a table card: the parent picks the size, the component draws
the face. Nothing there is in `rem`, or the corner indices would be unreadable on a small card.

`scripts/check-card-face.mjs` (part of `npm run check`) compiles the component for SSR and renders
all 40 cards, then asserts exactly these rules — two corner indices, the pip count and columns per
rank, mirrored lower pips (the CSS, not just the class), court figures without pips, the 5/7
frames, and a card that names itself. A pip layout rots silently otherwise: a missing pip still
looks like "a card" in a screenshot.

The card *back* (`card-back` in `GameBoard.svelte`) is the one card shape the face does not draw:
it paints its own texture, but on the same `5 / 7` box, so a dealt card is the shape of the hand it
turns into.

## Fitting a phone screen

The board is a stack of flex rows inside a grid column, which is a shape that overflows sideways
very easily. `scripts/check-page-layout.mjs` (part of `npm run check`) pins the contract:

* **`min-width: 0`** on the board and on the page element around it. A grid/flex item defaults to
  `min-width: auto`, so one wide row sizes the whole column and pushes the page sideways — this is
  the single most common cause of a phone-width layout bug.
* **Rows of cards wrap** (`.cards-container` for the hand, `.cards-display` for the table): a
  horizontally scrolling hand hides a card behind the screen edge. The header and score rows wrap
  too.
* **Anything that travels sideways is clipped**: the throw/grab moves a card 340-560px toward a
  player's side, which is off-screen on a phone, so `.anim-overlay` clips it (the card leaves the
  table rather than widening the page) and the viewport-fixed `.fx-overlay` clips the dealt piles.
* **The deal piles are placed relative to the viewport** (`min(210px, 26vw)`): a fixed 210px off
  centre is most of a phone's width.
* `overflow-x: clip` on the board root is the last line of defence, and no CSS may hardcode a
  width wider than a phone.

It is a lint, not a layout engine: it cannot tell you the phone looks right, only that the known
ways to break it are absent.

## Header chrome

The header carries the language picker and the one way back, as a single top-right cluster
(`.header-actions`, right-aligned in both header layouts via `margin-left: auto`):

* **one back button for the whole app**, next to the picker (the picker keeps the corner), shown
  only where there is somewhere to go back to (`canGoBack`: create, join, lobby, game) and routing
  through one `goBack()` — from a game it returns to the lobby, from anywhere else to the menu.
  It replaced a per-screen button on each screen plus a `⬅️` column on the game screen that was
  `hidden lg:flex`, which left a phone with no way out of a running game;
* the label is hidden under 420px (the arrow and the `aria-label` carry it), the target is a
  ~2.4rem pill so it is tappable, and the arrow leans left on hover;
* `scripts/check-page-layout.mjs` (see below) fails if any screen loses its way back, if the button
  leaves the header cluster, or if the tap target or the mobile label rule is removed.

## Other UX details

* **Keyboard**: `1`-`5` plays that card when it is your turn (ignored while a capture prompt
  is open or while typing in a field).
* **Notices and errors** are `role="status"`/`aria-live` regions, so a screen reader
  announces them instead of leaving the player to notice the text appear.
* **AI turns** say "AI's turn" under the seat name, like "Your turn"/"Human" for the others. It
  deliberately does *not* animate: with `AI_SEARCH_ITERATIONS=400` a turn takes ~30 ms, so a
  pulsing "thinking" dot would suggest a wait that is not there. One consequence worth keeping:
  the label must stay `display: block` (from `.turn-label`) so it sits *below* the seat name --
  an inline-flex override was what used to push it alongside "P1".
* **Sign-in is a username**, not an email (`{ id, username }` from the API). The form hints the
  allowed shape (3-20 characters: letters, digits, `.`, `_`, `-`) because the server validates
  it rather than sanitising.
