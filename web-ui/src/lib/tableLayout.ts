/**
 * Where a card sits on the table, and — more importantly — why it stays there.
 *
 * THE RULE: the first card goes in the MIDDLE and the following ones are laid out AROUND it, from
 * the inside out, and a card keeps the place it was given for as long as it is on the table. So
 * playing a card or taking one moves exactly one card: the one being played or taken. Nothing else
 * on the table shifts, which is what makes the board readable — a table that re-centres itself
 * under your eyes is impossible to follow.
 *
 * A flex row with `justify-content: center` does the opposite: it re-centres the whole row every
 * time the number of cards changes.
 *
 * Slots are ordinals around the centre: 0 is the middle, then 1 = one step right, 2 = one step
 * left, 3 = two steps right, 4 = two steps left, and so on. When a card leaves it frees its slot,
 * and the next card takes the INNERMOST free one, so the table fills its gaps instead of drifting
 * outwards for the whole deal.
 *
 * Pure on purpose (no Svelte, no DOM): `scripts/check-table-layout.mjs` runs it and asserts that
 * adding and removing cards leaves every other card's slot untouched.
 */
export class TableSlots {
	private slots = new Map<number, number>();

	/**
	 * The slot of a card, assigning one if it does not have one yet. Stable for as long as the card
	 * is on the table.
	 */
	slotFor(card: number): number {
		const existing = this.slots.get(card);
		if (existing !== undefined) return existing;

		const used = new Set(this.slots.values());
		let ordinal = 0;
		while (used.has(ordinal)) ordinal++;

		this.slots.set(card, ordinal);
		return ordinal;
	}

	/**
	 * Keep the slots of the cards that are on the table, and forget every other one — freeing the
	 * slot of a card that has left.
	 *
	 * Named for what it KEEPS rather than what it drops: called with the cards that are gone it
	 * would quietly delete exactly the ones to keep, which is an easy mistake to make and a silent
	 * one to debug.
	 *
	 * A card on its way out keeps the position it was last rendered with, so this does not move it;
	 * it only means the next card can take that spot.
	 */
	retain(visible: Iterable<number>): void {
		const alive = new Set(visible);
		for (const card of [...this.slots.keys()]) {
			if (!alive.has(card)) this.slots.delete(card);
		}
	}

	/** How many cards currently hold a slot (mostly for tests and debugging). */
	get size(): number {
		return this.slots.size;
	}
}

/**
 * The slot ordinal as a number of steps from the centre: 0, +1, -1, +2, -2, ...
 *
 * The first card is the centre one, then they alternate right and left, which is what "add them
 * around the centre" means: the row grows outwards on both sides instead of in one direction.
 */
export function slotOffsetUnits(ordinal: number): number {
	if (ordinal === 0) return 0;
	const step = Math.ceil(ordinal / 2);
	return ordinal % 2 === 0 ? -step : step;
}
