/**
 * Tiny dependency-free internationalisation for the Cuarenta UI.
 *
 * WHY NOT A LIBRARY: this app has exactly two locales and a few dozen strings, and the
 * project's rule everywhere else is "one file you can read". `svelte-i18n` would add a
 * dependency, a build step and a message-format runtime for the same result. This module is
 * the whole feature: a locale, a dictionary, and a `t()`.
 *
 * NATURAL-LANGUAGE KEYS: the key of a translation IS its English text
 * (`t('Sign in')` → `'Iniciar sesión'`). That has three consequences worth knowing:
 *
 *   * a missing translation falls back to the English text, so the UI can never show a raw
 *     identifier like `board.actions.capture.title` to a player;
 *   * the English locale needs no data at all — it is the identity map;
 *   * renaming an English string silently orphans its translation, which is exactly what
 *     `web-ui/scripts/check-i18n.mjs` exists to catch (it fails on a `t()` key with no
 *     translation AND on a translation no code uses).
 *
 * PLACEHOLDERS: `t('Cartón paid: Team 1 +{first} · Team 2 +{second}', { first: 6, second: 0 })`.
 * Values are stringified; there is no plural machinery because no string here needs it.
 */

export type Locale = 'en' | 'es';

/** The locales offered in the picker. Labels are ENDONYMS: a player looking for Spanish
 *  looks for "Español", not for "Spanish". */
export const LOCALES: ReadonlyArray<{ value: Locale; label: string; short: string }> = [
	{ value: 'en', label: 'English', short: 'EN' },
	{ value: 'es', label: 'Español', short: 'ES' }
];

export const DEFAULT_LOCALE: Locale = 'en';
/** Where the choice is remembered. Exported so it can be checked (and poked at in DevTools). */
export const LOCALE_STORAGE_KEY = 'cuarenta-locale';

/** One warning per session if the choice cannot be stored, instead of failing silently: a
 *  language that will not stay put is otherwise indistinguishable from a bug in the UI. */
let warnedAboutStorage = false;

function store(next: Locale): void {
	try {
		globalThis.localStorage?.setItem(LOCALE_STORAGE_KEY, next);
	} catch {
		if (!warnedAboutStorage) {
			warnedAboutStorage = true;
			console.warn(
				`[i18n] could not save the language to localStorage (${LOCALE_STORAGE_KEY}); ` +
					'it will not persist. Private mode, a sandboxed frame or blocked storage does this.'
			);
		}
	}
}

/** The stored choice, or null when there is none (or storage is unusable). */
function stored(): Locale | null {
	try {
		const value = globalThis.localStorage?.getItem(LOCALE_STORAGE_KEY);
		return value === 'en' || value === 'es' ? value : null;
	} catch {
		return null;
	}
}

/**
 * Spanish translations, keyed by the English text.
 *
 * Game vocabulary stays in Spanish in both locales because it is the game's own language and
 * the terms are used untranslated by players: caída, limpia, cartón. Everything else is
 * translated properly (including the UI's own "Team" → "Equipo").
 */
const ES: Record<string, string> = {
	// ---- chrome -----------------------------------------------------------------
	'Cuarenta Multiplayer': 'Cuarenta Multijugador',
	'Ecuador · 40 · caídas & limpias': 'Ecuador · 40 · caídas y limpias',
	'Cuarenta, the Ecuadorian card game': 'Cuarenta, el juego de cartas ecuatoriano',
	'Language': 'Idioma',
	'Ecuador': 'Ecuador',

	// ---- auth -------------------------------------------------------------------
	'Sign in': 'Iniciar sesión',
	'Sign up': 'Registrarse',
	'Username': 'Nombre de usuario',
	'yourname': 'tunombre',
	'3-20 characters: letters, digits, . _ -': '3-20 caracteres: letras, números, . _ -',
	'Password': 'Contraseña',
	'Working...': 'Procesando…',
	'Create account': 'Crear cuenta',
	'Authentication failed': 'No se pudo autenticar',
	'Signed in': 'Sesión iniciada',
	'Account created': 'Cuenta creada',

	// ---- menu -------------------------------------------------------------------
	'Menu': 'Menú',
	'You are signed in as {username}. Create a new game or join an existing game.':
		'Has iniciado sesión como {username}. Crea una partida nueva o únete a una existente.',
	'New game': 'Nueva partida',
	'Join game': 'Unirse a partida',
	'Sign out': 'Cerrar sesión',
	'My games': 'Mis partidas',
	'Loading games...': 'Cargando partidas…',
	'No games created yet.': 'Todavía no hay partidas creadas.',
	'Owner': 'Creador',
	'Created': 'Creada',
	'Players: {count}/4': 'Jugadores: {count}/4',
	'Duration: {minutes}m': 'Duración: {minutes} min',
	'Finished': 'Finalizada',
	'In progress': 'En curso',
	'Canceled': 'Cancelada',
	'Lobby': 'Sala',
	'Unknown': 'Desconocido',
	'Delete game': 'Eliminar partida',

	// ---- create -----------------------------------------------------------------
	'Pick your seat': 'Elige tu puesto',
	'Choose a team and spot index, then create the game.':
		'Elige equipo y puesto, y crea la partida.',
	'Team': 'Equipo',
	'Spot index': 'Puesto',
	'Seat {seat}': 'Puesto {seat}',
	'Team {team}': 'Equipo {team}',
	'Creating...': 'Creando…',
	'Create game': 'Crear partida',

	// ---- join -------------------------------------------------------------------
	'Enter the game 4-character code': 'Introduce el código de 4 caracteres',
	'Game code': 'Código de partida',
	'Lobby preview': 'Vista previa de la sala',
	'Human Players: {count}/4': 'Jugadores humanos: {count}/4',
	'Entering...': 'Entrando…',
	'Enter game': 'Entrar en la partida',
	'Type the game code to enter its lobby.': 'Escribe el código para entrar en su sala.',

	// ---- lobby ------------------------------------------------------------------
	'Game {id}': 'Partida {id}',
	'Players can join until the creator starts the game.':
		'Los jugadores pueden unirse hasta que el creador inicie la partida.',
	'Status': 'Estado',
	'Created by {who}': 'Creada por {who}',
	'you': 'ti',
	'Open': 'Libre',
	'Starting...': 'Iniciando…',
	'Show table': 'Ver la mesa',
	'Start game': 'Iniciar partida',
	'Canceling...': 'Cancelando…',
	'Cancel game': 'Cancelar partida',
	'Back to menu': 'Volver al menú',

	// ---- notices / errors -------------------------------------------------------
	'Game was canceled': 'La partida fue cancelada',
	'Connected to game {id}': 'Conectado a la partida {id}',
	'Opened lobby {id}': 'Sala {id} abierta',
	'Game {id} created': 'Partida {id} creada',
	'Game {id} started': 'Partida {id} iniciada',
	'Joined game {id}': 'Te uniste a la partida {id}',
	'Opened game {id}': 'Partida {id} abierta',
	'Are you sure you want to cancel this game? It will be deleted immediately for all players.':
		'¿Seguro que quieres cancelar esta partida? Se eliminará de inmediato para todos los jugadores.',
	'Game canceled successfully': 'Partida cancelada',
	'Could not create game': 'No se pudo crear la partida',
	'Could not join game': 'No se pudo unir a la partida',
	'Could not load game': 'No se pudo cargar la partida',
	'Could not start game': 'No se pudo iniciar la partida',
	'Could not cancel game': 'No se pudo cancelar la partida',

	// ---- board ------------------------------------------------------------------
	'Game': 'Partida',
	'Team 1': 'Equipo 1',
	'Team 2': 'Equipo 2',
	'Cartón:': 'Cartón:',
	'Your turn': 'Tu turno',
	// Card faces are drawn, not written, so these four are for the accessible name only.
	'Diamonds': 'Diamantes',
	'Hearts': 'Corazones',
	'Spades': 'Espadas',
	'Clubs': 'Tréboles',
	'{rank} of {suit}': '{rank} de {suit}',
	'Table is empty': 'La mesa está vacía',
	'You': 'Tú',
	'Your cards': 'Tus cartas',
	'Your hand': 'Tu mano',
	'Choose which cards to capture:': 'Elige qué cartas capturar:',
	'No capture': 'Sin captura',
	'Press 1-5 to play a card': 'Pulsa 1-5 para jugar una carta',
	'🃏 End of the deal — counting the cartón…': '🃏 Fin del reparto: contando el cartón…',
	'Cartón paid: Team 1 +{first} · Team 2 +{second}':
		'Cartón pagado: Equipo 1 +{first} · Equipo 2 +{second}',
	'+{points} points': '+{points} puntos',
	'no cartón': 'sin cartón',

	// ---- results ----------------------------------------------------------------
	'Winner': 'Ganador',
	'Time played:': 'Duración:',
	'Return to lobby': 'Volver a la sala'
};

/** The reactive locale. Module-level `$state` in a `.svelte.ts` module is shared by every
 *  component that reads it, which is what makes a dropdown in the header re-render the
 *  whole app. */
let current = $state<Locale>(DEFAULT_LOCALE);

export function locale(): Locale {
	return current;
}

export function localeLabel(value: Locale = current): string {
	return LOCALES.find((entry) => entry.value === value)?.label ?? value;
}

/** Switch locale: re-renders every `t()` in the app, remembers the choice in `localStorage`
 *  (that is what makes it survive a reload), and keeps the document's `lang` in sync (screen
 *  readers and the browser's own hyphenation use it). */
export function setLocale(next: Locale): void {
	if (next === current) return;
	current = next;
	apply(next);
	store(next);
}

/** Locale of a fresh visit: the stored choice, else the browser's preference, else English.
 *
 *  Client-only, and called from `onMount`: the server always renders English, so running this
 *  before hydration would make the two disagree. It also STORES what it resolved, so a choice
 *  made by the browser's own preference is remembered like one made in the dropdown. */
export function initLocale(): void {
	if (typeof window === 'undefined') return;
	const remembered = stored();
	const next = remembered ?? detectLocale();
	current = next;
	apply(next);
	if (!remembered) store(next);
}

/** The locale the browser asks for, from `navigator.language`. */
function detectLocale(): Locale {
	const langs = typeof navigator === 'undefined' ? [] : (navigator.languages ?? []);
	const nav = navigator?.language || langs[0] || '';
	return nav.toLowerCase().startsWith('es') ? 'es' : DEFAULT_LOCALE;
}

/** Keep `<html lang>` in step with the locale: it is what screen readers and the browser's own
 *  hyphenation use. */
function apply(next: Locale): void {
	if (typeof document !== 'undefined') {
		document.documentElement.lang = next;
	}
}

/**
 * Translate `key` (which is its English text), substituting `{name}` placeholders.
 *
 * Unknown keys return the key itself: a missing translation shows English, never an
 * identifier.
 */
export function t(key: string, vars?: Record<string, string | number>): string {
	const text = current === 'es' ? (ES[key] ?? key) : key;
	if (!vars) return text;
	return text.replace(/\{(\w+)\}/g, (match, name: string) =>
		name in vars ? String(vars[name]) : match
	);
}

/** Exposed for the dictionary checker (`web-ui/scripts/check-i18n.mjs`). */
export const DICTIONARY: Record<Locale, Record<string, string>> = { en: {}, es: ES };
