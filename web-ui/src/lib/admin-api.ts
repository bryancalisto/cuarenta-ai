/**
 * Admin dashboard API: the statistics the backend records about played games, and the
 * turn-by-turn replay of one of them.
 *
 * WHY ITS OWN MODULE: every type here is the wire contract of `backend/admin.go`, and the UI must
 * not have to guess it. The dashboard sends a *separate* credential from the player session: the
 * admin account lives in `ADMIN_USERNAME`/`ADMIN_PASSWORD` (backend/.env) and its token carries a
 * `role: "admin"` claim, so a normal player's JWT is rejected by these endpoints even if it is
 * passed. Nothing here is used by the game itself.
 */
import axios from 'axios';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080/api').replace(/\/$/, '');

const client = axios.create({ baseURL: API_BASE_URL });

/** Where the admin token is kept between reloads. A player session uses its own key. */
export const ADMIN_TOKEN_KEY = 'cuarenta-admin-token';

// ---------------------------------------------------------------------------------------------
// Wire types (mirrors of the Go structs)
// ---------------------------------------------------------------------------------------------

export type SeatKind = 'human' | 'ai';
export type GameStatus = 'running' | 'finished' | 'canceled';

/** A team as it appears in the match history: who sat in it, and how to label it. */
export type TeamTag = {
	team: number; // 1 or 2
	/** e.g. "1 (H, AI)" — the team, then its seats in seat order. */
	label: string;
	seats: SeatKind[];
	/** True when the team has at least one human and at least one AI. */
	mixed: boolean;
};

export type GameSeat = { seat: number; team: number; kind: SeatKind };

export type GameRow = {
	id: string;
	startedAt: string | null;
	endedAt: string | null;
	status: GameStatus;
	plies: number;
	deals: number;
	durationSeconds: number;
	scoreTeam1: number;
	scoreTeam2: number;
	seats: GameSeat[];
	winner: TeamTag | null;
	loser: TeamTag | null;
};

export type AdminTotals = {
	games: number;
	finished: number;
	canceled: number;
	running: number;
	plies: number;
	humans: number;
	ais: number;
	avgPliesPerGame: number;
	avgDurationSeconds: number;
	avgAiDecisionMs: number;
	captures: number;
	caidas: number;
	limpias: number;
};

/** Win rate of one seating kind: shared by humans and by AI seats. */
export type KindWinRate = {
	kind: SeatKind;
	seats: number; // seat-games (a 4-player game contributes 2 per team)
	wins: number;
	winRate: number;
};

/** Win rate per team composition, e.g. "H+AI" against "AI+AI". */
export type CompositionWinRate = {
	composition: string;
	opponent: string;
	games: number;
	wins: number;
	winRate: number;
};

/** The headline comparison: games where exactly one team held all the humans. */
export type AiVsHuman = {
	games: number;
	humanTeamWins: number;
	aiTeamWins: number;
	humanTeamWinRate: number;
	aiTeamWinRate: number;
};

/** Win rate of the AI seats grouped by the search budget they played with. */
export type BudgetWinRate = {
	iterations: number;
	games: number;
	aiTeams: number;
	aiWins: number;
	winRate: number;
};

export type Overview = {
	generatedAt: string;
	totals: AdminTotals;
	bySeatKind: KindWinRate[];
	byComposition: CompositionWinRate[];
	aiVsHuman: AiVsHuman;
	bySearchBudget: BudgetWinRate[];
	/** The most recent games, same shape as the history table rows. */
	recentGames: GameRow[];
};

export type TimeseriesPoint = {
	day: string; // YYYY-MM-DD
	games: number;
	plies: number;
	avgPlies: number;
	/** Null on days with no comparable game (no AI-vs-human game that day). */
	aiTeamWinRate: number | null;
	humanTeamWinRate: number | null;
};

export type Timeseries = {
	days: number;
	points: TimeseriesPoint[];
};

export type GamesPage = {
	games: GameRow[];
	total: number;
	limit: number;
	offset: number;
};

/** One move, as it was played. */
export type ReplayMove = {
	card: number;
	captures: number[];
	/** Game points the move scored (capture points plus caída/limpia). */
	points: number;
	caida: boolean;
	limpia: boolean;
};

/**
 * One frame of the replay: the position AFTER the move, or the opening deal for ply 0. `hands` is
 * indexed by seat, so the UI can show any seat's hand (that is the point of a replay in an
 * imperfect-information game).
 */
export type ReplayFrame = {
	ply: number;
	seat: number | null;
	kind: SeatKind | null;
	move: ReplayMove | null;
	scores: number[];
	table: number[];
	hands: number[][];
	captured: number[][];
	lastPlaced: number | null;
	deckRemaining: number;
	currentPlayer: number;
};

export type Replay = {
	game: GameRow;
	frames: ReplayFrame[];
};

export type AdminSignInResponse = {
	accessToken: string;
	admin: { username: string };
};

// ---------------------------------------------------------------------------------------------
// Calls
// ---------------------------------------------------------------------------------------------

function authHeader(token: string) {
	return { Authorization: `Bearer ${token}` };
}

/** The error message the backend sent, or a fallback. */
function messageOf(error: unknown, fallback: string): string {
	if (axios.isAxiosError(error)) {
		const fromServer = (error.response?.data as { error?: string } | undefined)?.error;
		if (fromServer) return fromServer;
	}
	return error instanceof Error ? error.message : fallback;
}

/** Human-readable reason a call failed, so the dashboard can show it instead of "failed". */
export function adminErrorMessage(error: unknown, fallback = 'Request failed'): string {
	return messageOf(error, fallback);
}

export function isUnauthorized(error: unknown): boolean {
	return axios.isAxiosError(error) && error.response?.status === 401;
}

/** Exchange the admin credentials for a token. */
export async function adminSignIn(username: string, password: string): Promise<AdminSignInResponse> {
	const response = await client.post('/admin/sign-in', { username, password });
	return response.data as AdminSignInResponse;
}

export async function getOverview(token: string): Promise<Overview> {
	const response = await client.get('/admin/stats/overview', { headers: authHeader(token) });
	return response.data as Overview;
}

export async function getGames(token: string, limit = 50, offset = 0): Promise<GamesPage> {
	const response = await client.get('/admin/stats/games', {
		headers: authHeader(token),
		params: { limit, offset }
	});
	return response.data as GamesPage;
}

export async function getTimeseries(token: string, days = 30): Promise<Timeseries> {
	const response = await client.get('/admin/stats/timeseries', {
		headers: authHeader(token),
		params: { days }
	});
	return response.data as Timeseries;
}

export async function getReplay(token: string, gameId: string): Promise<Replay> {
	const response = await client.get(`/admin/stats/games/${encodeURIComponent(gameId)}/replay`, {
		headers: authHeader(token)
	});
	return response.data as Replay;
}
