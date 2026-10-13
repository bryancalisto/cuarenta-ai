import axios from 'axios';

export type PublicUser = {
	id: string;
	username: string;
};

export type HumanPlayer = {
	userId: string;
	index: number;
	team: number;
	joinedAt: string;
};

export type Move = {
	cardIndex: number;
	captureIndices?: number[];
	playerIndex: number;
};

export type CaptureOption = {
	captureIndices: number[];
};

/**
 * Sent with the ONE snapshot published by the move that ended a deal and paid the cartón:
 * the piles as they were counted (`captured`/`counted`), the points each team was paid
 * (`carton`, index 0/1), and the cards still on the table when the deal ended. The state
 * clears the piles in that same step, so this payload is what lets the UI *show* the
 * counting instead of just jumping to the new scores.
 */
export type RoundEnd = {
	captured: number[][];
	counted: number[];
	carton: number[];
	tableCards?: number[];
};

export type Game = {
	id: string;
	humanPlayers: HumanPlayer[];
	timeSteps: number;
	cards: number[][];
	table?: number[];
	scores?: number[];
	cardsCaptured?: number[][];
	currentPlayer?: number;
	lastPlaced?: number | null;
	winner?: number | null;
	done?: boolean;
	deckRemaining?: number;
	newRoundStart?: boolean;
	/** Counts the deals of this match (1 = the opening deal). It changes exactly when every
	 *  seat has been dealt a fresh hand, which is what the deal animation triggers on. */
	dealId?: number;
	/** Present only on the snapshot of the move that ended a deal (see RoundEnd). */
	roundEnd?: RoundEnd;
	durationSeconds?: number;
	createdBy: string;
	createdAt: string;
	startedAt?: string;
	status: string;
	move?: Move;
	captureOptions?: CaptureOption[];
	awaitingCapture?: boolean;
};

export type AuthResponse = {
	accessToken: string;
	user: PublicUser;
};

export type WsMessage =
	| {
			type: 'connection.ready';
	  }
	| {
			type: 'game.snapshot';
			game: Game;
	  }
	| {
			type: 'error';
			error: {
				code: number;
				message: string;
			};
	  };

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080/api').replace(/\/$/, '');

const apiClient = axios.create({
	baseURL: API_BASE_URL
});

let signOutCallback: (() => void) | null = null;

export function configureApi(onSignOut: () => void): void {
	signOutCallback = onSignOut;
}

apiClient.interceptors.response.use(
	response => response,
	error => {
		if (error.response?.status === 401 && signOutCallback) {
			signOutCallback();
		}
		return Promise.reject(error);
	}
);

async function postJson<T>(path: string, token: string | null, body: unknown): Promise<T> {
	const response = await apiClient.post(path, body, {
		headers: token ? { Authorization: `Bearer ${token}` } : {}
	});
	return response.data as T;
}

// Sign-in identity is a username, not an email (see backend/db.go: UsernameProblem pins the
// allowed shape). Lookups are case-insensitive server-side.
export async function signUp(username: string, password: string): Promise<AuthResponse> {
	return postJson<AuthResponse>('/auth/sign-up', null, { username, password });
}

export async function signIn(username: string, password: string): Promise<AuthResponse> {
	return postJson<AuthResponse>('/auth/sign-in', null, { username, password });
}

export async function refreshToken(): Promise<string> {
	const response = await apiClient.post('/auth/refresh-token', {});
	return response.data.accessToken;
}

export async function getMyGames(token: string): Promise<Game[]> {
	const response = await apiClient.get('/games', {
		headers: {
			Authorization: `Bearer ${token}`
		}
	});

	return response.data.games || [];
}

export async function getGame(token: string, gameId: string): Promise<Game> {
	const response = await apiClient.get(`/games/${encodeURIComponent(gameId)}`, {
		headers: {
			Authorization: `Bearer ${token}`
		}
	});

	return response.data.game;
}

export async function createGame(
	token: string,
	body: { spotIndex: number; team: number }
): Promise<Game> {
	const data = await postJson<{ game: Game }>('/games', token, body);
	return data.game;
}

export async function joinGame(
	token: string,
	gameId: string,
	body: { spotIndex: number; team: number }
): Promise<Game> {
	const data = await postJson<{ game: Game }>(`/games/${encodeURIComponent(gameId)}/join`, token, body);
	return data.game;
}

export async function startGame(token: string, gameId: string): Promise<Game> {
	const data = await postJson<{ game: Game }>(`/games/${encodeURIComponent(gameId)}/start`, token, {});
	return data.game;
}

export async function cancelGame(token: string, gameId: string): Promise<{ message: string }> {
	return postJson<{ message: string }>(`/games/${encodeURIComponent(gameId)}/cancel`, token, {});
}


export async function getWebsocketTicket(token: string): Promise<string> {
	const response = await apiClient.post('/auth/ws-ticket', {}, {
		headers: {
			Authorization: `Bearer ${token}`
		}
	});

	return response.data.ticket;
}

export function openGameSocket(ticket: string): WebSocket {
	const url = new URL(API_BASE_URL);
	url.pathname = '/api/ws';
	url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
	url.searchParams.set('ticket', ticket);
	return new WebSocket(url.toString());
}
