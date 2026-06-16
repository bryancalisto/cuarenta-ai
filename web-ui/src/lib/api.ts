export type PublicUser = {
	id: string;
	email: string;
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

export type Game = {
	id: string;
	humanPlayers: HumanPlayer[];
	timeSteps: number;
	cards: number[][];
	table?: number[];
	scores?: number[];
	currentPlayer?: number;
	lastPlaced?: number | null;
	winner?: number | null;
	done?: boolean;
	deckRemaining?: number;
	durationSeconds?: number;
	createdBy: string;
	createdAt: string;
	startedAt?: string;
	status: string;
	pendingMoves?: Move[];
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

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080').replace(/\/$/, '');

function buildUrl(path: string): string {
	return new URL(path, API_BASE_URL).toString();
}

async function readJson<T>(response: Response): Promise<T> {
	const text = await response.text();
	const data = text ? JSON.parse(text) : null;

	if (!response.ok) {
		throw new Error(data?.error ?? 'Request failed');
	}

	return data as T;
}

async function postJson<T>(path: string, token: string | null, body: unknown): Promise<T> {
	const response = await fetch(buildUrl(path), {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			...(token ? { Authorization: `Bearer ${token}` } : {})
		},
		body: JSON.stringify(body)
	});

	return readJson<T>(response);
}

export async function signUp(email: string, password: string): Promise<AuthResponse> {
	return postJson<AuthResponse>('/auth/sign-up', null, { email, password });
}

export async function signIn(email: string, password: string): Promise<AuthResponse> {
	return postJson<AuthResponse>('/auth/sign-in', null, { email, password });
}

export async function getGame(token: string, gameId: string): Promise<Game> {
	const response = await fetch(buildUrl(`/games/${encodeURIComponent(gameId)}`), {
		headers: {
			Authorization: `Bearer ${token}`
		}
	});

	const data = await readJson<{ game: Game }>(response);
	return data.game;
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

export async function getWebsocketTicket(token: string): Promise<string> {
	const response = await fetch(buildUrl('/auth/ws-ticket'), {
		method: 'POST',
		headers: {
			Authorization: `Bearer ${token}`
		}
	});

	const data = await readJson<{ ticket: string }>(response);
	console.log('tickkk', data.ticket)
	return data.ticket;
}

export function openGameSocket(ticket: string): WebSocket {
	const url = new URL(buildUrl('/ws'));
	url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
	url.searchParams.set('ticket', ticket);
	console.log('URLLL:', url.toString())
	return new WebSocket(url.toString());
}
