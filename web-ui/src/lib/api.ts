import axios from 'axios';

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

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080').replace(/\/$/, '');

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

export async function signUp(email: string, password: string): Promise<AuthResponse> {
	return postJson<AuthResponse>('/auth/sign-up', null, { email, password });
}

export async function signIn(email: string, password: string): Promise<AuthResponse> {
	return postJson<AuthResponse>('/auth/sign-in', null, { email, password });
}

export async function refreshToken(): Promise<string> {
	const response = await apiClient.post('/auth/refresh-token', {});
	return response.data.accessToken;
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
	url.pathname = '/ws';
	url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
	url.searchParams.set('ticket', ticket);
	return new WebSocket(url.toString());
}
