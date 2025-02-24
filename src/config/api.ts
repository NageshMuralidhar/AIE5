// API Configuration
export const API_URL = import.meta.env.VITE_API_URL;
export const WS_URL = import.meta.env.VITE_WS_URL;

// API Endpoints
export const ENDPOINTS = {
    AUDIO_LIST: '/audio-list',
    AUDIO: (filename: string) => `/audio/${filename}`,
    PODCAST_CONTEXT: (id: string) => `/podcast/${id}/context`,
    CHAT: '/chat',
    PODCAST_CHAT: (id: string) => `/podcast-chat/${id}`,
} as const;

// API Helper Functions
export const getFullUrl = (endpoint: string) => `${API_URL}${endpoint}`;

// Common Headers
export const DEFAULT_HEADERS = {
    'Content-Type': 'application/json',
};

// API Error Handler
export class APIError extends Error {
    constructor(
        public status: number,
        public statusText: string,
        public data: any
    ) {
        super(`${status} ${statusText}`);
        this.name = 'APIError';
    }
}

// API Response Handler
export const handleResponse = async (response: Response) => {
    const data = await response.json();
    
    if (!response.ok) {
        throw new APIError(
            response.status,
            response.statusText,
            data
        );
    }
    
    return data;
};

// API Request Function
export const apiRequest = async (
    endpoint: string,
    options: RequestInit = {}
) => {
    const url = getFullUrl(endpoint);
    const response = await fetch(url, {
        ...options,
        headers: {
            ...DEFAULT_HEADERS,
            ...options.headers,
        },
    });
    
    return handleResponse(response);
}; 