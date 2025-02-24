import { useState, useCallback } from 'react';
import { apiRequest, APIError, ENDPOINTS } from '../config/api';

interface UseApiState<T> {
    data: T | null;
    error: APIError | null;
    loading: boolean;
}

export function useApi<T>() {
    const [state, setState] = useState<UseApiState<T>>({
        data: null,
        error: null,
        loading: false,
    });

    const execute = useCallback(async (
        endpoint: string,
        options: RequestInit = {}
    ) => {
        setState(prev => ({ ...prev, loading: true, error: null }));

        try {
            const data = await apiRequest(endpoint, options);
            setState({ data, loading: false, error: null });
            return data;
        } catch (error) {
            const apiError = error instanceof APIError 
                ? error 
                : new APIError(500, 'Internal Error', error);
            setState({ data: null, loading: false, error: apiError });
            throw apiError;
        }
    }, []);

    return {
        ...state,
        execute,
    };
}

// Example usage:
export function usePodcastContext(podcastId: string) {
    const api = useApi();

    const fetchContext = useCallback(async () => {
        return api.execute(ENDPOINTS.PODCAST_CONTEXT(podcastId));
    }, [podcastId, api]);

    return {
        ...api,
        fetchContext,
    };
}

export function usePodcastChat(podcastId: string) {
    const api = useApi();

    const sendMessage = useCallback(async (message: string) => {
        return api.execute(
            ENDPOINTS.PODCAST_CHAT(podcastId),
            {
                method: 'POST',
                body: JSON.stringify({ message }),
            }
        );
    }, [podcastId, api]);

    return {
        ...api,
        sendMessage,
    };
} 