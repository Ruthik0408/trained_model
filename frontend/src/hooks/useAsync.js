import { useEffect, useState, useCallback, useRef } from "react";
const shouldLogDebug = import.meta.env.DEV || import.meta.env.VITE_API_DEBUG === "true";
/**
 * useAsync Hook - Manage async operations with loading, data, and error states
 * 
 * Key Features:
 * - Prevents race conditions with cleanup on unmount
 * - Cancels in-flight requests when component unmounts (prevents memory leaks)
 * - Supports immediate execution or manual trigger
 * - Handles AbortController for cancellation
 * 
 * @param {Function} asyncFunction - Async function to execute (should handle AbortSignal)
 * @param {boolean} immediate - Execute immediately on mount (default: true)
 * @param {Array} deps - Dependencies array for re-execution (default: [])
 * @returns {Object} - { execute, status, data, error, isLoading, isError, isSuccess }
 */
export function useAsync(asyncFunction, immediate = true, deps = []) {
    const [status, setStatus] = useState("idle");
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);
    const isMountedRef = useRef(true);
    const abortControllerRef = useRef(null);

    const execute = useCallback(async () => {
        // Cancel any previous request
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        // Create new abort controller for this request
        abortControllerRef.current = new AbortController();
        const signal = abortControllerRef.current.signal;

        setStatus("pending");
        setData(null);
        setError(null);

        try {
            // Pass abort signal to async function
            const response = await asyncFunction(signal);

            // Only update state if component is still mounted and request wasn't aborted
            if (isMountedRef.current && !signal.aborted) {
                setData(response);
                setStatus("success");
            }
        } catch (err) {
            // Ignore abort errors - this is expected when component unmounts
            if (err?.name === "AbortError") {
                if (shouldLogDebug) {
                    console.debug("Request cancelled");
                }
                return;
            }

            // Update state only if component is still mounted
            if (isMountedRef.current) {
                setError(err instanceof Error ? err : new Error(String(err)));
                setStatus("error");
            }
        }
    }, [asyncFunction]);

    useEffect(() => {
        isMountedRef.current = true;

        if (immediate) {
            execute();
        }

        // Cleanup on unmount or dependency change
        return () => {
            isMountedRef.current = false;
            // Abort any in-flight request
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
        };
    }, [execute, immediate, ...deps]);

    return {
        execute,
        status,
        data,
        error,
        isLoading: status === "pending",
        isError: status === "error",
        isSuccess: status === "success",
    };
}
/**
 * useDebounce Hook - Debounce a value with configurable delay
 * Useful for search inputs and filter changes
 */
export function useDebounce(value, delayMs = 500) {
    const [debouncedValue, setDebouncedValue] = useState(value);
    useEffect(() => {
        const handler = setTimeout(() => {
            setDebouncedValue(value);
        }, delayMs);
        return () => clearTimeout(handler);
    }, [value, delayMs]);
    return debouncedValue;
}
/**
 * useLocalStorage Hook - Sync state with localStorage
 * Persists data across page reloads
 */
export function useLocalStorage(key, initialValue) {
    const [storedValue, setStoredValue] = useState(() => {
        if (typeof window === "undefined") {
            return initialValue;
        }
        try {
            const item = window.localStorage.getItem(key);
            return item ? JSON.parse(item) : initialValue;
        }
        catch (error) {
            if (shouldLogDebug) {
                console.error(`Error reading localStorage key "${key}":`, error);
            }
            return initialValue;
        }
    });
    const setValue = (value) => {
        try {
            setStoredValue(value);
            if (typeof window !== "undefined") {
                window.localStorage.setItem(key, JSON.stringify(value));
            }
        }
        catch (error) {
            if (shouldLogDebug) {
                console.error(`Error setting localStorage key "${key}":`, error);
            }
        }
    };
    return [storedValue, setValue];
}
/**
 * usePrevious Hook - Store previous value from render
 * Useful for comparing old vs new values
 */
export function usePrevious(value) {
    const ref = useRef(undefined);
    useEffect(() => {
        ref.current = value;
    }, [value]);
    return ref.current;
}
/**
 * useSessionStorage Hook - Similar to useLocalStorage but uses sessionStorage
 * Data cleared when tab is closed
 */
export function useSessionStorage(key, initialValue) {
    const [storedValue, setStoredValue] = useState(() => {
        if (typeof window === "undefined") {
            return initialValue;
        }
        try {
            const item = window.sessionStorage.getItem(key);
            return item ? JSON.parse(item) : initialValue;
        }
        catch (error) {
            if (shouldLogDebug) {
                console.error(`Error reading sessionStorage key "${key}":`, error);
            }
            return initialValue;
        }
    });
    const setValue = (value) => {
        try {
            setStoredValue(value);
            if (typeof window !== "undefined") {
                window.sessionStorage.setItem(key, JSON.stringify(value));
            }
        }
        catch (error) {
            if (shouldLogDebug) {
                console.error(`Error setting sessionStorage key "${key}":`, error);
            }
        }
    };
    return [storedValue, setValue];
}
