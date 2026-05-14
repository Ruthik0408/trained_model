import { useEffect, useState, useCallback, useRef } from "react";
/**
 * useAsync Hook - Manage async operations with loading, data, and error states
 * Prevents race conditions with cleanup on unmount
 */
export function useAsync(asyncFunction, immediate = true, deps = []) {
    const [status, setStatus] = useState("idle");
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);
    const isMountedRef = useRef(true);
    const execute = useCallback(async () => {
        setStatus("pending");
        setData(null);
        setError(null);
        try {
            const response = await asyncFunction();
            if (isMountedRef.current) {
                setData(response);
                setStatus("success");
            }
        }
        catch (err) {
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
        return () => {
            isMountedRef.current = false;
        };
    }, deps);
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
            console.error(`Error reading localStorage key "${key}":`, error);
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
            console.error(`Error setting localStorage key "${key}":`, error);
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
            console.error(`Error reading sessionStorage key "${key}":`, error);
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
            console.error(`Error setting sessionStorage key "${key}":`, error);
        }
    };
    return [storedValue, setValue];
}
