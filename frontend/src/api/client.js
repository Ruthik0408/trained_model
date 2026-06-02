import axios from "axios";
// API client with enhanced error handling and interceptors
export const api = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000",
    timeout: 300000, // 5 minute timeout for long-running workbench requests
});
// Request interceptor - add request logging
api.interceptors.request.use((config) => {
    const requestId = Math.random().toString(36).substring(2, 11);
    config.headers["X-Request-ID"] = requestId;
    config.metadata = { requestId, startTime: Date.now() };
    return config;
}, (error) => {
    console.error("Request configuration error:", error);
    return Promise.reject(error);
});
// Response interceptor - log timing and errors
api.interceptors.response.use((response) => {
    const metadata = response.config.metadata;
    if (metadata) {
        const duration = Date.now() - metadata.startTime;
        console.debug(`[${metadata.requestId}] ${response.config.method?.toUpperCase()} ${response.config.url} - ${response.status} - ${duration}ms`);
    }
    return response;
}, (error) => {
    const config = error.config;
    const metadata = config?.metadata;
    const status = error.response?.status || "unknown";
    const isTimeout = error.code === "ECONNABORTED" || error.message.toLowerCase().includes("timeout");
    const message = error.response?.data?.detail ||
        (isTimeout ? "The request took too long to complete." : error.message);
    if (metadata) {
        const duration = Date.now() - metadata.startTime;
        console.error(`[${metadata.requestId}] ${config?.method?.toUpperCase()} ${config?.url} - ${status} - ${duration}ms - ${message}`);
    }
    // Handle specific error cases
    if (error.response?.status === 503) {
        console.error("Service unavailable - backend connection failed");
    }
    else if (error.response?.status === 429) {
        console.warn("Rate limited - too many requests");
    }
    else if (isTimeout) {
        console.warn("Request timed out before the backend finished processing");
    }
    return Promise.reject(error);
});
