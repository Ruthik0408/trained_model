import { api } from "./client";
import { API_ENDPOINTS, CACHE_CONFIG } from "../constants";

/**
 * API Query Cache - Simple in-memory cache for GET requests
 * TTL: 1 minute for dynamic data
 */
const apiCache = new Map();

function getCacheKey(url, params) {
    return params ? `${url}?${new URLSearchParams(params).toString()}` : url;
}

function getCachedData(key, ttl) {
    const cached = apiCache.get(key);
    if (cached && Date.now() - cached.timestamp < ttl) {
        console.debug(`Cache hit: ${key}`);
        return cached.data;
    }
    if (cached) {
        apiCache.delete(key);
    }
    return null;
}

function setCacheData(key, data) {
    apiCache.set(key, { data, timestamp: Date.now() });
}

export function clearApiCache() {
    apiCache.clear();
    console.debug("API cache cleared");
}

/**
 * Helper to add AbortSignal to axios config
 * @param {AbortSignal} signal - AbortSignal from AbortController
 * @returns {Object} - Config object with signal
 */
function getConfigWithSignal(signal) {
    return signal ? { signal } : {};
}

// API functions with cache support and better error handling
// All functions support optional AbortSignal for request cancellation
export const getWorkbenchTables = (signal) => api.get(API_ENDPOINTS.TABLES, getConfigWithSignal(signal));
export const getWorkbenchConnection = (signal) => api.get(API_ENDPOINTS.CONNECTION, getConfigWithSignal(signal));
export const getWorkbenchDefaultFeatureRules = (payload, signal) => api.post(API_ENDPOINTS.DEFAULT_RULES, payload, getConfigWithSignal(signal));
export const previewWorkbench = (payload, signal) => api.post(API_ENDPOINTS.PREVIEW, payload, getConfigWithSignal(signal));
export const runWorkbench = (payload, signal) => api.post(API_ENDPOINTS.RUN, payload, getConfigWithSignal(signal));

export const getWorkbenchDatasets = (signal) => {
    const cacheKey = getCacheKey(API_ENDPOINTS.DATASETS);
    const cached = getCachedData(cacheKey, CACHE_CONFIG.DATA_TTL_MS);
    if (cached)
        return Promise.resolve({ data: cached });
    return api.get(API_ENDPOINTS.DATASETS, getConfigWithSignal(signal)).then((response) => {
        setCacheData(cacheKey, response.data);
        return response;
    });
};

export const getReviewTable = (datasetTable, anomalyFilter, runId, limit, offset, signal) => {
    const params = {
        dataset_table: datasetTable,
        anomaly_filter: anomalyFilter,
        run_id: runId,
        limit,
        offset,
    };
    const cacheKey = getCacheKey(API_ENDPOINTS.REVIEW_TABLE, params);
    const cached = getCachedData(cacheKey, CACHE_CONFIG.DATA_TTL_MS);
    if (cached)
        return Promise.resolve({ data: cached });
    return api.get(API_ENDPOINTS.REVIEW_TABLE, {
        params,
        ...getConfigWithSignal(signal),
    }).then((response) => {
        setCacheData(cacheKey, response.data);
        return response;
    });
};

export const getWorkbenchReviewRows = (params, signal) => {
    const queryParams = {
        dataset_table: params.datasetTable,
        anomaly_filter: params.anomalyFilter,
        limit: params.limit,
        offset: params.offset,
        run_id: params.runId,
    };
    const cacheKey = getCacheKey(API_ENDPOINTS.REVIEW_ROWS, queryParams);
    const cached = getCachedData(cacheKey, CACHE_CONFIG.DATA_TTL_MS);
    if (cached)
        return Promise.resolve({ data: cached });
    return api.get(API_ENDPOINTS.REVIEW_ROWS, {
        params: queryParams,
        ...getConfigWithSignal(signal),
    }).then((response) => {
        setCacheData(cacheKey, response.data);
        return response;
    });
};
export const submitWorkbenchFeedback = (payload, signal) => {
    const response = api.post("/api/workbench/feedback", payload, getConfigWithSignal(signal));
    response.then(() => {
        clearApiCache();
    });
    return response;
};
export const getWorkbenchReport = (params, signal) => {
    const queryParams = {
        dataset_table: params?.datasetTable,
        run_id: params?.runId,
    };
    const cacheKey = getCacheKey("/api/workbench/report", queryParams);
    const cached = getCachedData(cacheKey, CACHE_CONFIG.DATA_TTL_MS);
    if (cached)
        return Promise.resolve({ data: cached });
    return api.get("/api/workbench/report", {
        params: queryParams,
        ...getConfigWithSignal(signal),
    }).then((response) => {
        setCacheData(cacheKey, response.data);
        return response;
    });
};
