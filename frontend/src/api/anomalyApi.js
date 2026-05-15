import { api } from "./client";
/**
 * API Query Cache - Simple in-memory cache for GET requests
 * TTL: 1 minute for dynamic data
 */
const apiCache = new Map();
const DATA_CACHE_TTL = 60 * 1000; // 1 minute
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
// API functions with cache support and better error handling
export const getWorkbenchTables = () => api.get("/api/workbench/tables");
export const getWorkbenchConnection = () => api.get("/api/workbench/connection");
export const getWorkbenchDefaultFeatureRules = (payload) => api.post("/api/workbench/default-feature-rules", payload);
export const previewWorkbench = (payload) => api.post("/api/workbench/preview", payload);
export const runWorkbench = (payload) => api.post("/api/workbench/run", payload);
export const getWorkbenchDatasets = () => {
    const cacheKey = getCacheKey("/api/workbench/datasets");
    const cached = getCachedData(cacheKey, DATA_CACHE_TTL);
    if (cached)
        return Promise.resolve({ data: cached });
    return api.get("/api/workbench/datasets").then((response) => {
        setCacheData(cacheKey, response.data);
        return response;
    });
};
export const getReviewTable = (datasetTable, anomalyFilter, runId, limit, offset) => {
    const params = {
        dataset_table: datasetTable,
        anomaly_filter: anomalyFilter,
        run_id: runId,
        limit,
        offset,
    };
    const cacheKey = getCacheKey("/api/workbench/review-table", params);
    const cached = getCachedData(cacheKey, DATA_CACHE_TTL);
    if (cached)
        return Promise.resolve({ data: cached });
    return api.get("/api/workbench/review-table", {
        params,
    }).then((response) => {
        setCacheData(cacheKey, response.data);
        return response;
    });
};
export const getWorkbenchReviewRows = (params) => {
    const queryParams = {
        dataset_table: params.datasetTable,
        anomaly_filter: params.anomalyFilter,
        limit: params.limit,
        offset: params.offset,
        run_id: params.runId,
    };
    const cacheKey = getCacheKey("/api/workbench/review-rows", queryParams);
    const cached = getCachedData(cacheKey, DATA_CACHE_TTL);
    if (cached)
        return Promise.resolve({ data: cached });
    return api.get("/api/workbench/review-rows", {
        params: queryParams,
    }).then((response) => {
        setCacheData(cacheKey, response.data);
        return response;
    });
};
export const submitWorkbenchFeedback = (payload) => {
    const response = api.post("/api/workbench/feedback", payload);
    response.then(() => {
        clearApiCache();
    });
    return response;
};
export const generateIsolationReason = (payload) => api.post("/api/workbench/isolation-reason", payload);
export const getWorkbenchReport = (params) => {
    const queryParams = {
        dataset_table: params?.datasetTable,
        run_id: params?.runId,
    };
    const cacheKey = getCacheKey("/api/workbench/report", queryParams);
    const cached = getCachedData(cacheKey, DATA_CACHE_TTL);
    if (cached)
        return Promise.resolve({ data: cached });
    return api.get("/api/workbench/report", {
        params: queryParams,
    }).then((response) => {
        setCacheData(cacheKey, response.data);
        return response;
    });
};
