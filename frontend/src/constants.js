/**
 * Frontend Configuration Constants
 * Centralized source of truth for magic strings and configuration values
 */

// Screen IDs and Navigation
export const SCREENS = {
  WORKBENCH: "workbench",
  REVIEW: "review",
  REPORTS: "reports",
};

export const SCREEN_LABELS = {
  [SCREENS.WORKBENCH]: "Screen 1: Workbench",
  [SCREENS.REVIEW]: "Screen 2: Review",
  [SCREENS.REPORTS]: "Screen 3: Reports",
};

// LocalStorage / SessionStorage Keys
export const STORAGE_KEYS = {
  LATEST_WORKBENCH_RUN: "tulip.latestWorkbenchRun",
  API_CACHE: "tulip.apiCache",
};

// API Endpoints
export const API_ENDPOINTS = {
  TABLES: "/api/workbench/tables",
  CONNECTION: "/api/workbench/connection",
  DEFAULT_RULES: "/api/workbench/default-feature-rules",
  PREVIEW: "/api/workbench/preview",
  RUN: "/api/workbench/run",
  DATASETS: "/api/workbench/datasets",
  REVIEW_TABLE: "/api/workbench/review-table",
  REVIEW_ROWS: "/api/workbench/review-rows",
  REPORT_DATA: "/api/workbench/report-data",
  ANOMALY_REASON: "/api/workbench/anomaly-reason",
};

// Cache Configuration
export const CACHE_CONFIG = {
  DATA_TTL_MS: 60 * 1000, // 1 minute
  API_TIMEOUT_MS: 30 * 1000, // 30 seconds
};

// UI Messages and Status Strings
export const UI_MESSAGES = {
  LOADING: "Loading...",
  ERROR: "Something went wrong",
  RETRY: "Retry",
  RELOAD: "Reload Page",
  CONNECTION_ERROR: "Cannot connect to backend",
  RATE_LIMIT: "Too many requests - please wait",
  TIMEOUT: "Request took too long - please retry",
};

// HTTP Status Codes
export const HTTP_STATUS = {
  OK: 200,
  BAD_REQUEST: 400,
  NOT_FOUND: 404,
  RATE_LIMITED: 429,
  SERVER_ERROR: 500,
  SERVICE_UNAVAILABLE: 503,
};

// Anomaly Filters
export const ANOMALY_FILTERS = {
  ALL: "all",
  HUMAN_ONLY: "human",
  ML_ONLY: "ml",
  COMBINED: "combined",
};

// Pagination Defaults
export const PAGINATION = {
  DEFAULT_LIMIT: 50,
  DEFAULT_OFFSET: 0,
};
