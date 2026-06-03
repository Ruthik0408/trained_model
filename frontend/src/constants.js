/**
 * Frontend Configuration Constants
 * Centralized source of truth for magic strings and configuration values
 */

// Screen IDs and Navigation
export const SCREENS = {
  WORKBENCH: "workbench",
  USER_WORKBENCH: "user-workbench",
  ANOMALIES: "anomalies",
  REVIEW: "review",
  REPORTS: "reports",
};

export const SCREEN_LABELS = {
  [SCREENS.WORKBENCH]: "Screen 1: Admin Workbench",
  [SCREENS.USER_WORKBENCH]: "Screen 2: User Workbench",
  [SCREENS.ANOMALIES]: "Screen 3: Anomaly List",
  [SCREENS.REVIEW]: "Screen 4: Review",
  [SCREENS.REPORTS]: "Screen 5: Reports",
};

// LocalStorage / SessionStorage Keys
export const STORAGE_KEYS = {
  LATEST_WORKBENCH_RUN: "tulip.latestWorkbenchRun",
};

// API Endpoints
export const API_ENDPOINTS = {
  TABLES: "/api/workbench/tables",
  PREVIEW: "/api/workbench/preview",
  RUN: "/api/workbench/run",
  DATASETS: "/api/workbench/datasets",
  REVIEW_ROWS: "/api/workbench/review-rows",
  ANOMALIES: "/api/workbench/anomalies",
};

// Cache Configuration
export const CACHE_CONFIG = {
  DATA_TTL_MS: 60 * 1000, // 1 minute
};
