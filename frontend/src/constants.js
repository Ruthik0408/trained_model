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
};

// API Endpoints
export const API_ENDPOINTS = {
  TABLES: "/api/workbench/tables",
  PREVIEW: "/api/workbench/preview",
  RUN: "/api/workbench/run",
  DATASETS: "/api/workbench/datasets",
  REVIEW_ROWS: "/api/workbench/review-rows",
};

// Cache Configuration
export const CACHE_CONFIG = {
  DATA_TTL_MS: 60 * 1000, // 1 minute
};
