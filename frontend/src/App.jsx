import React, { useState } from "react";
import WorkbenchPage from "./pages/WorkbenchPage";
import ReviewPage from "./pages/ReviewPage";
import ReportsPage from "./pages/ReportsPage";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { SCREENS, SCREEN_LABELS, STORAGE_KEYS } from "./constants";

const screens = [
  { id: SCREENS.WORKBENCH, label: SCREEN_LABELS[SCREENS.WORKBENCH] },
  { id: SCREENS.REVIEW, label: SCREEN_LABELS[SCREENS.REVIEW] },
  { id: SCREENS.REPORTS, label: SCREEN_LABELS[SCREENS.REPORTS] },
];

function loadLatestWorkbenchRun() {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const rawValue = window.sessionStorage.getItem(STORAGE_KEYS.LATEST_WORKBENCH_RUN);
    if (!rawValue) {
      return null;
    }
    return JSON.parse(rawValue);
  }
  catch {
    return null;
  }
}
export default function App() {
    const [activeTab, setActiveTab] = useState("workbench");
    const [latestWorkbenchRun, setLatestWorkbenchRun] = useState(() => loadLatestWorkbenchRun());
    const activeIndex = screens.findIndex((screen) => screen.id === activeTab);
    const handleRunComplete = (run) => {
        setLatestWorkbenchRun(run);
        if (typeof window !== "undefined") {
            window.sessionStorage.setItem(STORAGE_KEYS.LATEST_WORKBENCH_RUN, JSON.stringify(run));
        }
    };
    const goPrevious = () => {
        if (activeIndex <= 0) {
            return;
        }
        setActiveTab(screens[activeIndex - 1].id);
    };
    const goNext = () => {
        if (activeIndex >= screens.length - 1) {
            return;
        }
        setActiveTab(screens[activeIndex + 1].id);
    };
    return (
    <ErrorBoundary>
      <div style={page}>
        <div style={shell}>
          {activeTab === SCREENS.WORKBENCH ? (
            <div style={nav}>
              <div>
                <div style={kicker}>Tulip 2.0</div>
                <h1 style={title}>ANOMALY DETECTION</h1>
              </div>
            </div>
          ) : null}

          {activeIndex > 0 ? (
            <button
              type="button"
              onClick={goPrevious}
              style={{ ...navArrow, ...navArrowLeft }}
              aria-label="Go to previous screen"
            >
              ‹
            </button>
          ) : null}
          {activeIndex < screens.length - 1 ? (
            <button
              type="button"
              onClick={goNext}
              style={{ ...navArrow, ...navArrowRight }}
              aria-label="Go to next screen"
            >
              ›
            </button>
          ) : null}

          <div style={activeTab === SCREENS.WORKBENCH ? visibleScreen : hiddenScreen}>
            <WorkbenchPage onRunComplete={handleRunComplete} />
          </div>
          <div style={activeTab === SCREENS.REVIEW ? visibleScreen : hiddenScreen}>
            <ReviewPage latestWorkbenchRun={latestWorkbenchRun} />
          </div>
          <div style={activeTab === SCREENS.REPORTS ? visibleScreen : hiddenScreen}>
            <ReportsPage latestWorkbenchRun={latestWorkbenchRun} />
          </div>
        </div>
      </div>
    </ErrorBoundary>
    );
}
const page = {
    minHeight: "100vh",
    background: "radial-gradient(circle at top left, rgba(59, 130, 246, 0.15), transparent 24%), radial-gradient(circle at top right, rgba(249, 115, 22, 0.18), transparent 22%), linear-gradient(180deg, #f8fafc 0%, #eff6ff 100%)",
    padding: "28px 20px 40px",
};
const shell = {
    maxWidth: 1400,
    margin: "0 auto",
    display: "grid",
    gap: 18,
    position: "relative",
};
const nav = {
    background: "rgba(255,255,255,0.78)",
    backdropFilter: "blur(14px)",
    border: "1px solid rgba(148, 163, 184, 0.22)",
    borderRadius: 28,
    padding: 24,
};
const kicker = {
    textTransform: "uppercase",
    letterSpacing: "0.18em",
    color: "#2563eb",
    fontWeight: 800,
    fontSize: 12,
};
const title = {
    margin: "8px 0 0",
    fontSize: 34,
    color: "#0f172a",
};
const navArrow = {
    position: "fixed",
    top: "50%",
    transform: "translateY(-50%)",
    width: 54,
    height: 54,
    borderRadius: 999,
    border: "1px solid rgba(148, 163, 184, 0.35)",
    background: "rgba(255,255,255,0.92)",
    color: "#0f172a",
    fontSize: 34,
    lineHeight: 1,
    cursor: "pointer",
    boxShadow: "0 14px 30px rgba(15, 23, 42, 0.12)",
    zIndex: 20,
};
const navArrowLeft = {
    left: 18,
};
const navArrowRight = {
    right: 18,
};
const visibleScreen = {
    display: "block",
};
const hiddenScreen = {
    display: "none",
};
