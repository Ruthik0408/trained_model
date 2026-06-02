import React, { useEffect, useMemo, useRef, useState } from "react";
import Card from "../components/Card";
import PredictionRow from "../components/PredictionRow";
import { getWorkbenchDatasets, getWorkbenchReviewRows, submitWorkbenchFeedback, } from "../api/anomalyApi";
export default function ReviewPage({ latestWorkbenchRun }) {
    const PAGE_SIZE = 50;
    const [datasets, setDatasets] = useState([]);
    const [datasetTable, setDatasetTable] = useState("");
    const [anomalyFilter, setAnomalyFilter] = useState("all");
    const [pending, setPending] = useState([]);
    const [tableData, setTableData] = useState({ rows: [], total_amount: 0, total_rows: 0 });
    const [pageOffset, setPageOffset] = useState(0);
    const [loading, setLoading] = useState(true);
    const [loadError, setLoadError] = useState("");
    const [focusedPredictionId, setFocusedPredictionId] = useState(null);
    const [activeSlideIndex, setActiveSlideIndex] = useState(0);
    const [activeSlideHeight, setActiveSlideHeight] = useState(null);
    const cardRefs = useRef({});
    const slideTrackRef = useRef(null);
    const hydratedLatestDataset = useMemo(() => {
        if (!latestWorkbenchRun?.datasetTable)
            return null;
        return {
            dataset_table: latestWorkbenchRun.datasetTable,
            run_id: latestWorkbenchRun.runId,
            selected_tables: latestWorkbenchRun.selectedTables || [],
            run_name: latestWorkbenchRun.runName || "Latest workbench run",
            total_rows: latestWorkbenchRun.totalRows || 0,
            final_anomaly_count: latestWorkbenchRun.finalAnomalyCount || 0,
        };
    }, [latestWorkbenchRun]);
    const hydratedRunIdKey = hydratedLatestDataset?.run_id != null ? String(hydratedLatestDataset.run_id) : null;
    useEffect(() => {
        if (latestWorkbenchRun?.datasetTable) {
            setDatasetTable(latestWorkbenchRun.datasetTable);
            setPageOffset(0);
        }
    }, [latestWorkbenchRun]);
    useEffect(() => {
        setPageOffset(0);
    }, [datasetTable, anomalyFilter]);
    useEffect(() => {
        setActiveSlideIndex(0);
        setFocusedPredictionId(null);
        if (slideTrackRef.current) {
            slideTrackRef.current.scrollTo({ left: 0, behavior: "auto" });
        }
    }, [datasetTable, anomalyFilter, pageOffset]);
    useEffect(() => {
        const load = async () => {
            setLoading(true);
            setLoadError("");
            try {
                const preferredRunId = hydratedLatestDataset?.dataset_table === datasetTable ? hydratedLatestDataset.run_id : undefined;
                const reviewRowsResponse = await getWorkbenchReviewRows({
                    datasetTable: datasetTable || undefined,
                    anomalyFilter,
                    limit: PAGE_SIZE,
                    offset: pageOffset,
                    runId: preferredRunId,
                });
                const responseDataset = reviewRowsResponse.data?.dataset_table ||
                    hydratedLatestDataset?.dataset_table ||
                    "";
                if (!datasetTable && responseDataset) {
                    setDatasetTable(responseDataset);
                }
                const nextRows = reviewRowsResponse.data?.rows || [];
                setPending(nextRows);
                setTableData(buildReviewTableData(nextRows, reviewRowsResponse.data?.summary, pageOffset));
                void getWorkbenchDatasets()
                    .then((datasetsResponse) => {
                    const fetchedDatasets = datasetsResponse.data || [];
                    const nextDatasets = [...fetchedDatasets];
                    if (hydratedLatestDataset &&
                        !nextDatasets.some((item) => item.dataset_table === hydratedLatestDataset.dataset_table &&
                            String(item.run_id) === hydratedRunIdKey)) {
                        nextDatasets.unshift(hydratedLatestDataset);
                    }
                    setDatasets(nextDatasets);
                })
                    .catch(() => {
                    if (hydratedLatestDataset) {
                        setDatasets((current) => current.length > 0 ? current : [hydratedLatestDataset]);
                    }
                });
            }
            catch (error) {
                const detail = error?.response?.data?.detail ||
                    error?.message ||
                    "Unable to load review data right now.";
                setPending([]);
                setTableData({ rows: [], total_amount: 0, total_rows: 0 });
                setLoadError(typeof detail === "string" ? detail : "Unable to load review data right now.");
            }
            finally {
                setLoading(false);
            }
        };
        load();
    }, [datasetTable, anomalyFilter, hydratedLatestDataset, hydratedRunIdKey, pageOffset]);
    const activeDataset = useMemo(() => datasets.find((item) => item.dataset_table === datasetTable &&
        (hydratedLatestDataset?.dataset_table !== datasetTable ||
            hydratedRunIdKey === null ||
            String(item.run_id) === hydratedRunIdKey)) ||
        (hydratedLatestDataset?.dataset_table === datasetTable ? hydratedLatestDataset : null) ||
        datasets.find((item) => item.dataset_table === datasetTable) ||
        hydratedLatestDataset, [datasetTable, datasets, hydratedLatestDataset, hydratedRunIdKey]);
    const activeTables = activeDataset?.selected_tables || [];
    const activeRunId = activeDataset?.run_id;
    useEffect(() => {
        if (pending.length === 0) {
            setActiveSlideHeight(null);
            return;
        }
        const safeIndex = Math.min(activeSlideIndex, pending.length - 1);
        const activePredictionId = pending[safeIndex]?.prediction_id;
        const activeNode = activePredictionId ? cardRefs.current[activePredictionId] : null;
        if (!activeNode) {
            return;
        }
        const syncHeight = () => {
            setActiveSlideHeight(activeNode.getBoundingClientRect().height);
        };
        syncHeight();
        if (typeof ResizeObserver === "undefined") {
            return;
        }
        const observer = new ResizeObserver(() => {
            syncHeight();
        });
        const observedNode = activePredictionId ? cardRefs.current[activePredictionId] : null;
        if (!observedNode) {
            return;
        }
        observer.observe(observedNode);
        return () => observer.disconnect();
    }, [activeSlideIndex, pending]);
    const currentPage = Math.floor(pageOffset / PAGE_SIZE) + 1;
    const totalPages = Math.max(1, Math.ceil((tableData.total_rows || 0) / PAGE_SIZE));
    const canGoPrevious = pageOffset > 0;
    const canGoNext = pageOffset + PAGE_SIZE < (tableData.total_rows || 0);
    const displayRuleCount = anomalyFilter === "rule"
        ? tableData.total_rows || 0
        : latestWorkbenchRun?.userRuleCount ?? latestWorkbenchRun?.userOutlierCount ?? 0;
    const displayMlCount = anomalyFilter === "ml"
        ? tableData.total_rows || 0
        : latestWorkbenchRun?.mlAnomalyCount || 0;
    const displayFinalCount = anomalyFilter === "all"
        ? tableData.total_rows || 0
        : latestWorkbenchRun?.finalAnomalyCount || activeDataset?.final_anomaly_count || 0;
    function handlePreviousPage() {
        setPageOffset((value) => Math.max(0, value - PAGE_SIZE));
    }
    function handleNextPage() {
        setPageOffset((value) => {
            const nextOffset = value + PAGE_SIZE;
            const totalRows = tableData.total_rows || 0;
            if (totalRows <= 0 || nextOffset >= totalRows) {
                return value;
            }
            return nextOffset;
        });
    }
    async function refreshCurrent() {
        try {
            setLoadError("");
            const reviewRowsResponse = await getWorkbenchReviewRows({
                datasetTable: datasetTable || undefined,
                anomalyFilter,
                limit: PAGE_SIZE,
                offset: pageOffset,
                runId: activeRunId,
            });
            const nextRows = reviewRowsResponse.data?.rows || [];
            setPending(nextRows);
            setTableData(buildReviewTableData(nextRows, reviewRowsResponse.data?.summary, pageOffset));
            return nextRows;
        }
        catch (error) {
            const detail = error?.response?.data?.detail ||
                error?.message ||
                "Unable to refresh review data right now.";
            setLoadError(typeof detail === "string" ? detail : "Unable to refresh review data right now.");
            return [];
        }
    }
    async function handleFeedback(predictionId, sourceRecordId, feedback) {
        if (!datasetTable)
            return;
        const currentIndex = pending.findIndex((item) => item.prediction_id === predictionId);
        const nextOriginalPredictionId = pending[currentIndex + 1]?.prediction_id;
        await submitWorkbenchFeedback({
            dataset_table: datasetTable,
            record_id: predictionId,
            source_record_id: sourceRecordId,
            feedback,
        });
        const nextRows = await refreshCurrent();
        const fallbackIndex = Math.min(Math.max(currentIndex, 0), Math.max(nextRows.length - 1, 0));
        const nextPredictionId = nextRows.find((item) => item.prediction_id === nextOriginalPredictionId)?.prediction_id ||
            nextRows[fallbackIndex]?.prediction_id;
        if (nextPredictionId) {
            setFocusedPredictionId(nextPredictionId);
            requestAnimationFrame(() => {
                cardRefs.current[nextPredictionId]?.scrollIntoView({
                    behavior: "smooth",
                    block: "nearest",
                    inline: "start",
                });
            });
        }
    }
    function handleSlideTrackScroll(event) {
        const viewportWidth = event.currentTarget.clientWidth;
        if (viewportWidth <= 0)
            return;
        const nextIndex = Math.round(event.currentTarget.scrollLeft / viewportWidth);
        if (nextIndex !== activeSlideIndex) {
            setActiveSlideIndex(nextIndex);
        }
    }
    const slideTrackStyle = {
        ...slideTrack,
        ...(activeSlideHeight ? { height: activeSlideHeight + 10 } : {}),
    };
    return (<div style={page}>
      <div style={header}>
        <div>
          <h1 style={title}>User Review And Feedback</h1>
        </div>

      </div>

      {activeDataset?.run_name ? (<Card>
          <div style={runInlineHeader}>
            <div style={runInlineTitle}>Latest Run Result</div>
            <div style={runInlineGroup}>
              <label style={controlLabel}>Anomaly View</label>
              <select value={anomalyFilter} onChange={(event) => setAnomalyFilter(event.target.value)} style={runInlineSelect}>
                <option value="all">Rule or ML anomalies</option>
                <option value="rule">Rule-based only</option>
                <option value="ml">ML-based only</option>
                <option value="not_reviewed">Not reviewed rows only</option>
                <option value="reviewed">Reviewed rows only</option>
              </select>
            </div>
            <div style={runInlineGroup}>
              <div style={controlLabel}>Feedback Slides</div>
              <div style={slideCount}>
                {tableData.total_rows || pending.length} cards
                {" · "}
                Page {currentPage} of {totalPages}
              </div>
            </div>
          </div>
          <div style={runResultGrid}>
            <div style={runResultCard}><strong>Total rows</strong><div>{latestWorkbenchRun?.totalRows || activeDataset.total_rows || 0}</div></div>
            <div style={runResultCard}><strong>Rule Anomaly</strong><div>{displayRuleCount}</div></div>
            <div style={runResultCard}><strong>ML anomalies</strong><div>{displayMlCount}</div></div>
            <div style={runResultCard}><strong>Final anomalies</strong><div>{displayFinalCount}</div></div>
            <div style={runResultCard}><strong>Amount total</strong><div>{Number(latestWorkbenchRun?.amountTotal || tableData.total_amount || 0).toLocaleString()}</div></div>
          </div>
        </Card>) : null}

      {loading ? <div>Loading review rows...</div> : null}
      {!loading && loadError ? <div style={errorBox}>{loadError}</div> : null}

      <div style={pagerRow}>
        <button type="button" onClick={handlePreviousPage} disabled={!canGoPrevious || loading} style={pagerButton}>
          Previous
        </button>
        <div style={pagerMeta}>
          Showing {pending.length} rows starting at {tableData.total_rows ? pageOffset + 1 : 0}
        </div>
        <button type="button" onClick={handleNextPage} disabled={!canGoNext || loading} style={pagerButton}>
          Next
        </button>
      </div>

      <div ref={slideTrackRef} style={slideTrackStyle} onScroll={handleSlideTrackScroll}>
        {pending.map((item, index) => (<div key={item.prediction_id} ref={(node) => {
                cardRefs.current[item.prediction_id] = node;
            }} style={focusedPredictionId === item.prediction_id ? focusedSlide : slide}>
            <PredictionRow item={item} onAction={handleFeedback} selectedTables={activeTables} />
          </div>))}
      </div>
    </div>);
}
const page = { display: "grid", gap: 18 };
const header = { display: "flex", justifyContent: "space-between", gap: 18, alignItems: "flex-start", flexWrap: "wrap" };
const title = { margin: "8px 0", fontSize: 36, color: "#111827" };
const button = { border: "none", borderRadius: 14, background: "#0f766e", color: "#fff", padding: "12px 16px", fontWeight: 800, cursor: "pointer" };
const controlLabel = { fontWeight: 700, color: "#9a3412" };
const runResultGrid = { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 8 };
const runResultCard = { borderRadius: 14, border: "1px solid #e5e7eb", background: "#f8fafc", padding: "10px 14px", display: "grid", gap: 4 };
const runInlineHeader = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 18, flexWrap: "wrap", marginBottom: 10 };
const runInlineTitle = { fontSize: 28, fontWeight: 800, color: "#111827", lineHeight: 1.1 };
const runInlineGroup = { display: "flex", alignItems: "center", gap: 10, flexWrap: "nowrap", minWidth: 0 };
const runInlineSelect = { borderRadius: 12, border: "1px solid #cbd5e1", padding: "8px 12px", fontSize: 15, minWidth: 260 };
const runFilterInput = { borderRadius: 14, border: "1px solid #cbd5e1", padding: "10px 14px", font: "inherit", minWidth: 0 };
const successBox = { marginBottom: 12, borderRadius: 14, background: "#ecfdf5", border: "1px solid #10b981", color: "#065f46", padding: "10px 12px", fontSize: 13, lineHeight: 1.5, fontWeight: 800 };
const queueTitle = { fontSize: 16, fontWeight: 700, color: "#111827" };
const slideCount = { borderRadius: 999, background: "#fffbeb", border: "1px solid #fed7aa", color: "#9a3412", padding: "8px 12px", fontWeight: 800, fontSize: 13 };
const pagerRow = { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" };
const pagerButton = { border: "1px solid #fdba74", borderRadius: 12, background: "#fff", color: "#9a3412", padding: "9px 14px", fontWeight: 700, cursor: "pointer" };
const pagerMeta = { color: "#6b7280", fontSize: 13, fontWeight: 600 };
const slideTrack = { display: "flex", alignItems: "flex-start", gap: 0, overflowX: "auto", padding: "4px 0 12px", scrollSnapType: "x mandatory", scrollBehavior: "smooth" };
const slide = { flex: "0 0 100%", minWidth: "100%", boxSizing: "border-box", scrollSnapAlign: "start", paddingRight: 4, transition: "transform 160ms ease, box-shadow 160ms ease" };
const focusedSlide = { ...slide, transform: "translateY(-2px)", boxShadow: "0 0 0 3px rgba(245, 158, 11, 0.25)", borderRadius: 24 };
const errorBox = { borderRadius: 12, border: "1px solid #fecaca", background: "#fef2f2", color: "#b91c1c", padding: "12px 14px" };

function buildReviewTableData(rows, summary, offset) {
    let runningTotal = 0;
    return {
        rows: rows.map((row, index) => {
            const amount = resolveReviewAmount(row.row_payload_json || {});
            runningTotal += amount;
            return {
                serial_no: offset + index + 1,
                prediction_id: row.prediction_id,
                anomaly: row.final_label ? "Yes" : "No",
                amount,
                total_amount: runningTotal,
                review_status: row.review_status,
                feedback: row.feedback,
            };
        }),
        total_amount: Number(summary?.total_amount || 0),
        total_rows: Number(summary?.total_rows || 0),
    };
}

function resolveReviewAmount(payload) {
    const aliasSet = new Set(["amount", "amount_passed", "amount_claimed", "invoice_amount", "schedule3_amount"]);
    for (const alias of aliasSet) {
        const direct = payload[alias];
        if (direct != null && direct !== "") {
            const numeric = Number(direct);
            if (!Number.isNaN(numeric)) {
                return numeric;
            }
        }
    }
    for (const [key, value] of Object.entries(payload)) {
        if (value == null || value === "") {
            continue;
        }
        const plainKey = key.split(".").pop()?.toLowerCase();
        if (plainKey && aliasSet.has(plainKey)) {
            const numeric = Number(value);
            if (!Number.isNaN(numeric)) {
                return numeric;
            }
        }
    }
    return 0;
}
