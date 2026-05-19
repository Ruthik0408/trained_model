import React, { useEffect, useMemo, useRef, useState } from "react";
import Card from "../components/Card";
import PredictionRow from "../components/PredictionRow";
import { generateIsolationReasonsBatch, getWorkbenchDatasets, getWorkbenchReviewRows, submitWorkbenchFeedback, } from "../api/anomalyApi";
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
    const [reasonMap, setReasonMap] = useState({});
    const [reasonLoadingIds, setReasonLoadingIds] = useState(new Set());
    const [reasonErrors, setReasonErrors] = useState({});
    const [showTable, setShowTable] = useState(false);
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
            setShowTable(true);
        }
    }, [latestWorkbenchRun]);
    useEffect(() => {
        setPageOffset(0);
    }, [datasetTable, anomalyFilter]);
    useEffect(() => {
        setActiveSlideIndex(0);
        setFocusedPredictionId(null);
        setReasonMap({});
        setReasonLoadingIds(new Set());
        setReasonErrors({});
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
        const rowsToExplain = pending
            .filter((item) => shouldBatchExplainRow(item))
            .filter((item) => !reasonMap[item.prediction_id]);
        if (rowsToExplain.length === 0) {
            setReasonLoadingIds(new Set());
            return;
        }
        const controller = new AbortController();
        const loadingIds = new Set(rowsToExplain.map((item) => item.prediction_id));
        setReasonLoadingIds(loadingIds);
        setReasonErrors({});
        generateIsolationReasonsBatch({
            rows: rowsToExplain.map((item) => buildBatchReasonPayload(item)),
        }, controller.signal)
            .then((response) => {
            const nextReasons = response?.data?.reasons || {};
            const nextErrors = response?.data?.errors || {};
            setReasonMap((current) => ({
                ...current,
                ...Object.fromEntries(Object.entries(nextReasons).map(([predictionId, result]) => [
                    predictionId,
                    result?.reason || "",
                ])),
            }));
            setReasonErrors(Object.fromEntries(Object.entries(nextErrors).map(([predictionId, detail]) => [
                predictionId,
                typeof detail === "string" ? detail : "Unable to generate anomaly explanation right now.",
            ])));
        })
            .catch((error) => {
            if (error?.name === "CanceledError" || error?.code === "ERR_CANCELED") {
                return;
            }
            const detail = error?.response?.data?.detail ||
                error?.message ||
                "Unable to generate anomaly explanations right now.";
            setReasonErrors(Object.fromEntries(rowsToExplain.map((item) => [
                item.prediction_id,
                typeof detail === "string" ? detail : "Unable to generate anomaly explanations right now.",
            ])));
        })
            .finally(() => {
            setReasonLoadingIds(new Set());
        });
        return () => controller.abort();
    }, [pending, reasonMap]);
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
    function jumpToPrediction(predictionId) {
        const nextIndex = pending.findIndex((item) => item.prediction_id === predictionId);
        if (nextIndex >= 0) {
            setActiveSlideIndex(nextIndex);
        }
        setFocusedPredictionId(predictionId);
        setShowTable(false);
        requestAnimationFrame(() => {
            cardRefs.current[predictionId]?.scrollIntoView({
                behavior: "smooth",
                block: "nearest",
                inline: "start",
            });
        });
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
        ...(activeSlideHeight ? { height: activeSlideHeight + 22 } : {}),
    };
    return (<div style={page}>
      <div style={header}>
        <div>
          <h1 style={title}>Human Review And Feedback</h1>
        </div>

      </div>

      {activeDataset?.run_name ? (<Card title="Latest Run Result">
          <div style={runCardHeader}>
            <div style={runFilterBlock}>
              <label style={controlLabel}>Anomaly View</label>
              <select value={anomalyFilter} onChange={(event) => setAnomalyFilter(event.target.value)} style={runFilterInput}>
                <option value="all">Rule or ML anomalies</option>
                <option value="rule">Rule-based only</option>
                <option value="ml">ML-based only</option>
                <option value="not_reviewed">Not reviewed rows only</option>
                <option value="reviewed">Reviewed rows only</option>
              </select>
            </div>
            <button type="button" onClick={() => setShowTable((value) => !value)} style={toggleButton}>
              {showTable ? "Hide Summary Table" : "Show Summary Table"}
            </button>
          </div>
          <div style={runResultGrid}>
            <div style={runResultCard}><strong>Total rows</strong><div>{latestWorkbenchRun?.totalRows || activeDataset.total_rows || 0}</div></div>
            <div style={runResultCard}><strong>Human outliers</strong><div>{latestWorkbenchRun?.humanOutlierCount || 0}</div></div>
            <div style={runResultCard}><strong>ML anomalies</strong><div>{latestWorkbenchRun?.mlAnomalyCount || 0}</div></div>
            <div style={runResultCard}><strong>Final anomalies</strong><div>{latestWorkbenchRun?.finalAnomalyCount || activeDataset.final_anomaly_count || 0}</div></div>
            <div style={runResultCard}><strong>Amount total</strong><div>{Number(latestWorkbenchRun?.amountTotal || tableData.total_amount || 0).toLocaleString()}</div></div>
          </div>
        </Card>) : null}

      {loading ? <div>Loading review rows...</div> : null}
      {!loading && loadError ? <div style={errorBox}>{loadError}</div> : null}

      <div style={reviewSlideHeader}>
        <div style={queueTitle}>Feedback Slides</div>
        <div style={slideCount}>
          {tableData.total_rows || pending.length} cards
          {" · "}
          Page {currentPage} of {totalPages}
        </div>
      </div>

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
            <PredictionRow item={item} onAction={handleFeedback} selectedTables={activeTables} llmReason={reasonMap[item.prediction_id] || ""} llmReasonLoading={reasonLoadingIds.has(item.prediction_id)} llmReasonError={reasonErrors[item.prediction_id] || ""}/>
          </div>))}
      </div>

      {showTable ? (<Card title="Review Summary Table">
          <div style={tableWrap}>
            <table style={table}>
              <thead>
                <tr>
                  <th style={th}>S.No</th>
                  <th style={th}>Anomaly</th>
                  <th style={th}>Amount</th>
                  <th style={th}>Total Amount</th>
                  <th style={th}>Status</th>
                  <th style={th}>Feedback</th>
                </tr>
              </thead>
              <tbody>
                {tableData.rows.map((row) => (<tr key={row.prediction_id} style={focusedPredictionId === row.prediction_id ? focusedTableRow : clickableRow} onClick={() => jumpToPrediction(row.prediction_id)}>
                    <td style={td}>{row.serial_no}</td>
                    <td style={td}>{row.anomaly}</td>
                    <td style={td}>{Number(row.amount || 0).toLocaleString()}</td>
                    <td style={td}>{Number(row.total_amount || 0).toLocaleString()}</td>
                    <td style={td}>{row.review_status || "PENDING_REVIEW"}</td>
                    <td style={td}>{row.feedback || "pending"}</td>
                  </tr>))}
              </tbody>
            </table>
          </div>
        </Card>) : null}
    </div>);
}
const page = { display: "grid", gap: 18 };
const header = { display: "flex", justifyContent: "space-between", gap: 18, alignItems: "flex-start", flexWrap: "wrap" };
const title = { margin: "8px 0", fontSize: 36, color: "#111827" };
const button = { border: "none", borderRadius: 14, background: "#0f766e", color: "#fff", padding: "12px 16px", fontWeight: 800, cursor: "pointer" };
const controlLabel = { fontWeight: 700, color: "#9a3412" };
const runResultGrid = { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 8 };
const runResultCard = { borderRadius: 14, border: "1px solid #e5e7eb", background: "#f8fafc", padding: "10px 14px", display: "grid", gap: 4 };
const runCardHeader = { display: "flex", justifyContent: "space-between", gap: 10, alignItems: "end", flexWrap: "wrap", marginBottom: 8 };
const runFilterBlock = { display: "grid", gap: 4, minWidth: "min(100%, 280px)" };
const runFilterInput = { borderRadius: 14, border: "1px solid #cbd5e1", padding: "10px 14px", font: "inherit", minWidth: 0 };
const successBox = { marginBottom: 12, borderRadius: 14, background: "#ecfdf5", border: "1px solid #10b981", color: "#065f46", padding: "10px 12px", fontSize: 13, lineHeight: 1.5, fontWeight: 800 };
const queueTitle = { fontSize: 16, fontWeight: 700, color: "#111827" };
const toggleButton = { border: "1px solid #fdba74", borderRadius: 12, background: "#fff7ed", color: "#9a3412", padding: "8px 14px", fontWeight: 700, cursor: "pointer" };
const reviewSlideHeader = { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16, flexWrap: "wrap" };
const slideCount = { borderRadius: 999, background: "#fffbeb", border: "1px solid #fed7aa", color: "#9a3412", padding: "8px 12px", fontWeight: 800, fontSize: 13 };
const pagerRow = { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" };
const pagerButton = { border: "1px solid #fdba74", borderRadius: 12, background: "#fff", color: "#9a3412", padding: "9px 14px", fontWeight: 700, cursor: "pointer" };
const pagerMeta = { color: "#6b7280", fontSize: 13, fontWeight: 600 };
const slideTrack = { display: "flex", alignItems: "flex-start", gap: 0, overflowX: "auto", padding: "4px 0 18px", scrollSnapType: "x mandatory", scrollBehavior: "smooth" };
const slide = { flex: "0 0 100%", minWidth: "100%", boxSizing: "border-box", scrollSnapAlign: "start", paddingRight: 4, transition: "transform 160ms ease, box-shadow 160ms ease" };
const focusedSlide = { ...slide, transform: "translateY(-2px)", boxShadow: "0 0 0 3px rgba(245, 158, 11, 0.25)", borderRadius: 24 };
const tableWrap = { overflowX: "auto" };
const table = { width: "100%", borderCollapse: "collapse" };
const th = { textAlign: "left", padding: "12px 14px", borderBottom: "1px solid #e5e7eb", color: "#9a3412", fontSize: 13 };
const td = { padding: "12px 14px", borderBottom: "1px solid #f3f4f6", color: "#111827", fontSize: 14 };
const clickableRow = { cursor: "pointer" };
const focusedTableRow = { background: "#fff7ed", cursor: "pointer" };
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

function shouldBatchExplainRow(item) {
    const reasons = item?.reasons_json || {};
    return Boolean(reasons.ml_anomaly_flag && Array.isArray(reasons.ml_feature_signals));
}

function buildBatchReasonPayload(item) {
    const reasons = item?.reasons_json || {};
    const payload = item?.row_payload_json || {};
    const baseReasons = Array.isArray(reasons.reason_list)
        ? reasons.reason_list.filter(Boolean)
        : item?.rule_codes
            ? [item.rule_codes]
            : [];
    return {
        prediction_id: item?.prediction_id,
        dataset_table: item?.dataset_table,
        review_key: String(payload.review_key || ""),
        if_score: toNullableNumber(firstDefined(reasons.if_score, payload.if_score, reasons.isolation_score, item.raw_ml_score, item.ml_score)),
        ml_threshold: toNullableNumber(firstDefined(reasons.ml_threshold, payload.ml_threshold, item.ml_threshold)),
        rule_anomaly: Boolean(reasons.rule_anomaly ?? payload.rule_anomaly ?? item.rule_flag ?? reasons.human_outlier_flag ?? false),
        rule_count: Number(reasons.rule_count ?? payload.rule_count ?? baseReasons.length ?? 0),
        existing_reasons: baseReasons,
        feature_signals: Array.isArray(reasons.ml_feature_signals) ? reasons.ml_feature_signals : [],
        row_payload: payload,
    };
}

function firstDefined(...values) {
    for (const value of values) {
        if (value !== undefined && value !== null && value !== "") {
            return value;
        }
    }
    return undefined;
}

function toNullableNumber(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
}
