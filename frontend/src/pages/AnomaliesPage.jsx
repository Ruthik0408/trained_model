import React, { useEffect, useMemo, useState } from "react";
import { getWorkbenchAnomalies } from "../api/anomalyApi";

const PAGE_SIZE = 50;

export default function AnomaliesPage() {
  const [tableFilter, setTableFilter] = useState("");
  const [anomalyType, setAnomalyType] = useState("all");
  const [reviewStatus, setReviewStatus] = useState("all");
  const [pageOffset, setPageOffset] = useState(0);
  const [data, setData] = useState({
    rows: [],
    summary: { total_rows: 0, reviewed_rows: 0, not_reviewed_rows: 0 },
    table_options: [],
    pagination: { limit: PAGE_SIZE, offset: 0, page_count: 0 },
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    setPageOffset(0);
  }, [tableFilter, anomalyType, reviewStatus]);

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      setLoading(true);
      setError("");
      try {
        const response = await getWorkbenchAnomalies(
          {
            tableFilter: tableFilter || undefined,
            anomalyType,
            reviewStatus,
            limit: PAGE_SIZE,
            offset: pageOffset,
          },
          controller.signal,
        );
        setData(response.data || {});
      }
      catch (nextError) {
        if (nextError?.name === "CanceledError" || nextError?.code === "ERR_CANCELED") {
          return;
        }
        const detail = nextError?.response?.data?.detail || nextError?.message;
        setError(typeof detail === "string" ? detail : "Unable to load anomalies.");
      }
      finally {
        setLoading(false);
      }
    }
    load();
    return () => controller.abort();
  }, [tableFilter, anomalyType, reviewStatus, pageOffset]);

  const rows = data?.rows || [];
  const summary = data?.summary || {};
  const tableOptions = data?.table_options || [];
  const totalRows = Number(summary.total_rows || 0);
  const currentPage = Math.floor(pageOffset / PAGE_SIZE) + 1;
  const totalPages = Math.max(1, Math.ceil(totalRows / PAGE_SIZE));
  const canGoPrevious = pageOffset > 0;
  const canGoNext = pageOffset + PAGE_SIZE < totalRows;
  const activeTableLabel = useMemo(() => {
    if (!tableFilter) {
      return "All tables";
    }
    return tableOptions.find((item) => item.value === tableFilter)?.label || tableFilter;
  }, [tableFilter, tableOptions]);

  return (
    <main style={page}>
      <header style={header}>
        <div>
          <div style={eyebrow}>ML_Features</div>
          <h1 style={title}>Anomaly List</h1>
        </div>
        <div style={summaryGrid}>
          <Metric label="Rows" value={totalRows} />
          <Metric label="Reviewed" value={summary.reviewed_rows || 0} />
          <Metric label="Not reviewed" value={summary.not_reviewed_rows || 0} />
        </div>
      </header>

      <section style={filterBar}>
        <label style={filterField}>
          <span style={filterLabel}>Table</span>
          <select value={tableFilter} onChange={(event) => setTableFilter(event.target.value)} style={select}>
            <option value="">All tables</option>
            {tableOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label} ({option.row_count})
              </option>
            ))}
          </select>
        </label>

        <label style={filterField}>
          <span style={filterLabel}>Anomaly type</span>
          <select value={anomalyType} onChange={(event) => setAnomalyType(event.target.value)} style={select}>
            <option value="all">Rule or ML</option>
            <option value="rule">Rule only</option>
            <option value="ml">ML only</option>
            <option value="rule_and_ml">Rule + ML</option>
          </select>
        </label>

        <label style={filterField}>
          <span style={filterLabel}>Review status</span>
          <select value={reviewStatus} onChange={(event) => setReviewStatus(event.target.value)} style={select}>
            <option value="all">All statuses</option>
            <option value="reviewed">Reviewed</option>
            <option value="not_reviewed">Not reviewed</option>
          </select>
        </label>

        <div style={filterContext}>
          <span>{activeTableLabel}</span>
          <strong>Page {currentPage} of {totalPages}</strong>
        </div>
      </section>

      <section style={tablePanel}>
        <div style={tableHeader}>
          <div style={tableTitle}>Database anomalies</div>
          <div style={pager}>
            <button type="button" onClick={() => setPageOffset((value) => Math.max(0, value - PAGE_SIZE))} disabled={!canGoPrevious} style={{ ...button, ...(!canGoPrevious ? disabledButton : {}) }}>
              Previous
            </button>
            <button type="button" onClick={() => setPageOffset((value) => value + PAGE_SIZE)} disabled={!canGoNext} style={{ ...button, ...(!canGoNext ? disabledButton : {}) }}>
              Next
            </button>
          </div>
        </div>

        {error ? <div style={errorBox}>{error}</div> : null}
        {loading ? <div style={emptyState}>Loading anomalies...</div> : null}
        {!loading && !error && rows.length === 0 ? <div style={emptyState}>No anomalies match these filters.</div> : null}

        {!loading && rows.length > 0 ? (
          <div style={scrollArea}>
            <table style={table}>
              <thead>
                <tr>
                  <Th>ID</Th>
                  <Th>fk_dak</Th>
                  <Th>Table</Th>
                  <Th>Type</Th>
                  <Th>Anomaly Description</Th>
                  <Th>User Feedback</Th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.id} style={bodyRow}>
                    <Td>{displayValue(row.id)}</Td>
                    <Td>{displayValue(row.fk_dak)}</Td>
                    <Td>{row.table_label || row.table_key}</Td>
                    <Td>
                      <span style={typeBadge}>{row.anomaly_type}</span>
                    </Td>
                    <Td>{row.anomaly_description}</Td>
                    <Td>
                      <span style={row.reviewed ? reviewedBadge : pendingBadge}>
                        {row.user_feedback}
                      </span>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>
    </main>
  );
}

function Metric({ label, value }) {
  return (
    <div style={metric}>
      <span>{label}</span>
      <strong>{Number(value || 0).toLocaleString()}</strong>
    </div>
  );
}

function Th({ children }) {
  return <th style={th}>{children}</th>;
}

function Td({ children }) {
  return <td style={td}>{children}</td>;
}

function displayValue(value) {
  if (value === null || value === undefined || value === "") {
    return "NA";
  }
  return String(value);
}

const page = {
  display: "grid",
  gap: 14,
};

const header = {
  display: "flex",
  justifyContent: "space-between",
  gap: 18,
  alignItems: "end",
};

const eyebrow = {
  textTransform: "uppercase",
  letterSpacing: "0.12em",
  color: "#0f766e",
  fontSize: 12,
  fontWeight: 800,
};

const title = {
  margin: "6px 0 0",
  fontSize: 30,
  color: "#0f172a",
};

const summaryGrid = {
  display: "grid",
  gridTemplateColumns: "repeat(3, minmax(110px, 1fr))",
  gap: 10,
};

const metric = {
  border: "1px solid #d9e2ec",
  borderRadius: 8,
  background: "#ffffff",
  padding: "10px 12px",
  display: "grid",
  gap: 4,
  color: "#475569",
  fontSize: 12,
};

const filterBar = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
  gap: 12,
  alignItems: "end",
  padding: 14,
  border: "1px solid #d9e2ec",
  borderRadius: 8,
  background: "#ffffff",
};

const filterField = {
  display: "grid",
  gap: 6,
};

const filterLabel = {
  fontSize: 12,
  fontWeight: 800,
  color: "#334155",
};

const select = {
  height: 38,
  border: "1px solid #cbd5e1",
  borderRadius: 8,
  padding: "0 10px",
  background: "#ffffff",
  color: "#0f172a",
};

const filterContext = {
  minHeight: 38,
  display: "flex",
  gap: 12,
  alignItems: "center",
  justifyContent: "space-between",
  color: "#475569",
  fontSize: 13,
};

const tablePanel = {
  border: "1px solid #d9e2ec",
  borderRadius: 8,
  background: "#ffffff",
  overflow: "hidden",
};

const tableHeader = {
  minHeight: 54,
  padding: "10px 14px",
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  borderBottom: "1px solid #e2e8f0",
};

const tableTitle = {
  fontWeight: 800,
  color: "#0f172a",
};

const pager = {
  display: "flex",
  gap: 8,
};

const button = {
  height: 34,
  border: "1px solid #0f766e",
  borderRadius: 8,
  background: "#0f766e",
  color: "#ffffff",
  padding: "0 12px",
  fontWeight: 800,
  cursor: "pointer",
};

const disabledButton = {
  opacity: 0.45,
  cursor: "not-allowed",
};

const scrollArea = {
  overflowX: "auto",
};

const table = {
  width: "100%",
  borderCollapse: "collapse",
  minWidth: 920,
};

const th = {
  textAlign: "left",
  padding: "11px 12px",
  fontSize: 12,
  textTransform: "uppercase",
  color: "#475569",
  background: "#f8fafc",
  borderBottom: "1px solid #e2e8f0",
};

const td = {
  padding: "12px",
  borderBottom: "1px solid #eef2f7",
  color: "#0f172a",
  verticalAlign: "top",
  fontSize: 13,
};

const bodyRow = {
  background: "#ffffff",
};

const typeBadge = {
  display: "inline-flex",
  border: "1px solid #bfdbfe",
  borderRadius: 8,
  padding: "3px 8px",
  color: "#1d4ed8",
  background: "#eff6ff",
  fontWeight: 800,
  whiteSpace: "nowrap",
};

const reviewedBadge = {
  display: "inline-flex",
  borderRadius: 8,
  padding: "3px 8px",
  color: "#065f46",
  background: "#d1fae5",
  fontWeight: 800,
  whiteSpace: "nowrap",
};

const pendingBadge = {
  display: "inline-flex",
  borderRadius: 8,
  padding: "3px 8px",
  color: "#92400e",
  background: "#fef3c7",
  fontWeight: 800,
  whiteSpace: "nowrap",
};

const emptyState = {
  padding: 28,
  color: "#64748b",
  textAlign: "center",
};

const errorBox = {
  margin: 14,
  padding: 12,
  border: "1px solid #fecaca",
  borderRadius: 8,
  background: "#fef2f2",
  color: "#991b1b",
};
