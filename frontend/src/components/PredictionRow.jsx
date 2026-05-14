import React, { useEffect, useMemo, useState } from "react";
const DEFAULT_FIELDS = [
    ["review_key"],
    ["invoice_number"],
    ["invoice_date"],
    ["bill_no"],
    ["bill_date"],
    ["reference_no"],
    ["reference_date"],
    ["amount_claimed"],
    ["amount_passed"],
    ["amount_disallowed"],
    ["record_status"],
];
const FIELD_ALIASES = {
    review_key: ["review_key", "dakid_no"],
    invoice_number: ["invoice_number", "invoice_no",],
    invoice_date: ["invoice_date"],
    bill_no: ["bill_no"],
    bill_date: ["bill_date"],
    amount_claimed: ["amount_claimed", "amount",],
    amount_passed: ["amount_passed"],
    amount_disallowed: ["amount_disallowed"],
    record_status: ["record_status"],
};
export default function PredictionRow({ item, onAction, selectedTables = [], isActive = false, }) {
    const payload = item?.row_payload_json ?? {};
    const reasons = item?.reasons_json ?? {};
    const businessSections = useMemo(() => getBusinessSections(payload, selectedTables), [payload, selectedTables]);
    const businessDetails = useMemo(() => businessSections.flatMap((section) => section.rows), [businessSections]);
    const metrics = useMemo(() => getReviewMetrics(item, payload, reasons), [item, payload, reasons]);
    const reasonList = useMemo(() => getReasonList(item, reasons), [item, reasons]);
    const [saving, setSaving] = useState(null);
    const [showAllColumns, setShowAllColumns] = useState(false);
    const [columnSearch, setColumnSearch] = useState("");
    const activeFeedback = typeof item?.feedback === "string" ? item.feedback.toLowerCase() : "";
    const sortedColumns = useMemo(() => Object.entries(payload).sort(([left], [right]) => left.localeCompare(right)), [payload]);
    const visibleColumns = useMemo(() => sortedColumns.filter(([key]) => isVisiblePayloadKey(key, selectedTables)), [sortedColumns, selectedTables]);
    const defaultSelectedColumns = useMemo(() => {
        const initialBusinessKeys = businessDetails.map(([key]) => key);
        const otherPayloadKeys = visibleColumns
            .map(([key]) => key)
            .filter((key) => !initialBusinessKeys.includes(key));
        return [...initialBusinessKeys, ...otherPayloadKeys].slice(0, 10);
    }, [businessDetails, visibleColumns]);
    const [selectedColumns, setSelectedColumns] = useState(defaultSelectedColumns);
    useEffect(() => {
        setSelectedColumns((current) => {
            const validCurrent = current.filter((key) => visibleColumns.some(([visibleKey]) => visibleKey === key));
            if (validCurrent.length > 0) {
                return validCurrent.slice(0, 10);
            }
            return defaultSelectedColumns;
        });
    }, [visibleColumns, defaultSelectedColumns]);
    const normalizedSearch = columnSearch.trim().toLowerCase();
    const allColumns = useMemo(() => visibleColumns.filter(([key]) => key.toLowerCase().includes(normalizedSearch)), [visibleColumns, normalizedSearch]);
    const selectedColumnEntries = useMemo(() => selectedColumns
        .filter((key) => visibleColumns.some(([visibleKey]) => visibleKey === key))
        .map((key) => [key, payload[key]]), [selectedColumns, payload, visibleColumns]);
    const filteredSelectableColumns = useMemo(() => visibleColumns.filter(([key]) => !selectedColumns.includes(key) &&
        key.toLowerCase().includes(normalizedSearch)), [visibleColumns, selectedColumns, normalizedSearch]);
    const addSelectedColumn = (key) => {
        setSelectedColumns((current) => {
            if (current.includes(key) || current.length >= 10) {
                return current;
            }
            return [...current, key];
        });
    };
    const removeSelectedColumn = (key) => {
        setSelectedColumns((current) => current.filter((itemKey) => itemKey !== key));
    };
    const submit = async (feedback) => {
        if (!onAction) {
            return;
        }
        setSaving(feedback);
        try {
            await onAction(item.prediction_id, item.source_record_id, feedback);
        }
        finally {
            setSaving(null);
        }
    };
    return (<article style={card}>
      <div style={topGrid}>
        <div style={reviewSummary}>
          <div>
            <div style={eyebrow}>Review Key</div>
            <div style={title}>
              {resolvePayloadValue(payload, "review_key") ||
            `Prediction #${item.prediction_id}`}
            </div>
          </div>

          <div style={metricRow}>
            <div style={metricChip}>
              Rule anomaly: <strong>{metrics.ruleAnomaly ? "Yes" : "No"}</strong>
            </div>
            <div style={metricChip}>
              Rule count: <strong>{formatInteger(metrics.ruleCount)}</strong>
            </div>
            <div style={metricChip}>
              IF score: <strong>{formatNumber(metrics.ifScore)}</strong>
            </div>
            <div style={metricChip}>
              IF threshold: <strong>{formatNumber(metrics.mlThreshold)}</strong>
            </div>
          </div>

          {reasonList.length > 0 ? (<div style={reasonBox}>
              <div style={reasonTitle}>Why anomaly</div>
              {reasonList.map((reason) => (<div key={reason} style={reasonItem}>{formatReason(reason)}</div>))}
            </div>) : null}

          <div style={toolbarRow}>
            <button type="button" onClick={() => setShowAllColumns((value) => !value)} style={secondaryButton}>
              {showAllColumns ? "Hide All Columns" : "All Columns"}
            </button>
          </div>
        </div>

        <section style={decisionPanel}>
          <div style={sectionTitle}>Review Decision</div>
          <div style={helperText}>
            Use <strong>accept</strong> when this is a valid anomaly, <strong>reject</strong>
            when it is a false positive, and <strong>maybe</strong> when it needs follow-up.
          </div>
          {activeFeedback ? (<div style={feedbackState}>
              Current selection: <strong>{activeFeedback}</strong>
            </div>) : null}
          <div style={actionRow}>
            <button type="button" onClick={() => submit("accept")} disabled={saving !== null} style={{
            ...actionButton,
            background: "#0f766e",
            ...(activeFeedback === "accept" ? activeActionButton : {}),
        }}>
              {saving === "accept" ? "Saving..." : "Accept"}
            </button>
            <button type="button" onClick={() => submit("reject")} disabled={saving !== null} style={{
            ...actionButton,
            background: "#b91c1c",
            ...(activeFeedback === "reject" ? activeActionButton : {}),
        }}>
              {saving === "reject" ? "Saving..." : "Reject"}
            </button>
            <button type="button" onClick={() => submit("maybe")} disabled={saving !== null} style={{
            ...actionButton,
            background: "#b45309",
            ...(activeFeedback === "maybe" ? activeActionButton : {}),
        }}>
              {saving === "maybe" ? "Saving..." : "Maybe"}
            </button>
          </div>
        </section>
      </div>

      <section style={detailsPanel}>
        <div style={sectionTitle}>Joined Table Details</div>
        <div style={businessSectionsGrid}>
          {businessSections.map((section) => (<div key={section.tableName} style={businessSectionBlock}>
              <div style={tableSectionTitle}>{section.label}</div>
              {section.rows.map(([key, label, value]) => (<div key={key} style={fieldRow}>
                  <span style={fieldLabel}>{label}</span>
                  <span style={fieldValue}>{displayValue(value)}</span>
                </div>))}
            </div>))}
        </div>
      </section>

      {showAllColumns ? (<section style={allColumnsPanel}>
          <div style={allColumnsHeader}>
            <div style={selectionCount}>
              {selectedColumns.length}/10 selected
            </div>
          </div>

          <input value={columnSearch} onChange={(event) => setColumnSearch(event.target.value)} placeholder="Search column name" style={searchInput}/>

          <div style={selectedChipRow}>
            {selectedColumns.map((key) => (<button key={key} type="button" onClick={() => removeSelectedColumn(key)} style={selectedChip}>
                {key} ×
              </button>))}
          </div>

          <div style={selectedColumnsGrid}>
            {selectedColumnEntries.map(([key, value]) => (<div key={key} style={allColumnCard}>
                <div style={allColumnKey}>{key}</div>
                <div style={allColumnValue}>{displayValue(value)}</div>
              </div>))}
          </div>

          <div style={pickerGrid}>
            {filteredSelectableColumns.slice(0, 30).map(([key]) => {
                const disabled = selectedColumns.length >= 10;
                return (<button key={key} type="button" onClick={() => addSelectedColumn(key)} disabled={disabled} style={{
                        ...pickerButton,
                        ...(disabled ? disabledPickerButton : {}),
                    }}>
                  {key}
                </button>);
            })}
          </div>

          <div style={allColumnsGrid}>
            {allColumns.map(([key, value]) => (<div key={key} style={allColumnCard}>
                <div style={allColumnKey}>{key}</div>
                <div style={allColumnValue}>{displayValue(value)}</div>
              </div>))}
          </div>
        </section>) : null}
    </article>);
}
function getReviewMetrics(item, payload, reasons) {
    const reasonList = Array.isArray(reasons.reason_list) ? reasons.reason_list : [];
    const ruleAnomaly = Number(reasons.rule_anomaly ??
        payload.rule_anomaly ??
        item.rule_flag ??
        reasons.human_outlier_flag ??
        0);
    const ruleCount = Number(reasons.rule_count ?? payload.rule_count ?? reasonList.length ?? 0);
    const ifScore = firstDefined(reasons.if_score, payload.if_score, reasons.isolation_score, item.raw_ml_score, item.ml_score);
    const mlThreshold = firstDefined(reasons.ml_threshold, payload.ml_threshold, item.ml_threshold);
    const ensembleScore = firstDefined(reasons.ensemble_score, payload.ensemble_score, item.ml_score, item.final_score);
    return { ruleAnomaly, ruleCount, ifScore, mlThreshold, ensembleScore };
}
function getReasonList(item, reasons) {
    const rawReasons = Array.isArray(reasons.reason_list)
        ? reasons.reason_list
        : item?.rule_codes
            ? [item.rule_codes]
            : [];
    const splitReasons = rawReasons
        .flatMap((reason) => String(reason || "").split(","))
        .map((reason) => reason.trim())
        .filter(Boolean);
    if (reasons.llm_if_reason) {
        splitReasons.push(reasons.llm_if_reason);
    }
    const seen = new Set();
    return splitReasons.filter((reason) => {
        const normalized = normalizeReasonText(reason);
        if (!normalized || seen.has(normalized)) {
            return false;
        }
        seen.add(normalized);
        return true;
    });
}
function getBusinessSections(payload, selectedTables = []) {
    const tables = selectedTables.length > 0 ? selectedTables : inferTablesFromPayload(payload);
    const sections = [];
    for (const tableName of tables) {
        const tableEntries = Object.entries(payload)
            .filter(([key]) => belongsToTable(key, tableName))
            .filter(([key]) => !isIgnoredBusinessKey(key))
            .sort((left, right) => compareBusinessColumns(left[0], right[0], left[1], right[1]));
        const nonEmpty = tableEntries.filter(([, value]) => value !== null && value !== undefined && value !== "");
        const picked = [...nonEmpty, ...tableEntries]
            .filter(([key], index, arr) => arr.findIndex(([existing]) => existing === key) === index)
            .slice(0, 12)
            .map(([key, value]) => [key, formatBusinessRowLabel(tableName, key), value]);
        if (picked.length > 0) {
            sections.push({
                tableName,
                label: formatTableLabel(tableName),
                rows: picked,
            });
        }
    }
    if (sections.length > 0) {
        return sections;
    }
    return [
        {
            tableName: "default",
            label: "Business Details",
            rows: DEFAULT_FIELDS.map(([key, label]) => [key, label, resolvePayloadValue(payload, key)]),
        },
    ];
}
function inferTablesFromPayload(payload) {
    const names = new Set();
    for (const key of Object.keys(payload)) {
        if (key.includes(".")) {
            names.add(key.split(".", 1)[0]);
            continue;
        }
        if (key.includes("__")) {
            names.add(key.split("__", 1)[0]);
        }
    }
    return [...names];
}
function belongsToTable(key, tableName) {
    return (key.startsWith(`${tableName}.`) || key.startsWith(`${tableName}__`));
}
function isVisiblePayloadKey(key, selectedTables) {
    if (selectedTables.length === 0) {
        return true;
    }
    if (selectedTables.some((tableName) => belongsToTable(key, tableName))) {
        return true;
    }
    if (key.startsWith("feature_") || key.startsWith("iqr_flag::")) {
        return true;
    }
    return !key.includes(".") && !key.includes("__");
}
function isIgnoredBusinessKey(key) {
    const lower = key.toLowerCase();
    return (lower === "amount" ||
        lower === "review_key" ||
        lower.startsWith("feature_") ||
        lower.startsWith("iqr_flag::"));
}
function compareBusinessColumns(leftKey, rightKey, leftValue, rightValue) {
    const scoreDiff = scoreBusinessColumn(rightKey, rightValue) -
        scoreBusinessColumn(leftKey, leftValue);
    if (scoreDiff !== 0) {
        return scoreDiff;
    }
    return leftKey.localeCompare(rightKey);
}
function scoreBusinessColumn(key, value) {
    const lower = key.toLowerCase();
    let score = value !== null && value !== undefined && value !== "" ? 100 : 0;
    if (lower.includes("reference"))
        score += 60;
    if (lower.includes("invoice"))
        score += 60;
    if (lower.includes("amount"))
        score += 50;
    if (lower.includes("status"))
        score += 45;
    if (lower.includes("date"))
        score += 35;
    if (lower.includes("number") || lower.endsWith(".no") || lower.endsWith("_no"))
        score += 30;
    if (lower.includes("name"))
        score += 25;
    if (lower.includes("reason") || lower.includes("remarks"))
        score += 15;
    if (lower.includes("fk_"))
        score -= 20;
    return score;
}
function formatBusinessRowLabel(tableName, key) {
    const rawColumn = key.startsWith(`${tableName}.`)
        ? key.slice(tableName.length + 1)
        : key.startsWith(`${tableName}__`)
            ? key.slice(tableName.length + 2)
            : key;
    return rawColumn;
}
function formatTableLabel(tableName) {
    return tableName
        .replace(/_/g, " ")
        .replace(/\b\w/g, (char) => char.toUpperCase());
}
function resolvePayloadValue(payload, fieldKey) {
    const aliases = FIELD_ALIASES[fieldKey] || [fieldKey];
    for (const alias of aliases) {
        const direct = payload[alias];
        if (direct !== undefined && direct !== null && direct !== "") {
            return direct;
        }
    }
    const normalizedTargets = new Set(aliases.map((value) => normalizeKey(value)).concat(normalizeKey(fieldKey)));
    for (const [key, value] of Object.entries(payload)) {
        if (value === undefined || value === null || value === "") {
            continue;
        }
        const normalizedKey = normalizeKey(key);
        if (normalizedTargets.has(normalizedKey) ||
            [...normalizedTargets].some((target) => normalizedKey.endsWith(target))) {
            return value;
        }
    }
    return payload[fieldKey];
}
function normalizeKey(value) {
    return String(value)
        .toLowerCase()
        .replace(/[^a-z0-9]/g, "");
}
function firstDefined(...values) {
    for (const value of values) {
        if (value !== undefined && value !== null && value !== "") {
            return value;
        }
    }
    return undefined;
}
function formatNumber(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric.toFixed(2) : "NA";
}
function formatInteger(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? String(Math.trunc(numeric)) : "NA";
}
function displayValue(value) {
    if (value === null || value === undefined || value === "") {
        return "NA";
    }
    return String(value);
}
function formatReason(value) {
    return value
        .replace(/^OUTLIER::/i, "")
        .replace(/_/g, " ");
}
function normalizeReasonText(value) {
    return formatReason(String(value || ""))
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, " ")
        .trim();
}
const card = {
    background: "linear-gradient(180deg, #fffdf7 0%, #ffffff 100%)",
    border: "1px solid rgba(146, 64, 14, 0.14)",
    borderRadius: 24,
    padding: 22,
    boxShadow: "0 18px 40px rgba(120, 53, 15, 0.08)",
};
const topGrid = {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 320px), 1fr))",
    gap: 18,
    alignItems: "stretch",
    marginBottom: 16,
};
const reviewSummary = {
    minWidth: 0,
    display: "grid",
    alignContent: "start",
};
const eyebrow = {
    fontSize: 11,
    letterSpacing: "0.16em",
    textTransform: "uppercase",
    color: "#9a3412",
    fontWeight: 700,
};
const title = {
    fontSize: 22,
    fontWeight: 700,
    color: "#1c1917",
    marginTop: 4,
    wordBreak: "break-word",
};
const metricRow = {
    display: "flex",
    gap: 8,
    flexWrap: "wrap",
    marginBottom: 14,
};
const metricChip = {
    background: "#fff7ed",
    border: "1px solid #fed7aa",
    borderRadius: 999,
    padding: "8px 12px",
    fontSize: 12,
    color: "#7c2d12",
};
const reasonBox = {
    border: "1px solid #fed7aa",
    background: "#fffbeb",
    borderRadius: 14,
    padding: 12,
    display: "grid",
    gap: 6,
};
const reasonTitle = {
    color: "#7c2d12",
    fontSize: 12,
    fontWeight: 800,
    textTransform: "uppercase",
};
const reasonItem = {
    color: "#1c1917",
    fontSize: 13,
    fontWeight: 700,
};
const reasonMuted = {
    color: "#92400e",
    fontSize: 12,
    fontWeight: 700,
};
const reasonError = {
    color: "#b91c1c",
    fontSize: 12,
    fontWeight: 700,
};
const toolbarRow = {
    display: "flex",
    gap: 10,
    flexWrap: "wrap",
    marginTop: 14,
    marginBottom: 16,
};
const secondaryButton = {
    color: "#7c2d12",
    background: "#fff7ed",
    border: "1px solid #fdba74",
    borderRadius: 12,
    padding: "9px 14px",
    fontWeight: 700,
    cursor: "pointer",
};
const detailsPanel = {
    background: "#ffffff",
    border: "1px solid #fed7aa",
    borderRadius: 18,
    padding: 18,
};
const businessSectionsGrid = {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 320px), 1fr))",
    gap: 16,
    marginTop: 12,
    alignItems: "start",
};
const decisionPanel = {
    background: "#ffffff",
    border: "1px solid #fed7aa",
    borderRadius: 18,
    padding: 18,
    minHeight: 160,
};
const sectionTitle = {
    fontWeight: 800,
    fontSize: 16,
    color: "#1c1917",
};
const fieldRow = {
    display: "flex",
    justifyContent: "space-between",
    gap: 14,
    padding: "10px 0",
    borderBottom: "1px solid #ffedd5",
};
const fieldLabel = {
    color: "#9a3412",
    fontWeight: 600,
    fontSize: 13,
};
const fieldValue = {
    color: "#292524",
    textAlign: "right",
    wordBreak: "break-word",
    fontSize: 12,
    lineHeight: 1.5,
    maxWidth: "52%",
};
const businessSectionBlock = {
    display: "grid",
    gap: 2,
    minWidth: 0,
};
const tableSectionTitle = {
    color: "#7c2d12",
    fontWeight: 800,
    fontSize: 12,
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    marginBottom: 2,
};
const helperText = {
    color: "#57534e",
    fontSize: 14,
    lineHeight: 1.6,
    marginTop: 10,
};
const feedbackState = {
    marginTop: 12,
    color: "#7c2d12",
    fontSize: 13,
    fontWeight: 700,
};
const actionRow = {
    display: "flex",
    gap: 10,
    flexWrap: "wrap",
    marginTop: 14,
};
const actionButton = {
    color: "#fff",
    border: "none",
    borderRadius: 12,
    padding: "10px 16px",
    fontWeight: 700,
    cursor: "pointer",
};
const activeActionButton = {
    boxShadow: "0 0 0 3px rgba(15, 23, 42, 0.12)",
    transform: "translateY(-1px)",
};
const allColumnsPanel = {
    marginTop: 18,
    background: "#fff",
    border: "1px solid #fed7aa",
    borderRadius: 18,
    padding: 18,
};
const allColumnsHeader = {
    display: "flex",
    justifyContent: "space-between",
    gap: 12,
    alignItems: "center",
    flexWrap: "wrap",
};
const allColumnsGrid = {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
    gap: 10,
    marginTop: 12,
};
const selectedColumnsGrid = {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
    gap: 10,
    marginTop: 12,
};
const allColumnCard = {
    border: "1px solid #ffedd5",
    borderRadius: 12,
    padding: 10,
    background: "#fffbf5",
};
const allColumnKey = {
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: "0.08em",
    color: "#9a3412",
    fontWeight: 700,
    marginBottom: 6,
    wordBreak: "break-word",
};
const allColumnValue = {
    color: "#292524",
    lineHeight: 1.45,
    fontSize: 13,
    wordBreak: "break-word",
};
const searchInput = {
    width: "100%",
    borderRadius: 12,
    border: "1px solid #fdba74",
    padding: "9px 12px",
    font: "inherit",
    marginTop: 12,
};
const selectionCount = {
    borderRadius: 999,
    border: "1px solid #fed7aa",
    padding: "6px 10px",
    color: "#9a3412",
    fontSize: 12,
    fontWeight: 700,
    background: "#fff7ed",
};
const selectedChipRow = {
    display: "flex",
    gap: 8,
    flexWrap: "wrap",
    marginTop: 12,
};
const selectedChip = {
    border: "1px solid #fdba74",
    borderRadius: 999,
    background: "#fff7ed",
    color: "#9a3412",
    padding: "6px 10px",
    fontSize: 12,
    cursor: "pointer",
};
const pickerGrid = {
    display: "flex",
    gap: 8,
    flexWrap: "wrap",
    marginTop: 10,
};
const pickerButton = {
    border: "1px solid #fed7aa",
    borderRadius: 999,
    background: "#fff",
    color: "#7c2d12",
    padding: "7px 11px",
    fontSize: 12,
    cursor: "pointer",
};
const disabledPickerButton = {
    opacity: 0.5,
    cursor: "not-allowed",
};
