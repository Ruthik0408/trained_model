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
export default function PredictionRow({ item, onAction, selectedTables = [], }) {
    const payload = item?.row_payload_json ?? {};
    const reasons = item?.reasons_json ?? {};
    const businessSections = useMemo(() => getBusinessSections(payload, selectedTables), [payload, selectedTables]);
    const businessDetails = useMemo(() => businessSections.flatMap((section) => section.rows), [businessSections]);
    const metrics = useMemo(() => getReviewMetrics(item, payload, reasons), [item, payload, reasons]);
    const baseReasonList = useMemo(() => getBaseReasonList(item, reasons), [item, reasons]);
    const mlSignalItems = useMemo(() => getMlSignalItems(reasons, payload, baseReasonList), [reasons, payload, baseReasonList]);
    const displayReasonItems = useMemo(() => getDisplayReasonItems(baseReasonList, mlSignalItems), [baseReasonList, mlSignalItems]);
    const [saving, setSaving] = useState(null);
    const [showAllColumns, setShowAllColumns] = useState(false);
    const [columnSearch, setColumnSearch] = useState("");
    const [expandedReasonKeys, setExpandedReasonKeys] = useState([]);
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
    useEffect(() => {
        setExpandedReasonKeys([]);
    }, [item?.prediction_id]);
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
              {getReviewTitle(item, payload) ||
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

          {displayReasonItems.length > 0 ? (<div style={reasonBox}>
              <div style={reasonTitle}>Why anomaly</div>
              {displayReasonItems.map((reasonItem) => renderReasonItem(reasonItem, expandedReasonKeys, setExpandedReasonKeys))}
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
            Use <strong>Accept</strong> when this is a valid anomaly, <strong>Reject</strong>
            when it is a false positive, and <strong>Maybe</strong> when it needs follow-up.
          </div>
          {activeFeedback ? (<div style={feedbackState}>
              Current selection: <strong>{activeFeedback}</strong>
            </div>) : null}
          <div style={actionRow}>
            <button type="button" onClick={() => submit("Accept")} disabled={saving !== null} style={{
            ...actionButton,
            background: "#0f766e",
            ...(activeFeedback === "Accept" ? activeActionButton : {}),
        }}>
              {saving === "Accept" ? "Saving..." : "Accept"}
            </button>
            <button type="button" onClick={() => submit("Reject")} disabled={saving !== null} style={{
            ...actionButton,
            background: "#b91c1c",
            ...(activeFeedback === "Reject" ? activeActionButton : {}),
        }}>
              {saving === "Reject" ? "Saving..." : "Reject"}
            </button>
            <button type="button" onClick={() => submit("Maybe")} disabled={saving !== null} style={{
            ...actionButton,
            background: "#b45309",
            ...(activeFeedback === "Maybe" ? activeActionButton : {}),
        }}>
              {saving === "Maybe" ? "Saving..." : "Maybe"}
            </button>
          </div>
        </section>
      </div>

      <section style={detailsPanel}>
          <div style={sectionTitle}>Anomaly Details</div>
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
        reasons.user_rule_flag ??
        reasons.default_rule_flag ??
        0);
    const explicitRuleCount = firstDefined(reasons.rule_count, payload.rule_count);
    const ruleCount = Number(explicitRuleCount ?? (ruleAnomaly ? 1 : 0));
    const ifScore = firstDefined(reasons.if_score, payload.if_score, reasons.isolation_score, item.raw_ml_score, item.ml_score);
    const mlThreshold = firstDefined(reasons.ml_threshold, payload.ml_threshold, item.ml_threshold);
    const ensembleScore = firstDefined(reasons.ensemble_score, payload.ensemble_score, item.ml_score, item.final_score);
    return { ruleAnomaly, ruleCount, ifScore, mlThreshold, ensembleScore };
}
function getBaseReasonList(item, reasons) {
    const rawReasons = Array.isArray(reasons.reason_list)
        ? reasons.reason_list
        : item?.rule_codes
            ? [item.rule_codes]
            : [];
    const splitReasons = rawReasons
        .flatMap(splitReasonText)
        .map((reason) => reason.trim())
        .filter(Boolean);
    return dedupeReasons(splitReasons);
}

function splitReasonText(reason) {
    const text = String(reason || "").trim();
    if (!text) {
        return [];
    }
    if (/[.!?]/.test(text)) {
        return [text];
    }
    return text.split(",");
}
function getMlSignalItems(reasons, payload, baseReasonList) {
    const signals = Array.isArray(reasons?.ml_feature_signals) ? reasons.ml_feature_signals : [];
    const groupedSignals = collapseMlSignals(signals);
    void baseReasonList;
    return groupedSignals
        .map((signal) => ({
        key: String(signal?.feature || Math.random()),
        text: formatMlSignal(signal, payload),
        details: formatMlSignalComparison(signal, payload),
    }))
        .filter((item) => item.text)
        .slice(0, 3);
}
function getDisplayReasonItems(baseReasonList, mlSignalItems) {
    const signalByText = new Map((mlSignalItems || []).map((item) => [normalizeReasonText(item.text), item]));
    const items = [];
    for (const reason of baseReasonList || []) {
        const text = formatReason(reason);
        const normalized = normalizeReasonText(text);
        const signalMatch = signalByText.get(normalized);
        if (!signalMatch && isCoveredByMlSignal(text, mlSignalItems)) {
            continue;
        }
        items.push({
            key: signalMatch?.key || normalized || text,
            text,
            details: Array.isArray(signalMatch?.details) ? signalMatch.details : [],
        });
        signalByText.delete(normalized);
    }
    for (const signalItem of signalByText.values()) {
        items.push(signalItem);
    }
    return items;
}

function isCoveredByMlSignal(reasonText, mlSignalItems) {
    if (!Array.isArray(mlSignalItems) || mlSignalItems.length === 0) {
        return false;
    }
    const reasonTokens = contentTokens(reasonText);
    if (reasonTokens.length < 5) {
        return false;
    }
    return mlSignalItems.some((item) => {
        const signalTokens = contentTokens(item?.text);
        if (signalTokens.length < 5) {
            return false;
        }
        const signalSet = new Set(signalTokens);
        const overlap = reasonTokens.filter((token) => signalSet.has(token)).length;
        return overlap / Math.min(reasonTokens.length, signalTokens.length) >= 0.65;
    });
}

function contentTokens(value) {
    const stopWords = new Set(["the", "is", "a", "an", "and", "or", "to", "from", "of", "for", "with", "which", "this", "that", "normal", "usual"]);
    return normalizeReasonText(value)
        .split(" ")
        .filter((token) => token.length > 1 && !stopWords.has(token));
}
function renderReasonItem(signalItem, expandedReasonKeys, setExpandedReasonKeys) {
    const isExpanded = expandedReasonKeys.includes(signalItem.key);
    const hasDetails = signalItem.details.length > 0;
    return (<div key={signalItem.key} style={reasonItemBlock}>
      <div style={reasonItemRow}>
        <div style={reasonItem}>{signalItem.text}</div>
        {hasDetails ? (<button type="button" onClick={() => setExpandedReasonKeys((current) => current.includes(signalItem.key)
                ? current.filter((item) => item !== signalItem.key)
                : [...current, signalItem.key])} style={reasonExpandButton}>
            {isExpanded ? "▲" : "▼"}
          </button>) : null}
      </div>
      {isExpanded && hasDetails ? (<div style={reasonDetailBox}>
          {signalItem.details.map((detail) => (<div key={detail} style={reasonDetailItem}>{detail}</div>))}
        </div>) : null}
    </div>);
}
function formatMlSignalComparison(signal, payload) {
    const comparison = signal?.comparison;
    const parsed = parseMlSignalFeature(signal.feature);
    const currentValue = formatComparisonValue(parsed, signal?.value);
    if (!comparison || typeof comparison !== "object") {
        return currentValue ? [`Current value: ${currentValue}`] : [];
    }
    if (comparison.kind === "numeric") {
        const details = [];
        if (currentValue) {
            details.push(`${comparisonCurrentLabel(parsed)}: ${currentValue}`);
        }
        const median = formatComparisonValue(parsed, comparison.median);
        const p25 = formatComparisonValue(parsed, comparison.p25);
        const p75 = formatComparisonValue(parsed, comparison.p75);
        const min = formatComparisonValue(parsed, comparison.min);
        const max = formatComparisonValue(parsed, comparison.max);
        if (median) {
            details.push(`${comparisonTypicalLabel(parsed)}: ${median}`);
        }
        if (p25 || p75) {
            details.push(`${comparisonUsualRangeLabel(parsed)}: ${p25 || "NA"} to ${p75 || "NA"}`);
        }
        if (min || max) {
            details.push(`Observed range: ${min || "NA"} to ${max || "NA"}`);
        }
        return details;
    }
    if (comparison.kind === "category") {
        const ratio = toFiniteNumber(comparison.match_ratio);
        const matchCount = toFiniteNumber(comparison.match_count);
        const totalCount = toFiniteNumber(comparison.total_count);
        const details = [];
        if (currentValue) {
            details.push(`Current value: ${currentValue}`);
        }
        if (ratio !== null && totalCount !== null) {
            details.push(`Rows with this value: ${trimTrailingZeros(ratio * 100)}% (${Math.trunc(matchCount || 0)} of ${Math.trunc(totalCount)})`);
        }
        return details;
    }
    if (comparison.kind === "missing") {
        const ratio = toFiniteNumber(comparison.present_ratio);
        const presentCount = toFiniteNumber(comparison.present_count);
        const totalCount = toFiniteNumber(comparison.total_count);
        if (ratio !== null && totalCount !== null) {
            return [`Rows where this field is present: ${trimTrailingZeros(ratio * 100)}% (${Math.trunc(presentCount || 0)} of ${Math.trunc(totalCount)})`];
        }
    }
    return [];
}
function formatComparisonValue(parsed, value) {
    const formatted = formatMeaningfulDisplayValue(parsed, value);
    if (!formatted) {
        return formatted;
    }
    if (parsed.kind === "date_gap") {
        const numeric = toFiniteNumber(value);
        if (numeric === null) {
            return formatted;
        }
        const rounded = Math.round(Math.abs(numeric));
        return `${rounded} ${rounded === 1 ? "day" : "days"}`;
    }
    return formatted;
}
function comparisonCurrentLabel(parsed) {
    return parsed.kind === "date_gap" ? "Current gap" : "Current value";
}
function comparisonTypicalLabel(parsed) {
    return parsed.kind === "date_gap" ? "Typical gap" : "Typical value";
}
function comparisonUsualRangeLabel(parsed) {
    return parsed.kind === "date_gap" ? "Usual gap range" : "Usual range";
}
function collapseMlSignals(signals) {
    const grouped = new Map();
    for (const signal of signals) {
        if (!signal || typeof signal !== "object") {
            continue;
        }
        const groupKey = getMlSignalGroupKey(signal.feature);
        const current = grouped.get(groupKey);
        if (!current || scoreMlSignal(signal) > scoreMlSignal(current)) {
            grouped.set(groupKey, signal);
        }
    }
    return [...grouped.values()].sort((left, right) => scoreMlSignal(right) - scoreMlSignal(left));
}
function getMlSignalGroupKey(feature) {
    const parsed = parseMlSignalFeature(feature);
    return parsed.groupKey;
}
function scoreMlSignal(signal) {
    const parsed = parseMlSignalFeature(signal?.feature);
    const strength = Number(signal?.strength ?? Math.abs(Number(signal?.scaled_value ?? 0)) ?? 0);
    const direction = String(signal?.direction || "").toLowerCase();
    const value = signal?.value;
    let score = Number.isFinite(strength) ? strength : 0;
    if (parsed.kind === "missing" || parsed.kind === "date_gap" || parsed.kind === "iqr") {
        score += 100;
    }
    if (parsed.kind === "category") {
        if (isTruthyCategoryValue(parsed.category) || isTruthyRawValue(value)) {
            score += 30;
        }
        if (isBlankCategoryValue(parsed.category)) {
            score += 20;
        }
        if (isFalseCategoryValue(parsed.category)) {
            score -= 10;
        }
    }
    if (direction === "high") {
        score += 5;
    }
    return score;
}
function formatMlSignal(signal, payload) {
    if (!signal || typeof signal !== "object") {
        return "";
    }
    const parsed = parseMlSignalFeature(signal.feature);
    if (parsed.kind === "missing") {
        return `${parsed.label} is missing.`;
    }
    if (parsed.kind === "iqr") {
        return formatIqrSignalReason(parsed, signal);
    }
    if (parsed.kind === "date_gap") {
        return formatDateGapSignalReason(parsed, signal);
    }
    if (parsed.kind === "category") {
        return formatCategorySignalReason(parsed, signal, payload);
    }
    return formatNumericSignalReason(parsed, signal);
}
function parseMlSignalFeature(feature) {
    const rawFeature = String(feature || "").trim();
    if (rawFeature.endsWith("__missing")) {
        const source = rawFeature.slice(0, -10);
        return {
            kind: "missing",
            groupKey: source.toLowerCase(),
            field: source,
            label: formatFieldLabel(source),
        };
    }
    if (rawFeature.startsWith("iqr_flag::")) {
        const source = rawFeature.slice("iqr_flag::".length);
        return {
            kind: "iqr",
            groupKey: source.toLowerCase(),
            field: source,
            label: formatFieldLabel(source),
        };
    }
    if (rawFeature.includes("::")) {
        const [field, category] = rawFeature.split("::", 2);
        return {
            kind: "category",
            groupKey: String(field || "").toLowerCase(),
            field,
            category: String(category || "").trim(),
            label: formatFieldLabel(field),
        };
    }
    if (rawFeature.startsWith("gap_days_") && rawFeature.includes("_to_")) {
        const gapFeature = rawFeature.slice("gap_days_".length);
        const [left, right] = gapFeature.split("_to_", 2);
        return {
            kind: "date_gap",
            groupKey: rawFeature.toLowerCase(),
            field: rawFeature,
            left,
            right,
            leftLabel: formatFieldLabel(left),
            rightLabel: formatFieldLabel(right),
            label: `${formatFieldLabel(left)} to ${formatFieldLabel(right)}`,
        };
    }
    return {
        kind: "numeric",
        groupKey: rawFeature.toLowerCase(),
        field: rawFeature,
        label: formatFieldLabel(rawFeature),
    };
}
function formatFieldLabel(feature) {
    const text = String(feature || "")
        .split(".")
        .pop()
        ?.replace(/__(missing|flag|ratio|diff)$/i, "")
        .replace(/[_\-]+/g, " ")
        .replace(/\s+/g, " ")
        .trim() || "Field";
    return text.replace(/\b\w/g, (char) => char.toUpperCase());
}
function formatIqrSignalReason(parsed, signal) {
    const direction = normalizeSignalDirection(signal?.direction);
    const value = formatMeaningfulDisplayValue(parsed, signal?.value);
    if (value) {
        return `${parsed.label} is ${direction === "low" ? "much lower" : "much higher"} than the usual range (${value}).`;
    }
    return `${parsed.label} is ${direction === "low" ? "much lower" : "much higher"} than the usual range.`;
}
function formatDateGapSignalReason(parsed, signal) {
    const numeric = toFiniteNumber(signal?.value);
    const direction = normalizeSignalDirection(signal?.direction);
    if (Number.isFinite(numeric) && numeric < 0) {
        const days = Math.round(Math.abs(numeric));
        const dayLabel = days === 1 ? "day" : "days";
        return `The date order between ${parsed.leftLabel} and ${parsed.rightLabel} looks inconsistent (${days} ${dayLabel}).`;
    }
    if (Number.isFinite(numeric)) {
        const days = Math.round(Math.abs(numeric));
        const dayLabel = days === 1 ? "day" : "days";
        if (direction === "low") {
            return `The gap from ${parsed.leftLabel} to ${parsed.rightLabel} is ${days} ${dayLabel}, which is shorter than usual.`;
        }
        return `The gap from ${parsed.leftLabel} to ${parsed.rightLabel} is ${days} ${dayLabel}, which is longer than usual.`;
    }
    if (direction === "low") {
        return `The gap from ${parsed.leftLabel} to ${parsed.rightLabel} is unusually small for similar records.`;
    }
    return `The gap from ${parsed.leftLabel} to ${parsed.rightLabel} is unusually large for similar records.`;
}
function formatCategorySignalReason(parsed, signal, payload) {
    const rawCategory = String(parsed.category || "").trim();
    const categoryValue = resolveCategoryDisplayValue(parsed, rawCategory, payload);
    if (isBlankCategoryValue(categoryValue)) {
        return `${parsed.label} is blank or missing.`;
    }
    if (isBooleanLikeCategory(categoryValue)) {
        const boolLabel = isTruthyCategoryValue(categoryValue) ? "Yes" : "No";
        return `${parsed.label} is marked as ${boolLabel}, which is unusual for similar records.`;
    }
    if (looksLikeCodeField(parsed.field)) {
        return `${parsed.label} has an unusual code or type value for similar records.`;
    }
    if (looksLikeFreeTextField(parsed.field)) {
        return `${parsed.label} includes "${String(categoryValue).trim()}", which is unusual for similar records.`;
    }
    return `${parsed.label} is "${String(categoryValue).trim()}", which is unusual for similar records.`;
}
function formatNumericSignalReason(parsed, signal) {
    const direction = normalizeSignalDirection(signal?.direction);
    const numeric = toFiniteNumber(signal?.value);
    const displayValue = formatMeaningfulDisplayValue(parsed, signal?.value);
    if (looksLikePercentField(parsed.field) && Number.isFinite(numeric)) {
        return `${parsed.label} is ${trimTrailingZeros(numeric)}%, which is ${direction === "low" ? "lower" : "higher"} than usual.`;
    }
    if (looksLikeAmountField(parsed.field)) {
        if (displayValue) {
            return `${parsed.label} is ${direction === "low" ? "lower" : "higher"} than usual (${displayValue}).`;
        }
        return `${parsed.label} is ${direction === "low" ? "lower" : "higher"} than usual.`;
    }
    if (looksLikeIdentifierField(parsed.field)) {
        if (Number.isFinite(numeric) && numeric >= 0 && numeric <= 1) {
            const percentText = `${trimTrailingZeros(numeric * 100)}%`;
            const entityLabel = describeFrequencyEncodedField(parsed.label);
            if (direction === "low") {
                return `${parsed.label} is uncommon for similar records; this ${entityLabel} appears in about ${percentText} of rows.`;
            }
            return `${parsed.label} is more common than usual for similar records; this ${entityLabel} appears in about ${percentText} of rows.`;
        }
        if (displayValue) {
            return `${parsed.label} does not match the usual pattern for similar records (recorded value: ${displayValue}).`;
        }
        return `${parsed.label} does not match the usual pattern for similar records.`;
    }
    if (displayValue) {
        return `${parsed.label} has an unusually ${direction === "low" ? "low" : "high"} recorded value (${displayValue}).`;
    }
    return `${parsed.label} has an unusual recorded value for similar records.`;
}
function resolveCategoryDisplayValue(parsed, rawCategory, payload) {
    const actualValue = findPayloadValueForSignalField(payload, parsed.field);
    if (actualValue !== undefined && actualValue !== null && actualValue !== "") {
        return String(actualValue).trim();
    }
    return prettifyCategoryValue(rawCategory);
}
function findPayloadValueForSignalField(payload, field) {
    if (!payload || typeof payload !== "object") {
        return undefined;
    }
    if (field in payload) {
        return payload[field];
    }
    const normalizedField = normalizeKey(field);
    for (const [key, value] of Object.entries(payload)) {
        if (normalizeKey(key) === normalizedField || normalizeKey(key).endsWith(normalizedField)) {
            return value;
        }
    }
    return undefined;
}
function prettifyCategoryValue(value) {
    return String(value || "")
        .replace(/[_\-]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}
function normalizeSignalDirection(direction) {
    return String(direction || "").trim().toLowerCase() === "low" ? "low" : "high";
}
function toFiniteNumber(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
}
function formatMeaningfulDisplayValue(parsed, value) {
    const numeric = toFiniteNumber(value);
    if (numeric === null) {
        const text = String(value || "").trim();
        return text ? text : "";
    }
    if (looksLikePercentField(parsed.field)) {
        return `${trimTrailingZeros(numeric)}%`;
    }
    if (looksLikeAmountField(parsed.field)) {
        return numeric.toLocaleString("en-IN", { maximumFractionDigits: 2 });
    }
    if (looksLikeCodeField(parsed.field) && Math.abs(numeric) > 999) {
        return "";
    }
    if (Math.abs(numeric) >= 1000) {
        return numeric.toLocaleString("en-IN", { maximumFractionDigits: 2 });
    }
    return trimTrailingZeros(numeric);
}
function trimTrailingZeros(value) {
    return Number(value).toLocaleString("en-IN", {
        minimumFractionDigits: 0,
        maximumFractionDigits: 2,
    });
}
function looksLikeAmountField(field) {
    const lower = String(field || "").toLowerCase();
    return ["amount", "paid", "claimed", "disallowed", "price", "total"].some((token) => lower.includes(token));
}
function looksLikePercentField(field) {
    const lower = String(field || "").toLowerCase();
    return lower.includes("percent") || lower.includes("percentage") || lower.endsWith("_pct") || lower.endsWith(" pct");
}
function looksLikeIdentifierField(field) {
    const lower = String(field || "").toLowerCase();
    return lower.includes("bill no") ||
        lower.includes("invoice no") ||
        lower.includes("pv no") ||
        lower.includes("reference no") ||
        lower.endsWith("_no") ||
        lower.endsWith(".no") ||
        lower.includes("number") ||
        lower.includes(" id");
}
function looksLikeCodeField(field) {
    const lower = String(field || "").toLowerCase();
    return lower.includes(" type") || lower.includes("_type") || lower.includes(" code") || lower.includes("_code") || lower.endsWith("_id");
}
function looksLikeFreeTextField(field) {
    const lower = String(field || "").toLowerCase();
    return lower.includes("subject") || lower.includes("remarks") || lower.includes("detail") || lower.includes("description");
}
function isBooleanLikeCategory(value) {
    const normalized = String(value || "").trim().toLowerCase();
    return ["true", "false", "yes", "no", "0", "1"].includes(normalized);
}
function isTruthyCategoryValue(value) {
    const normalized = String(value || "").trim().toLowerCase();
    return ["true", "yes", "1"].includes(normalized);
}
function isFalseCategoryValue(value) {
    const normalized = String(value || "").trim().toLowerCase();
    return ["false", "no", "0"].includes(normalized);
}
function isBlankCategoryValue(value) {
    const normalized = String(value || "").trim().toLowerCase();
    return ["blank", "missing", "null", "none", ""].includes(normalized);
}
function isTruthyRawValue(value) {
    const numeric = toFiniteNumber(value);
    if (numeric !== null) {
        return numeric >= 0.5;
    }
    return isTruthyCategoryValue(value);
}
function dedupeReasons(reasons) {
    const seen = new Set();
    return reasons.filter((reason) => {
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
            .filter(([key]) => belongsToTable(key, tableName) ||
            (tables.length === 1 && isSingleTablePayloadKey(key)))
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
function getReviewTitle(item, payload) {
    const reviewKey = resolvePayloadValue(payload, "review_key");
    if (reviewKey && !isPlaceholderReviewKey(reviewKey)) {
        return reviewKey;
    }
    const rowIdentifier = resolvePayloadValue(payload, "dakid_no") ||
        resolvePayloadValue(payload, "id") ||
        resolvePayloadValue(payload, "fk_dak");
    if (rowIdentifier) {
        return `Row ${rowIdentifier}`;
    }
    return item?.prediction_id ? `Prediction #${item.prediction_id}` : "";
}
function isPlaceholderReviewKey(value) {
    const normalized = String(value || "").trim().toLowerCase();
    return normalized.endsWith(".null.null") || normalized === "null.null.null";
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
function isSingleTablePayloadKey(key) {
    return !key.includes(".") &&
        !key.includes("__") &&
        !key.startsWith("feature_") &&
        !key.startsWith("iqr_flag::");
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
function describeFrequencyEncodedField(label) {
    const cleaned = String(label || "")
        .replace(/^Fk\s+/i, "")
        .trim();
    return cleaned || "value";
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
        .replace(/^(?:OUTLIER|RULE)::/i, "")
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
const reasonItemBlock = {
    display: "grid",
    gap: 6,
};
const reasonItemRow = {
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: 10,
};
const reasonExpandButton = {
    border: "1px solid #fdba74",
    background: "#fff",
    color: "#9a3412",
    borderRadius: 8,
    fontSize: 11,
    fontWeight: 800,
    lineHeight: 1,
    padding: "6px 8px",
    cursor: "pointer",
    flex: "0 0 auto",
};
const reasonDetailBox = {
    borderLeft: "2px solid #fdba74",
    paddingLeft: 10,
    display: "grid",
    gap: 4,
};
const reasonDetailItem = {
    color: "#92400e",
    fontSize: 12,
    fontWeight: 600,
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
