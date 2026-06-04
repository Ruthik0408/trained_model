import React, { useEffect, useMemo, useState } from "react";
import Card from "../components/Card";
import { clearApiCache, getWorkbenchTables, previewWorkbench, runWorkbench, } from "../api/anomalyApi";
const INITIAL_TABLE_LOAD_RETRIES = 6;
const INITIAL_TABLE_LOAD_DELAY_MS = 1000;
const DEFAULT_USER_RULE = (index) => ({
    name: `User rule ${index + 1}`,
    first_column: "",
    second_column: "",
    operator: ">",
    value: "",
    second_value: "",
});
export default function WorkbenchPage({ onRunComplete, includeRuleAnomalyStep = true }) {
    const [tables, setTables] = useState([]);
    const [tableLoadError, setTableLoadError] = useState("");
    const [selectedTables, setSelectedTables] = useState([]);
    const [userRules, setUserRules] = useState([]);
    const [featureRules, setFeatureRules] = useState([]);
    const [fromDate, setFromDate] = useState("");
    const [toDate, setToDate] = useState("");
    const [loading, setLoading] = useState(true);
    const [previewing, setPreviewing] = useState(false);
    const [running, setRunning] = useState(false);
    const [autoFeatureLoading, setAutoFeatureLoading] = useState(false);
    const [autoFeatureError, setAutoFeatureError] = useState("");
    const [result, setResult] = useState(null);
    const [previewResult, setPreviewResult] = useState(null);
    const [runError, setRunError] = useState("");
    const [runWarning, setRunWarning] = useState("");
    const [tableSearch, setTableSearch] = useState("");
    const [showSuggestions, setShowSuggestions] = useState(false);
    const [showAllTables, setShowAllTables] = useState(false);
    const [activeStep, setActiveStep] = useState(1);
    async function loadTables(attempt = 0) {
        setLoading(true);
        setTableLoadError("");
        try {
            const response = await getWorkbenchTables();
            setTables(response.data || []);
        }
        catch (error) {
            if (attempt < INITIAL_TABLE_LOAD_RETRIES) {
                setTimeout(() => {
                    loadTables(attempt + 1);
                }, INITIAL_TABLE_LOAD_DELAY_MS);
                return;
            }
            setTables([]);
            setTableLoadError("Unable to load tables from the backend. Make sure the backend is running and refresh or retry.");
        }
        finally {
            setLoading(false);
        }
    }
    useEffect(() => {
        loadTables();
    }, []);
    const columnOptions = useMemo(() => {
        return selectedTables.flatMap((tableName) => {
            const table = tables.find((item) => item.table_name === tableName);
            return (table?.columns || []).map((column) => ({
                label: `${tableName}.${column.column_name}`,
                value: `${tableName}.${column.column_name}`,
            }));
        });
    }, [selectedTables, tables]);
    const availableColumns = useMemo(() => new Set(columnOptions.map((item) => item.value)), [columnOptions]);
    useEffect(() => {
        setUserRules((current) => current.filter((rule) => !!rule.first_column &&
            availableColumns.has(rule.first_column) &&
            (!rule.second_column || availableColumns.has(rule.second_column))));
        setFeatureRules((current) => current.filter((rule) => !!rule.first_column &&
            availableColumns.has(rule.first_column) &&
            (!rule.second_column || availableColumns.has(rule.second_column))));
    }, [availableColumns]);
    useEffect(() => {
        setRunError("");
        setRunWarning("");
        setPreviewResult(null);
        setResult(null);
    }, [selectedTables, fromDate, toDate]);
    const filteredTables = useMemo(() => {
        const query = tableSearch.trim().toLowerCase();
        if (!query)
            return [];
        return tables
            .filter((table) => table.table_name.toLowerCase().includes(query))
            .slice(0, 25);
    }, [tableSearch, tables]);
    const visibleTables = useMemo(() => {
        if (tableSearch.trim())
            return filteredTables;
        if (showAllTables)
            return tables.slice(0, 200);
        return [];
    }, [filteredTables, showAllTables, tableSearch, tables]);
    const hasExplicitDateRange = Boolean(fromDate && toDate);
    const isDateStepComplete = hasExplicitDateRange && isDateSelectionValid(fromDate, toDate);
    const canEditRules = selectedTables.length > 0 && isDateStepComplete;
    const canRun = canEditRules && !previewing && !running && !autoFeatureLoading;
    const dateValidationError = getDateValidationError(fromDate, toDate);
    useEffect(() => {
        setFeatureRules([]);
        setAutoFeatureError("");
        setAutoFeatureLoading(false);
    }, [selectedTables, fromDate, toDate]);
    const finalStepId = includeRuleAnomalyStep ? 4 : 3;
    const steps = useMemo(() => {
        const baseSteps = [
            { id: 1, label: "Select Tables", complete: selectedTables.length > 0, disabled: false },
            { id: 2, label: "Date Range", complete: isDateStepComplete, disabled: selectedTables.length === 0 },
        ];
        if (includeRuleAnomalyStep) {
            baseSteps.push({ id: 3, label: "Rule Anomalies", complete: true, disabled: !canEditRules });
        }
        baseSteps.push({ id: finalStepId, label: "Preview And Run", complete: Boolean(result), disabled: !canEditRules });
        return baseSteps;
    }, [canEditRules, finalStepId, includeRuleAnomalyStep, isDateStepComplete, result, selectedTables.length]);
    useEffect(() => {
        const activeConfig = steps.find((step) => step.id === activeStep);
        if (activeConfig && !activeConfig.disabled) {
            return;
        }
        const fallbackStep = [...steps]
            .filter((step) => step.id < activeStep && !step.disabled)
            .pop();
        if (fallbackStep) {
            setActiveStep(fallbackStep.id);
            return;
        }
        const firstAvailableStep = steps.find((step) => !step.disabled);
        if (firstAvailableStep) {
            setActiveStep(firstAvailableStep.id);
        }
    }, [activeStep, steps]);
    function goToStep(stepId) {
        const target = steps.find((step) => step.id === stepId);
        if (!target || target.disabled) {
            return;
        }
        setActiveStep(stepId);
    }
    function goNextStep() {
        const nextStep = steps.find((step) => step.id > activeStep && !step.disabled);
        if (nextStep) {
            setActiveStep(nextStep.id);
        }
    }
    function goPreviousStep() {
        const previousSteps = steps.filter((step) => step.id < activeStep && !step.disabled);
        const previousStep = previousSteps[previousSteps.length - 1];
        if (previousStep) {
            setActiveStep(previousStep.id);
        }
    }
    function toggleTable(tableName) {
        setSelectedTables((current) => {
            if (current.includes(tableName)) {
                return current.filter((item) => item !== tableName);
            }
            if (current.length >= 3)
                return current;
            setTableSearch("");
            setShowSuggestions(false);
            setShowAllTables(false);
            return [...current, tableName];
        });
    }
    function addUserRule() {
        setUserRules((current) => [...current, DEFAULT_USER_RULE(current.length)]);
    }
    function buildWorkbenchPayload() {
        if (selectedTables.length === 0) {
            throw new Error("Please select at least one table.");
        }
        if (dateValidationError) {
            throw new Error(dateValidationError);
        }
        if (!hasExplicitDateRange) {
            throw new Error("Please select both From date and To date before running the workbench.");
        }
        const activeUserRules = includeRuleAnomalyStep ? userRules
            .filter((rule) => rule.first_column && availableColumns.has(rule.first_column))
            .map((rule) => ({
            ...rule,
            second_column: rule.second_column || null,
            value: rule.second_column ? null : rule.value || null,
            second_value: rule.operator === "between" ? rule.second_value || null : null,
        })) : [];
        const activeFeatureRules = featureRules
            .filter((rule) => rule.first_column && availableColumns.has(rule.first_column))
            .map((rule) => ({
            name: rule.name,
            feature_type: rule.feature_type,
            first_column: rule.first_column,
            second_column: featureUsesSecondColumn(rule.feature_type) ? rule.second_column || null : null,
            operator: rule.operator || null,
        }));
        const normalizedFromDate = normalizeDateInput(fromDate);
        const normalizedToDate = normalizeDateInput(toDate);
        return {
            run_name: "Tulip anomaly workbench",
            selected_tables: selectedTables,
            joins: [],
            user_rules: activeUserRules,
            feature_rules: activeFeatureRules,
            from_date: normalizedFromDate,
            to_date: normalizedToDate,
        };
    }
    async function executePreview() {
        setPreviewing(true);
        setRunError("");
        setRunWarning("");
        try {
            const response = await previewWorkbench(buildWorkbenchPayload());
            setPreviewResult(response.data || null);
            setResult(null);
        }
        catch (error) {
            const formatted = formatWorkbenchError(extractWorkbenchError(error));
            if (formatted.isFatal) {
                setPreviewResult(null);
                setRunError(formatted.message);
            }
            else {
                setRunWarning(formatted.message);
            }
        }
        finally {
            setPreviewing(false);
        }
    }
    async function executeRun() {
        setRunning(true);
        setRunError("");
        setRunWarning("");
        try {
            const response = await runWorkbench(buildWorkbenchPayload());
            const data = response.data;
            clearApiCache();
            setResult(data);
            onRunComplete?.({
                runId: data.run_id,
                runName: data.run_name,
                datasetTable: data.metrics?.dataset_table,
                selectedTables: data.metrics?.selected_tables,
                totalRows: data.total_rows,
                userRuleCount: data.user_rule_count,
                mlAnomalyCount: data.ml_anomaly_count,
                finalAnomalyCount: data.final_anomaly_count,
                amountTotal: data.amount_total,
            });
        }
        catch (error) {
            const formatted = formatWorkbenchError(extractWorkbenchError(error));
            if (formatted.isFatal) {
                setResult(null);
                setRunError(formatted.message);
            }
            else {
                setRunWarning(formatted.message);
            }
        }
        finally {
            setRunning(false);
        }
    }
    return (<div style={page}>
      <div style={stepper}>
        {steps.map((step) => {
            const isActive = step.id === activeStep;
            return (<button key={step.id} type="button" onClick={() => goToStep(step.id)} disabled={step.disabled} style={{
                    ...stepButton,
                    ...(isActive ? activeStepButton : {}),
                    ...(step.complete ? completeStepButton : {}),
                    ...(step.disabled ? disabledStepButton : {}),
                }}>
              <span style={stepCircle}>{step.id}</span>
              <span style={stepLabel}>{step.label}</span>
            </button>);
        })}
      </div>

      {activeStep === 1 ? (<div style={wizardPanel}>
        <Card title="1. Select Tables">
          <div style={searchWrap}>
            <input value={tableSearch} onChange={(event) => {
            setTableSearch(event.target.value);
            setShowSuggestions(true);
            setShowAllTables(false);
        }} onFocus={() => setShowSuggestions(true)} placeholder="Search tables" style={{ ...input, marginBottom: 0 }}/>
            <button type="button" onClick={() => {
            setShowAllTables((current) => !current);
            setShowSuggestions(true);
        }} style={browseButton} title="Show trained tables">
              ▼
            </button>
          </div>

          {loading ? <div>Loading tables...</div> : null}
          {tableLoadError ? (<div style={errorBox}>
              {tableLoadError}
              <button type="button" onClick={loadTables} style={{ ...secondaryButton, marginTop: 10, padding: "8px 12px" }}>
                Retry
              </button>
            </div>) : null}
          {showSuggestions && (tableSearch.trim() || showAllTables) ? (<div style={suggestionPanel}>
              {visibleTables.map((table) => {
                const active = selectedTables.includes(table.table_name);
                return (<button key={table.table_name} type="button" onClick={() => toggleTable(table.table_name)} style={{ ...suggestionItem, ...(active ? activeSuggestionItem : {}) }}>
                    {table.table_name}
                  </button>);
            })}
              {!visibleTables.length ? <div style={hint}>No matching trained tables found.</div> : null}
            </div>) : null}

          {selectedTables.length > 0 ? (<div style={chipRow}>
              {selectedTables.map((table) => (<button key={table} type="button" onClick={() => toggleTable(table)} style={selectedChip}>
                  {table} ×
                </button>))}
            </div>) : null}
        </Card>
      </div>) : null}

      {activeStep === 2 ? (<div style={wizardPanel}>
        <Card title="2. Date Range">
          {runWarning ? <div style={warningBox}>{runWarning}</div> : null}
          {runError ? <div style={errorBox}>{runError}</div> : null}
          {autoFeatureError ? <div style={errorBox}>{autoFeatureError}</div> : null}
          {dateValidationError ? <div style={errorBox}>{dateValidationError}</div> : null}

          <div style={dateGrid}>
            <div style={field}>
              <label htmlFor="from-date" style={label}>From date</label>
              <input id="from-date" type="date" value={fromDate} onChange={(event) => setFromDate(event.target.value)} disabled={!selectedTables.length} style={{
            ...input,
            ...(fromDate && !normalizeDateInput(fromDate) ? invalidInput : {}),
        }}/>
            </div>
            <div style={field}>
              <label htmlFor="to-date" style={label}>To date</label>
              <input id="to-date" type="date" value={toDate} onChange={(event) => setToDate(event.target.value)} disabled={!selectedTables.length} style={{
            ...input,
            ...(toDate && !normalizeDateInput(toDate) ? invalidInput : {}),
        }}/>
            </div>
          </div>
        </Card>
      </div>) : null}

      {includeRuleAnomalyStep && activeStep === 3 ? (<div style={wizardPanel}>
        <Card title="3. Rule Anomaly">
          {autoFeatureError ? <div style={errorBox}>{autoFeatureError}</div> : null}
          <div style={actionBar}>
            <button type="button" onClick={addUserRule} disabled={!canEditRules} style={secondaryButton}>Add User Rule</button>
          </div>

          <div style={stack}>
            {userRules.map((rule, index) => {
            return (<div key={index} style={ruleRow}>
                  <SearchableOptionSelect value={rule.first_column} onChange={(value) => updateList(setUserRules, index, { first_column: value })} options={columnOptions} placeholder="Column 1" compact disabled={!canEditRules}/>

                  <select value={rule.operator} disabled={!canEditRules} onChange={(event) => updateList(setUserRules, index, { operator: event.target.value })} style={compactOperatorInput}>
                    <option value=">">{">"}</option>
                    <option value=">=">{">="}</option>
                    <option value="<">{"<"}</option>
                    <option value="<=">{"<="}</option>
                    <option value="=">{"="}</option>
                    <option value="!=">{"!="}</option>
                    <option value="null">null</option>
                    <option value="not null">not null</option>
                  </select>

                  <SearchableOptionSelect value={rule.second_column} onChange={(value) => updateList(setUserRules, index, { second_column: value })} options={columnOptions} placeholder="Column 2" compact disabled={!canEditRules}/>

                  <button type="button" onClick={() => removeListItem(setUserRules, index)} disabled={!canEditRules} style={ruleRemoveButton} title="Remove" aria-label="Remove user rule">
                    x
                  </button>
                </div>);
        })}
          </div>
        </Card>
      </div>) : null}

      {activeStep === finalStepId ? (<div style={wizardPanel}>
        <Card title={`${finalStepId}. Preview And Run`}>
          {runWarning ? <div style={warningBox}>{runWarning}</div> : null}
          {runError ? <div style={errorBox}>{runError}</div> : null}
          {autoFeatureError ? <div style={errorBox}>{autoFeatureError}</div> : null}

          <div style={finalActionBox}>
            <div style={finalActionTitle}>Finish Setup</div>
            <div style={finalActionText}>Preview the trained model setup first, then score the selected Postgres rows with the saved trained model.</div>
            <div style={statusRow}>
              <div style={statusChip}>Saved model inference: ready</div>
            </div>
            <div style={actionBar}>
              <button type="button" onClick={executePreview} disabled={!canRun} style={previewResult && !previewing ? successButton : primaryButton}>
                {previewing ? "Previewing..." : "Preview Model Setup"}
              </button>
              <button type="button" onClick={executeRun} disabled={!canRun} style={result && !running ? successButton : secondaryButton}>
                {running ? "Running..." : result ? "Run Completed" : "Find Anomalies With Saved Model"}
              </button>
            </div>
          </div>
        </Card>
      </div>) : null}

      <div style={wizardNav}>
        {activeStep !== 1 ? (<button type="button" onClick={goPreviousStep} disabled={!steps.some((step) => step.id < activeStep && !step.disabled)} style={secondaryButton}>
            Previous
          </button>) : <div />}
        {activeStep !== finalStepId ? (<button type="button" onClick={goNextStep} disabled={!steps.some((step) => step.id > activeStep && !step.disabled)} style={primaryButton}>
            Next
          </button>) : <div />}
      </div>

    </div>);
}
function SearchableOptionSelect({ value, onChange, options, placeholder, compact = false, disabled = false, }) {
    const [searchText, setSearchText] = useState(value);
    const [showOptions, setShowOptions] = useState(false);
    const [showAllOptions, setShowAllOptions] = useState(false);
    useEffect(() => {
        setSearchText(value);
    }, [value]);
    const visibleOptions = useMemo(() => {
        const query = searchText.trim().toLowerCase();
        if (showAllOptions)
            return options.slice(0, 200);
        if (!query)
            return [];
        return options.filter((option) => option.label.toLowerCase().includes(query)).slice(0, 50);
    }, [options, searchText, showAllOptions]);
    return (<div style={compact ? compactSelectWrap : selectWrap}>
      <div style={compact ? compactSearchWrap : searchWrap}>
        <input value={searchText} disabled={disabled} onChange={(event) => {
            const nextValue = event.target.value;
            setSearchText(nextValue);
            setShowOptions(true);
            setShowAllOptions(false);
            const matchedOption = options.find((option) => option.value === nextValue);
            onChange(matchedOption ? matchedOption.value : "");
        }} placeholder={placeholder} style={disabled ? { ...(compact ? compactInput : input), ...disabledInput } : compact ? compactInput : input}/>
        <button type="button" disabled={disabled} onClick={() => {
            setShowAllOptions((current) => !current);
            setShowOptions(true);
        }} style={disabled ? { ...(compact ? compactBrowseButton : browseButton), ...disabledButton } : compact ? compactBrowseButton : browseButton} title={placeholder}>
          ▼
        </button>
      </div>
      {showOptions && !disabled && (searchText.trim() || showAllOptions) ? (<div style={suggestionPanel}>
          {visibleOptions.map((option) => {
                const active = option.value === value;
                return (<button key={option.value} type="button" onClick={() => {
                        setSearchText(option.value);
                        onChange(option.value);
                        setShowOptions(false);
                        setShowAllOptions(false);
                    }} style={{ ...suggestionItem, ...(active ? activeSuggestionItem : {}) }}>
                {option.label}
              </button>);
            })}
          {!visibleOptions.length ? <div style={hint}>No matching options found.</div> : null}
        </div>) : null}
    </div>);
}
function extractWorkbenchError(error) {
    const detail = error?.response?.data?.detail;
    if (Array.isArray(detail) && detail.length > 0) {
        return detail
            .map((item) => {
            const location = Array.isArray(item.loc) ? item.loc.join(".") : "request";
            return `${location}: ${item.msg}`;
        })
            .join("; ");
    }
    if (typeof detail === "string" && detail.trim())
        return detail;
    if (detail && typeof detail === "object")
        return JSON.stringify(detail);
    if (typeof error?.response?.data === "string" && error.response.data.trim())
        return error.response.data;
    if (typeof error?.message === "string" && error.message.trim())
        return error.message;
    return "Workbench run failed.";
}
function formatWorkbenchError(detail) {
    const message = typeof detail === "string" ? detail : JSON.stringify(detail ?? "Workbench run failed.");
    const lowered = message.toLowerCase();
    if (lowered.includes("year") && lowered.includes("out of range")) {
        return {
            message: "Some date values in the selected data are invalid. The workbench will skip invalid dates, but please review date columns used in feature rules.",
            isFatal: false,
        };
    }
    if (message.includes("The selected join returned no rows.") ||
        message.includes("The selected SQL join returned no rows.")) {
        return {
            message: "The selected join did not produce rows for this run. Please review the chosen join keys and join type.",
            isFatal: true,
        };
    }
    if (message.includes("No usable feature columns were produced")) {
        return {
            message: "No ML features were selected for the chosen dates. Check the selected date range and source data.",
            isFatal: true,
        };
    }
    return { message, isFatal: true };
}
function updateList(setter, index, patch) {
    setter((current) => current.map((item, itemIndex) => (itemIndex === index ? { ...item, ...patch } : item)));
}
function removeListItem(setter, index) {
    setter((current) => current.filter((_, itemIndex) => itemIndex !== index));
}
function featureUsesSecondColumn(featureType) {
    return featureType === "daysbetween";
}
function isDateSelectionValid(fromDate, toDate) {
    const normalizedFromDate = normalizeDateInput(fromDate);
    const normalizedToDate = normalizeDateInput(toDate);
    if (!fromDate && !toDate) {
        return true;
    }
    if ((fromDate && !normalizedFromDate) || (toDate && !normalizedToDate)) {
        return false;
    }
    if (normalizedFromDate && normalizedToDate) {
        return normalizedFromDate <= normalizedToDate;
    }
    return true;
}
function normalizeDateInput(value) {
    const rawValue = String(value || "").trim();
    if (!rawValue) {
        return null;
    }
    const isoMatch = rawValue.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (isoMatch) {
        const [, year, month, day] = isoMatch;
        return isRealDateParts(Number(year), Number(month), Number(day))
            ? `${year}-${month}-${day}`
            : null;
    }
    const slashMatch = rawValue.match(/^(\d{2})\/(\d{2})\/(\d{4})$/);
    if (slashMatch) {
        const [, day, month, year] = slashMatch;
        return isRealDateParts(Number(year), Number(month), Number(day))
            ? `${year}-${month}-${day}`
            : null;
    }
    return null;
}
function isRealDateParts(year, month, day) {
    if (!Number.isInteger(year) || !Number.isInteger(month) || !Number.isInteger(day)) {
        return false;
    }
    if (year < 2000 || year > 2100) {
        return false;
    }
    const candidate = new Date(Date.UTC(year, month - 1, day));
    return candidate.getUTCFullYear() === year &&
        candidate.getUTCMonth() === month - 1 &&
        candidate.getUTCDate() === day;
}
function getDateValidationError(fromDate, toDate) {
    const normalizedFromDate = normalizeDateInput(fromDate);
    const normalizedToDate = normalizeDateInput(toDate);
    if (fromDate && !normalizedFromDate) {
        return "Enter a valid From date.";
    }
    if (toDate && !normalizedToDate) {
        return "Enter a valid To date.";
    }
    if (normalizedFromDate && normalizedToDate && normalizedFromDate > normalizedToDate) {
        return "From date cannot be after To date.";
    }
    return "";
}
const page = { display: "grid", gap: 18 };
const stepper = {
    display: "grid",
    gridTemplateColumns: "repeat(5, minmax(110px, 1fr))",
    gap: 10,
    alignItems: "stretch",
    overflowX: "auto",
    padding: "4px 2px 10px",
};
const stepButton = {
    border: "1px solid #cbd5e1",
    borderRadius: 14,
    background: "#ffffff",
    color: "#334155",
    padding: "12px 10px",
    display: "grid",
    justifyItems: "center",
    gap: 8,
    minHeight: 86,
    cursor: "pointer",
    font: "inherit",
};
const activeStepButton = {
    borderColor: "#0f766e",
    background: "#ecfdf5",
    color: "#0f766e",
    boxShadow: "0 10px 24px rgba(15, 118, 110, 0.12)",
};
const completeStepButton = {
    borderColor: "#86efac",
};
const disabledStepButton = {
    opacity: 0.45,
    cursor: "not-allowed",
    background: "#f8fafc",
};
const invalidInput = {
    borderColor: "#dc2626",
    boxShadow: "0 0 0 1px rgba(220, 38, 38, 0.2)",
};
const stepCircle = {
    width: 34,
    height: 34,
    borderRadius: 999,
    display: "grid",
    placeItems: "center",
    background: "#0f172a",
    color: "#ffffff",
    fontWeight: 900,
    fontSize: 14,
};
const stepLabel = {
    fontSize: 13,
    fontWeight: 800,
    textAlign: "center",
    lineHeight: 1.25,
};
const wizardPanel = {
    minWidth: 0,
};
const wizardNav = {
    display: "flex",
    justifyContent: "space-between",
    gap: 12,
    flexWrap: "wrap",
};
const finalActionBox = {
    marginTop: 16,
    borderRadius: 16,
    border: "1px solid #bbf7d0",
    background: "#f0fdf4",
    padding: 14,
};
const finalActionTitle = {
    color: "#065f46",
    fontWeight: 900,
    fontSize: 14,
};
const finalActionText = {
    color: "#166534",
    fontSize: 13,
    marginTop: 4,
    lineHeight: 1.45,
};
const grid = { display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 16 };
const field = { display: "grid", gap: 8 };
const label = { color: "#334155", fontSize: 14, fontWeight: 700 };
const searchWrap = { display: "grid", gridTemplateColumns: "1fr auto", gap: 10, alignItems: "center" };
const compactSearchWrap = { display: "grid", gridTemplateColumns: "minmax(0, 1fr) 34px", gap: 4, alignItems: "center" };
const input = { width: "100%", boxSizing: "border-box", borderRadius: 14, border: "1px solid #cbd5e1", padding: "11px 12px", font: "inherit" };
const compactInput = { ...input, borderRadius: 12, padding: "8px 9px", minWidth: 0, height: 38 };
const disabledInput = { background: "#f8fafc", color: "#94a3b8", cursor: "not-allowed" };
const compactOperatorInput = { ...compactInput, minWidth: 0 };
const browseButton = { borderRadius: 14, border: "1px solid #cbd5e1", background: "#fff", color: "#334155", padding: "11px 14px", fontWeight: 800, cursor: "pointer", minWidth: 48 };
const compactBrowseButton = { ...browseButton, borderRadius: 12, padding: 0, minWidth: 0, width: 34, height: 38 };
const disabledButton = { opacity: 0.55, cursor: "not-allowed" };
const suggestionPanel = { marginTop: 10, maxHeight: 220, overflowY: "auto", borderRadius: 14, border: "1px solid #e2e8f0", background: "#fff", padding: 6, display: "grid", gap: 4 };
const suggestionItem = { textAlign: "left", border: "none", background: "transparent", padding: "10px 12px", borderRadius: 10, cursor: "pointer", color: "#0f172a" };
const activeSuggestionItem = { background: "#eff6ff", color: "#1d4ed8", fontWeight: 700 };
const chipRow = { display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12 };
const selectedChip = { borderRadius: 999, border: "1px solid #fdba74", background: "#fff7ed", color: "#9a3412", padding: "7px 11px", cursor: "pointer", fontWeight: 700 };
const hint = { marginTop: 10, color: "#64748b", fontSize: 13 };
const errorBox = { marginTop: 12, borderRadius: 14, background: "#fff1f2", border: "1px solid #fecdd3", color: "#b91c1c", padding: "10px 12px", fontSize: 13, lineHeight: 1.5 };
const warningBox = { marginBottom: 12, borderRadius: 14, background: "#fff7ed", border: "1px solid #fdba74", color: "#9a3412", padding: "10px 12px", fontSize: 13, lineHeight: 1.5 };
const statusRow = { display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 };
const statusChip = { borderRadius: 999, border: "1px solid #cbd5e1", background: "#f8fafc", color: "#334155", padding: "6px 10px", fontSize: 13, fontWeight: 700 };
const dateGrid = { display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 12, marginBottom: 12 };
const actionBar = { display: "flex", gap: 10, flexWrap: "wrap", marginTop: 12 };
const primaryButton = { border: "none", borderRadius: 14, background: "#0f766e", color: "#fff", padding: "12px 16px", fontWeight: 800, cursor: "pointer" };
const secondaryButton = { border: "1px solid #cbd5e1", borderRadius: 14, background: "#fff", color: "#0f172a", padding: "12px 16px", fontWeight: 800, cursor: "pointer" };
const successButton = { ...primaryButton, background: "#16a34a" };
const stack = { display: "grid", gap: 12, marginTop: 12 };
const ruleRow = {
    display: "grid",
    gridTemplateColumns: "minmax(112px, 1fr) 88px minmax(112px, 1fr) 38px",
    gap: 6,
    alignItems: "start",
};
const ruleRemoveButton = {
    border: "1px solid #fecaca",
    borderRadius: 10,
    background: "#fff",
    color: "#b91c1c",
    width: 38,
    height: 38,
    padding: 0,
    fontSize: 16,
    fontWeight: 900,
    lineHeight: "36px",
    cursor: "pointer",
};
const selectWrap = { marginBottom: 12 };
const compactSelectWrap = { minWidth: 0 };
