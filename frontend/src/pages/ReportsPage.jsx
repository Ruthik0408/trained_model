import React, { useEffect, useMemo, useState } from "react";
import { getWorkbenchDatasets, getWorkbenchReport, getWorkbenchReviewRows, } from "../api/anomalyApi";
const FILTER_OPTIONS = [
    { label: "All", value: "all" },
    { label: "Rule", value: "rule" },
    { label: "ML", value: "ml" },
    { label: "Not Reviewed", value: "not_reviewed" },
    { label: "Reviewed", value: "reviewed" },
];
// Fixed orbit positions (matches the mock layout). cx/cy are SVG coords on a 1000x480 canvas.
// leftPct / topPct are the corresponding CSS percentages used to position the bulb DOM nodes.
const NETWORK_POSITIONS = [
    { leftPct: "20%", topPct: "28%", cx: 200, cy: 135, size: 60, tone: "red" },
    { leftPct: "17.5%", topPct: "60%", cx: 175, cy: 290, size: 62, tone: "red" },
    { leftPct: "34%", topPct: "84%", cx: 340, cy: 405, size: 60, tone: "red" },
    { leftPct: "52%", topPct: "14%", cx: 520, cy: 70, size: 62, tone: "orange" },
    { leftPct: "85%", topPct: "25%", cx: 850, cy: 120, size: 62, tone: "blue" },
    { leftPct: "87%", topPct: "51%", cx: 870, cy: 245, size: 62, tone: "blue" },
    { leftPct: "77%", topPct: "78%", cx: 770, cy: 375, size: 62, tone: "orange" },
];
// Approver junction dots placed along the curved edges between center and rim nodes.
const APPROVER_DOTS = [
    { left: "32%", top: "43%" },
    { left: "56%", top: "36%" },
    { left: "66%", top: "42%" },
    { left: "69%", top: "55%" },
    { left: "64%", top: "71%" },
];
const BREAKUP_COLORS = ["#ff4d4f", "#ff9820", "#ffbc36", "#57c768", "#2b8ef9"];
const RISK_COLORS = ["#ff4d4f", "#ff9820", "#57c768"];
export default function ReportsPage({ latestWorkbenchRun }) {
    const [datasets, setDatasets] = useState([]);
    const [datasetTable, setDatasetTable] = useState("");
    const [anomalyFilter, setAnomalyFilter] = useState("all");
    const [report, setReport] = useState(null);
    const [detailRows, setDetailRows] = useState([]);
    const [loading, setLoading] = useState(true);
    const [loadError, setLoadError] = useState("");
    const [showLabels, setShowLabels] = useState(true);
    const [searchQuery, setSearchQuery] = useState("");
    const [riskLevelFilter, setRiskLevelFilter] = useState("all");
    const [statusFilter, setStatusFilter] = useState("all");
    const [dateFromFilter, setDateFromFilter] = useState("");
    const [dateToFilter, setDateToFilter] = useState("");
    const [showFiltersMenu, setShowFiltersMenu] = useState(false);
    const [showProfileMenu, setShowProfileMenu] = useState(false);
    const [showAllAlerts, setShowAllAlerts] = useState(false);
    const [viewportWidth, setViewportWidth] = useState(() => typeof window === "undefined" ? 1400 : window.innerWidth);
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
    useEffect(() => {
        if (latestWorkbenchRun?.datasetTable) {
            setDatasetTable(latestWorkbenchRun.datasetTable);
        }
    }, [latestWorkbenchRun]);
    useEffect(() => {
        if (typeof window === "undefined")
            return;
        const handleResize = () => setViewportWidth(window.innerWidth);
        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, []);
    useEffect(() => {
        const load = async () => {
            setLoading(true);
            setLoadError("");
            try {
                const preferredRunId = hydratedLatestDataset?.dataset_table === datasetTable ? hydratedLatestDataset.run_id : undefined;
                const preferredDatasetTable = datasetTable || hydratedLatestDataset?.dataset_table;
                const [datasetsResult, reportResult, detailResult] = await Promise.allSettled([
                    getWorkbenchDatasets(),
                    getWorkbenchReport({
                        datasetTable: preferredDatasetTable,
                        runId: preferredRunId,
                    }),
                    getWorkbenchReviewRows({
                        datasetTable: datasetTable || undefined,
                        anomalyFilter,
                        runId: preferredRunId,
                    }),
                ]);
                const datasetsResponse = datasetsResult.status === "fulfilled" ? datasetsResult.value : null;
                const reportResponse = reportResult.status === "fulfilled" ? reportResult.value : null;
                const detailResponse = detailResult.status === "fulfilled" ? detailResult.value : null;
                const fetchedDatasets = datasetsResponse?.data || [];
                const nextDatasets = [...fetchedDatasets];
                if (hydratedLatestDataset &&
                    !nextDatasets.some((item) => item.dataset_table === hydratedLatestDataset.dataset_table &&
                        item.run_id === hydratedLatestDataset.run_id)) {
                    nextDatasets.unshift(hydratedLatestDataset);
                }
                const responseDataset = detailResponse?.data?.dataset_table ||
                    hydratedLatestDataset?.dataset_table ||
                    nextDatasets[0]?.dataset_table ||
                    "";
                if (!datasetTable && responseDataset) {
                    setDatasetTable(responseDataset);
                }
                setDatasets(nextDatasets);
                setReport(reportResponse?.data || null);
                setDetailRows(detailResponse?.data?.rows || []);
                if (datasetsResult.status === "rejected" &&
                    reportResult.status === "rejected" &&
                    detailResult.status === "rejected") {
                    const reason = reportResult.reason || detailResult.reason || datasetsResult.reason;
                    const detail = reason?.response?.data?.detail || reason?.message || "Unable to load report data right now.";
                    setLoadError(typeof detail === "string" ? detail : "Unable to load report data right now.");
                }
            }
            finally {
                setLoading(false);
            }
        };
        load();
    }, [datasetTable, anomalyFilter, hydratedLatestDataset]);
    const activeDataset = useMemo(() => datasets.find((item) => item.dataset_table === datasetTable &&
        (hydratedLatestDataset?.dataset_table !== datasetTable ||
            hydratedLatestDataset.run_id == null ||
            item.run_id === hydratedLatestDataset.run_id)) ||
        (hydratedLatestDataset?.dataset_table === datasetTable ? hydratedLatestDataset : null) ||
        datasets.find((item) => item.dataset_table === datasetTable) ||
        hydratedLatestDataset, [datasetTable, datasets, hydratedLatestDataset]);
    const totalSourceRows = report?.total_rows || activeDataset?.total_rows || latestWorkbenchRun?.totalRows || 0;
    const dashboardRows = useMemo(() => {
        return detailRows.map((row) => {
            const payload = row.row_payload_json || {};
            const anomalyType = joinReasons(row) || "No reason";
            const amount = Number(resolvePayloadValue(payload, ["amount", "amount_claimed", "amount_passed", "invoice_amount"]) || 0);
            const riskScore = Math.max(35, Math.round(Number(severityScore(row) * 100)));
            return {
                predictionId: row.prediction_id,
                billNo: resolvePayloadValue(payload, ["bill_no", "invoice_no", "invoice_number", "reference_no", "bill_number"]) || "Unavailable",
                vendorName: resolvePayloadValue(payload, ["resolved_vendor_name", "vendor_name", "vendor", "supplier_name", "party_name"]) || "Unavailable",
                office: resolvePayloadValue(payload, ["resolved_office_name", "office_name", "office", "branch", "location"]) || "Unavailable",
                anomalyType,
                amount,
                riskScore,
                riskLabel: riskScore >= 85 ? "High" : riskScore >= 60 ? "Medium" : "Low",
                detectedOn: resolvePayloadValue(payload, ["detected_on", "reference_date", "invoice_date", "posting_date", "created_at"]) || "Unavailable",
                status: mapReviewStatus(row.review_status, row.feedback),
                feedback: row.feedback ? capitalize(row.feedback) : "Pending",
                entityType: inferEntityType(payload),
            };
        });
    }, [detailRows]);
    const filteredDashboardRows = useMemo(() => {
        const normalizedSearch = searchQuery.trim().toLowerCase();
        const fromTime = dateFromFilter ? new Date(`${dateFromFilter}T00:00:00`).getTime() : null;
        const toTime = dateToFilter ? new Date(`${dateToFilter}T23:59:59`).getTime() : null;
        return dashboardRows.filter((row) => {
            if (riskLevelFilter !== "all" && row.riskLabel !== riskLevelFilter) {
                return false;
            }
            if (statusFilter !== "all" && row.status !== statusFilter) {
                return false;
            }
            if (normalizedSearch) {
                const haystack = [row.billNo, row.vendorName, row.office, row.anomalyType].join(" ").toLowerCase();
                if (!haystack.includes(normalizedSearch)) {
                    return false;
                }
            }
            if (fromTime != null || toTime != null) {
                const parsedDate = parseDisplayDate(row.detectedOn);
                if (!parsedDate || Number.isNaN(parsedDate.getTime())) {
                    return false;
                }
                const detectedTime = parsedDate.getTime();
                if (fromTime != null && detectedTime < fromTime) {
                    return false;
                }
                if (toTime != null && detectedTime > toTime) {
                    return false;
                }
            }
            return true;
        });
    }, [dashboardRows, searchQuery, riskLevelFilter, statusFilter, dateFromFilter, dateToFilter]);
    const filteredPredictionIds = useMemo(() => new Set(filteredDashboardRows.map((row) => row.predictionId)), [filteredDashboardRows]);
    const filteredDetailRows = useMemo(() => detailRows.filter((row) => filteredPredictionIds.has(row.prediction_id)), [detailRows, filteredPredictionIds]);
    const riskBuckets = useMemo(() => {
        let high = 0;
        let medium = 0;
        let low = 0;
        for (const row of filteredDashboardRows) {
            if (row.riskLabel === "High")
                high += 1;
            else if (row.riskLabel === "Medium")
                medium += 1;
            else
                low += 1;
        }
        return { high, medium, low };
    }, [filteredDashboardRows]);
    const reasonStats = useMemo(() => {
        const counts = new Map();
        for (const row of filteredDetailRows) {
            const reasons = Array.isArray(row.reasons_json?.reason_list) ? row.reasons_json.reason_list : [];
            if (reasons.length === 0) {
                counts.set("Others", (counts.get("Others") || 0) + 1);
                continue;
            }
            for (const reason of reasons) {
                const label = simplifyReason(formatReason(reason));
                counts.set(label, (counts.get(label) || 0) + 1);
            }
        }
        const fallbackFromTable = filteredDashboardRows.reduce((acc, row) => {
            const label = simplifyReason(formatReason(row.anomalyType));
            acc.set(label, (acc.get(label) || 0) + 1);
            return acc;
        }, new Map());
        const source = counts.size > 0 ? counts : fallbackFromTable;
        return Array.from(source.entries())
            .map(([label, count]) => ({
            label,
            count,
            percent: filteredDashboardRows.length > 0 ? (count / filteredDashboardRows.length) * 100 : 0,
        }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 5);
    }, [filteredDetailRows, filteredDashboardRows]);
    const highlights = useMemo(() => {
        return filteredDashboardRows
            .slice()
            .sort((a, b) => b.riskScore - a.riskScore)
            .slice(0, showAllAlerts ? filteredDashboardRows.length : 4)
            .map((row) => ({
            id: row.predictionId,
            title: row.anomalyType,
            description: `${row.vendorName} linked to ${row.billNo}`,
            note: `${row.office} • ${formatIndianCurrency(row.amount)}`,
            tone: row.riskLabel === "High" ? "high" : row.riskLabel === "Medium" ? "medium" : "low",
        }));
    }, [filteredDashboardRows, showAllAlerts]);
    const networkNodes = useMemo(() => {
        const rows = dashboardRows.slice(0, NETWORK_POSITIONS.length);
        return rows.map((row, index) => {
            const position = NETWORK_POSITIONS[index];
            const label = row.entityType === "office" ? "OFFICE" : row.entityType === "vendor" ? "VENDOR" : "BILL";
            return {
                id: String(row.predictionId),
                label,
                sublabel: row.entityType === "vendor" ? row.vendorName : row.billNo,
                meta: row.entityType === "office" ? `${row.office}` : formatIndianCurrency(row.amount),
                cx: position.cx,
                cy: position.cy,
                leftPct: position.leftPct,
                topPct: position.topPct,
                size: position.size,
                tone: position.tone,
            };
        });
    }, [filteredDashboardRows]);
    const approverNodes = useMemo(() => {
        const seen = new Set();
        const approvers = [];
        for (const row of detailRows) {
            const payload = row.row_payload_json || {};
            const candidates = [
                resolvePayloadValue(payload, ["approver_emp_id", "approver_id", "employee_id", "emp_id", "user_id"]),
                resolvePayloadValue(payload, ["approver_code", "approver_name"]),
            ].filter((value) => value && value !== "Unavailable");
            for (const candidate of candidates) {
                const normalized = candidate.trim();
                if (!normalized || seen.has(normalized)) {
                    continue;
                }
                seen.add(normalized);
                const position = APPROVER_DOTS[approvers.length];
                if (!position) {
                    return approvers;
                }
                approvers.push({ ...position, empId: normalized });
                break;
            }
        }
        return approvers;
    }, [filteredDetailRows]);
    const sparkData = useMemo(() => buildSparkData(filteredDashboardRows), [filteredDashboardRows]);
    const layout = getLayout(viewportWidth);
    const activeTables = report?.selected_tables?.length ? report.selected_tables : activeDataset?.selected_tables || [];
    const filteredAnomalies = filteredDashboardRows.length;
    const filteredReviewedCount = filteredDashboardRows.filter((row) => row.status === "Under Review" || row.status === "Approved").length;
    const filteredAcceptedCount = filteredDashboardRows.filter((row) => row.status === "Approved").length;
    const filteredPendingCount = Math.max(filteredAnomalies - filteredReviewedCount, 0);
    const financialImpact = filteredDashboardRows.reduce((sum, row) => sum + row.amount, 0);
    const highRiskPercent = filteredAnomalies > 0 ? (riskBuckets.high / filteredAnomalies) * 100 : 0;
    const breakupConicGradient = buildConicGradient(reasonStats, BREAKUP_COLORS);
    const riskConicGradient = buildConicGradient([
        { count: riskBuckets.high },
        { count: riskBuckets.medium },
        { count: riskBuckets.low },
    ], RISK_COLORS);
    const centerVendorRow = filteredDashboardRows[0];
    const notificationCount = filteredAnomalies;
    return (<div style={page}>
      <style>{pageStyles}</style>

      {/* ============== TOOLBAR ============== */}
      <header style={toolbar}>
        <div style={brandBlock}>
          <div style={brandLogo}>
            <svg width="38" height="38" viewBox="0 0 40 40" fill="none" aria-hidden="true">
              <path d="M9 24 Q4 28 5 33 Q11 32 13 27" fill="#7ec85c"/>
              <path d="M31 24 Q36 28 35 33 Q29 32 27 27" fill="#7ec85c"/>
              <ellipse cx="13" cy="14" rx="6" ry="9" fill="#ff5d6e"/>
              <ellipse cx="27" cy="14" rx="6" ry="9" fill="#ff5d6e"/>
              <ellipse cx="20" cy="11" rx="6" ry="10" fill="#ff8a98"/>
              <path d="M20 21 L20 31" stroke="#5fa844" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </div>
          <div>
            <div style={brandTitle}>Tulip 2.0</div>
            <div style={brandSubtitle}>Anomaly Detection Dashboard</div>
          </div>
        </div>

        <div style={{ ...toolbarFilters, gridTemplateColumns: layout.toolbarColumns }}>
          <DateRangeCard label="Date Range" fromValue={dateFromFilter} toValue={dateToFilter} onFromChange={setDateFromFilter} onToChange={setDateToFilter} onClear={() => {
            setDateFromFilter("");
            setDateToFilter("");
        }}/>
          <SelectCard label="PAO" value={datasetTable} options={datasets.map((item) => ({
            label: item.dataset_table,
            value: item.dataset_table,
        }))} onChange={setDatasetTable} emptyLabel="All PAOs"/>
          <SelectCard label="Anomaly Type" value={anomalyFilter} options={FILTER_OPTIONS.map((item) => ({ label: item.label, value: item.value }))} onChange={(value) => setAnomalyFilter(value)}/>
          <SelectCard label="Risk Level" value={riskLevelFilter} options={[
            { label: "All Risk Levels", value: "all" },
            { label: "High", value: "High" },
            { label: "Medium", value: "Medium" },
            { label: "Low", value: "Low" },
        ]} onChange={(value) => setRiskLevelFilter(value)}/>
        </div>

        <div style={profileWrap}>
          <div style={bellWrap}>
            <BellIcon />
            <span style={bellBadge}>{notificationCount}</span>
          </div>
          <button type="button" style={profileButton} onClick={() => setShowProfileMenu((value) => !value)}>
            <div style={profileText}>
              <div style={profileName}>Admin</div>
              <div style={profileMeta}>CGDA</div>
            </div>
            <ChevronDownIcon />
          </button>
          {showProfileMenu ? (<div style={profileMenu}>
              <div style={profileMenuTitle}>Admin Panel</div>
              <div style={profileMenuRow}><span>Dataset</span><strong>{datasetTable || "All"}</strong></div>
              <div style={profileMenuRow}><span>Visible alerts</span><strong>{formatIndianNumber(filteredAnomalies)}</strong></div>
              <div style={profileMenuRow}><span>Risk filter</span><strong>{riskLevelFilter}</strong></div>
            </div>) : null}
        </div>
      </header>

      {!loading && loadError ? <div style={errorBanner}>{loadError}</div> : null}

      {/* ============== MAIN GRID ============== */}
      <div style={{ ...contentGrid, gridTemplateColumns: layout.contentColumns }}>
        {/* ---------- LEFT RAIL ---------- */}
        <aside style={railColumn}>
          <section style={panelCard}>
            <div style={panelHeading}>ANOMALY SUMMARY</div>
            <div style={summaryStack}>
              <SummaryStatCard accent="#2b8ef9" bg="#e8f2ff" icon={<DocumentIcon tone="#2b8ef9"/>} label="Total Bills Processed" value={loading ? "..." : formatIndianNumber(totalSourceRows)}/>
              <SummaryStatCard accent="#ff4d4f" bg="#ffecec" icon={<AlertIcon tone="#ff4d4f"/>} label="Total Anomalies Detected" value={loading ? "..." : formatIndianNumber(filteredAnomalies)} subtext={`${totalSourceRows > 0 ? ((filteredAnomalies / totalSourceRows) * 100).toFixed(2) : "0.00"}% of total bills`}/>
              <SummaryStatCard accent="#ff9d22" bg="#fff2dd" icon={<RupeeIcon tone="#ff9d22"/>} label="Total Financial Impact" value={loading ? "..." : formatIndianCurrencyCompact(financialImpact)}/>
              <SummaryStatCard accent="#8a56f5" bg="#efe9fc" icon={<ShieldIcon tone="#8a56f5"/>} label="High Risk Anomalies" value={loading ? "..." : formatIndianNumber(riskBuckets.high)} subtext={`${highRiskPercent.toFixed(2)}% of total anomalies`}/>
            </div>
          </section>

          <section style={panelCard}>
            <div style={panelHeading}>ANOMALY BREAKUP</div>
            <div style={donutColumn}>
              <div style={{ ...donut, background: breakupConicGradient || "#eef2f9" }}>
                <div style={donutCore}>
                  <div style={donutCoreValue}>{formatIndianNumber(filteredAnomalies)}</div>
                  <div style={donutCoreLabel}>Total</div>
                </div>
              </div>
              <div style={donutLegendStack}>
                {reasonStats.length === 0 ? (<div style={emptyTagText}>No breakdown available.</div>) : (reasonStats.map((item, index) => (<React.Fragment key={item.label}>
                      <LegendLine color={BREAKUP_COLORS[index % BREAKUP_COLORS.length]} label={item.label} value={`${formatIndianNumber(item.count)} (${item.percent.toFixed(1)}%)`}/>
                    </React.Fragment>)))}
              </div>
            </div>
          </section>

          <section style={panelCard}>
            <div style={panelHeading}>ANOMALY TREND</div>
            <TrendPanel data={sparkData}/>
          </section>
        </aside>

        {/* ---------- CENTER COLUMN ---------- */}
        <main style={centerColumn}>
          {/* 3D NETWORK */}
          <section style={panelCard}>
            <div style={networkHead}>
              <div style={networkControls}>
                <InlineSelect value={riskLevelFilter} onChange={(value) => setRiskLevelFilter(value)} options={[
            { label: "All Risk Levels", value: "all" },
            { label: "High", value: "High" },
            { label: "Medium", value: "Medium" },
            { label: "Low", value: "Low" },
        ]}/>
                <ToggleSwitch checked={showLabels} onChange={() => setShowLabels((value) => !value)} label="Show Labels"/>
                <button type="button" style={ctrlButton} onClick={() => {
            setRiskLevelFilter("all");
            setShowLabels(true);
        }}>
                  Reset View
                </button>
              </div>
            </div>

            <div style={networkCanvas}>
              <div style={gridFloor}></div>

              {/* Connection lines + approver junction dots */}
              <svg style={netSvg} viewBox="0 0 1000 480" preserveAspectRatio="none">
                <defs>
                  <linearGradient id="gradRed" x1="0" x2="1">
                    <stop offset="0%" stopColor="#ff4d4f" stopOpacity="0.85"/>
                    <stop offset="100%" stopColor="#ff4d4f" stopOpacity="0.18"/>
                  </linearGradient>
                  <linearGradient id="gradOrange" x1="0" x2="1">
                    <stop offset="0%" stopColor="#ff9820" stopOpacity="0.85"/>
                    <stop offset="100%" stopColor="#ff9820" stopOpacity="0.18"/>
                  </linearGradient>
                  <linearGradient id="gradBlue" x1="0" x2="1">
                    <stop offset="0%" stopColor="#2b8ef9" stopOpacity="0.85"/>
                    <stop offset="100%" stopColor="#2b8ef9" stopOpacity="0.18"/>
                  </linearGradient>
                </defs>

                {/* Center → rim curved paths */}
                {networkNodes.map((node) => (<path key={`spoke-${node.id}`} d={curvePath(500, 240, node.cx, node.cy)} stroke={`url(#${gradId(node.tone)})`} strokeWidth="3" fill="none" strokeLinecap="round"/>))}

                {/* Rim-to-rim faint cross connections (between consecutive nodes) */}
                {networkNodes.map((node, index) => {
            const next = networkNodes[(index + 1) % networkNodes.length];
            if (!next)
                return null;
            return (<path key={`cross-${node.id}-${next.id}`} d={curvePath(node.cx, node.cy, next.cx, next.cy, 0.35)} stroke={`url(#${gradId(node.tone)})`} strokeWidth="1.5" fill="none" strokeLinecap="round" opacity="0.5"/>);
        })}

                {/* Approver junction dots */}
                {approverNodes.map((dot, index) => {
            const x = (parseFloat(dot.left) / 100) * 1000;
            const y = (parseFloat(dot.top) / 100) * 480;
            return (<circle key={`approver-${index}`} cx={x} cy={y} r="5" fill="#57c768" stroke="#fff" strokeWidth="1.5"/>);
        })}
              </svg>

              {/* Approver micro-labels */}
              {showLabels &&
            approverNodes.map((dot, index) => (<div key={`approver-label-${index}`} style={{
                    ...approverLabel,
                    left: dot.left,
                    top: dot.top,
                }}>
                    <div style={approverLabelKind}>APPROVER</div>
                    <div>Emp ID: {dot.empId}</div>
                  </div>))}

              {/* Center vendor bulb */}
              <div style={{ ...nodeWrap, left: "50%", top: "50%" }}>
                <Bulb tone="red" size={90}/>
              </div>

              {/* Center info card */}
              <div style={centerCard}>
                <div style={centerCardKind}>VENDOR</div>
                <div style={centerCardName}>{centerVendorRow?.vendorName || "Unavailable"}</div>
                <div style={centerCardScore}>Risk Score: {centerVendorRow?.riskScore ?? 0}</div>
              </div>

              {/* Rim nodes */}
              {networkNodes.map((node) => (<div key={node.id} style={{ ...nodeWrap, left: node.leftPct, top: node.topPct }}>
                  <Bulb tone={node.tone} size={node.size}/>
                  {showLabels ? (<div style={nodeLabelBlock}>
                      <div style={nodeLabelKind}>{node.label}</div>
                      <div style={nodeLabelName}>{node.sublabel}</div>
                      <div style={nodeLabelMeta}>{node.meta}</div>
                    </div>) : null}
                </div>))}
            </div>

            <div style={netFoot}>
              <div style={legendBox}>
                <span style={legendBoxLabel}>LEGEND:</span>
                <LegendChip color="#ff4d4f" label="High Risk"/>
                <LegendChip color="#ff9d22" label="Medium Risk"/>
                <LegendChip color="#57c768" label="Low Risk"/>
                <LegendChip color="#2b8ef9" label="Normal"/>
              </div>
              <div style={legendBox}>
                <EntityChip tone="#ff4d4f" label="Vendor"/>
                <EntityChip tone="#ff9d22" label="Bill"/>
                <EntityChip tone="#2b8ef9" label="Office"/>
                <EntityChip tone="#57c768" label="Approver"/>
              </div>
            </div>
          </section>

          {/* ANOMALY LIST */}
          <section style={panelCard}>
            <div style={tableHeadBar}>
              <div style={tableTitle}>ANOMALY LIST</div>
              <div style={tableActions}>
                <div style={searchBox}>
                  <SearchIcon />
                  <input value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="Search Bill / Vendor / Office" style={searchInput}/>
                </div>
                <div style={filterMenuWrap}>
                  <button type="button" style={filterBtn} onClick={() => setShowFiltersMenu((value) => !value)}>
                    <FilterIcon />
                    Filters
                    <ChevronDownIcon />
                  </button>
                  {showFiltersMenu ? (<div style={filterMenu}>
                      <div style={filterMenuGrid}>
                        <SelectCard label="Status" value={statusFilter} options={[
                { label: "All Statuses", value: "all" },
                { label: "Pending", value: "Pending" },
                { label: "Under Review", value: "Under Review" },
                { label: "Approved", value: "Approved" },
            ]} onChange={(value) => setStatusFilter(value)}/>
                      </div>
                      <button type="button" style={clearFilterButton} onClick={() => {
                setRiskLevelFilter("all");
                setStatusFilter("all");
                setDateFromFilter("");
                setDateToFilter("");
                setSearchQuery("");
                setShowFiltersMenu(false);
            }}>
                        Clear Filters
                      </button>
                    </div>) : null}
                </div>
              </div>
            </div>

            <div style={scrollingTableWrapper}>
              <table style={reportTable}>
                <thead>
                  <tr>
                    <th style={tableHeadCell}>Bill No.</th>
                    <th style={tableHeadCell}>Vendor Name</th>
                    <th style={tableHeadCell}>Anomaly Type</th>
                    <th style={tableHeadCell}>Amount (₹)</th>
                    <th style={tableHeadCell}>Risk Score</th>
                    <th style={tableHeadCell}>Office</th>
                    <th style={tableHeadCell}>Detected On</th>
                    <th style={tableHeadCell}>Status</th>
                    <th style={tableHeadCell}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {loading ? (<tr>
                      <td colSpan={9} style={emptyTableCell}>Loading results...</td>
                    </tr>) : filteredDashboardRows.length === 0 ? (<tr>
                      <td colSpan={9} style={emptyTableCell}>No results available for this selection.</td>
                    </tr>) : (filteredDashboardRows.slice(0, 6).map((row) => (<tr key={row.predictionId} style={tableBodyRow}>
                        <td style={tableBodyCell}>{row.billNo}</td>
                        <td style={tableBodyCell}>{row.vendorName}</td>
                        <td style={tableBodyCell}>{truncateText(row.anomalyType, 34)}</td>
                        <td style={tableBodyCell}>{formatIndianNumber(row.amount)}</td>
                        <td style={tableBodyCell}>
                          <span style={riskScorePill(row.riskLabel)}>
                            <UserMiniIcon />
                            {row.riskScore}
                          </span>
                        </td>
                        <td style={tableBodyCell}>{row.office}</td>
                        <td style={tableBodyCell}>{row.detectedOn}</td>
                        <td style={tableBodyCell}>
                          <span style={statusPill(row.status)}>{row.status}</span>
                        </td>
                        <td style={tableBodyCell}>
                          <div style={actionRow}>
                            <ActionButton><EyeIcon /></ActionButton>
                            <ActionButton><FileIcon /></ActionButton>
                            <ActionButton><FlagIcon /></ActionButton>
                          </div>
                        </td>
                      </tr>)))}
                </tbody>
              </table>
            </div>
          </section>
        </main>

        {/* ---------- RIGHT RAIL ---------- */}
        <aside style={railColumn}>
          {/* HIGHLIGHTS */}
          <section style={panelCard}>
            <div style={panelHeading}>ANOMALY HIGHLIGHTS</div>
            <div style={scrollHighlightsWrap}>
              <div style={highlightsStack}>
                {loading ? (<div style={emptyBox}>Loading highlights...</div>) : highlights.length === 0 ? (<div style={emptyBox}>No anomalies available yet.</div>) : (highlights.map((item) => (<React.Fragment key={item.id}>
                      <HighlightCard item={item}/>
                    </React.Fragment>)))}
              </div>
            </div>
            <button type="button" style={viewAllButton} onClick={() => setShowAllAlerts((value) => !value)}>
              {showAllAlerts ? "Show Fewer Alerts" : "View All Alerts"}
            </button>
          </section>

          {/* RISK DISTRIBUTION */}
          <section style={panelCard}>
            <div style={panelHeading}>RISK DISTRIBUTION</div>
            <div style={donutColumn}>
              <div style={{ ...donut, background: riskConicGradient || "#eef2f9" }}>
                <div style={donutCore}>
                  <div style={donutCoreValue}>{formatIndianNumber(filteredAnomalies)}</div>
                  <div style={donutCoreLabel}>Total</div>
                </div>
              </div>
              <div style={donutLegendStack}>
                <LegendLine color={RISK_COLORS[0]} label="High Risk" value={`${formatIndianNumber(riskBuckets.high)} (${formatPercent(riskBuckets.high, filteredAnomalies)})`}/>
                <LegendLine color={RISK_COLORS[1]} label="Medium Risk" value={`${formatIndianNumber(riskBuckets.medium)} (${formatPercent(riskBuckets.medium, filteredAnomalies)})`}/>
                <LegendLine color={RISK_COLORS[2]} label="Low Risk" value={`${formatIndianNumber(riskBuckets.low)} (${formatPercent(riskBuckets.low, filteredAnomalies)})`}/>
              </div>
            </div>
          </section>

          {/* RUN CONTEXT (preserved from original) */}
          <section style={panelCard}>
            <div style={panelHeading}>RUN CONTEXT</div>
            <div style={contextStack}>
              <ContextRow label="Run Name" value={report?.run_name || activeDataset?.run_name || latestWorkbenchRun?.runName || "Unavailable"}/>
              <ContextRow label="Accepted" value={formatIndianNumber(filteredAcceptedCount)}/>
              <ContextRow label="Pending Review" value={formatIndianNumber(filteredPendingCount)}/>
              <ContextRow label="Source Tables" value={String(activeTables.length || 0)}/>
            </div>
            <div style={tableTags}>
              {activeTables.length > 0 ? (activeTables.slice(0, 4).map((table) => (<span key={table} style={contextTag}>{table}</span>))) : (<div style={emptyTagText}>No source table metadata available.</div>)}
            </div>
          </section>
        </aside>
      </div>
    </div>);
}
/* ===================== Subcomponents ===================== */
function SummaryStatCard({ icon, label, value, subtext, accent, bg, }) {
    return (<div style={statCard}>
      <div style={{ ...statIconWrap, background: bg }}>{icon}</div>
      <div>
        <div style={statLabel}>{label}</div>
        <div style={{ ...statValue, color: accent }}>{value}</div>
        {subtext ? <div style={statSubtext}>{subtext}</div> : null}
      </div>
    </div>);
}
function SelectCard({ label, value, options, onChange, emptyLabel, }) {
    return (<div style={filterShell}>
      <div style={filterLabel}>{label}</div>
      <div style={filterValueRow}>
        <select value={value} onChange={(event) => onChange(event.target.value)} style={selectInput}>
          {!value && emptyLabel ? <option value="">{emptyLabel}</option> : null}
          {options.map((option) => (<option key={option.value} value={option.value}>
              {option.label}
            </option>))}
        </select>
      </div>
    </div>);
}
function DateRangeCard({ label, fromValue, toValue, onFromChange, onToChange, onClear, }) {
    return (<div style={filterShell}>
      <div style={filterLabel}>{label}</div>
      <div style={dateRangeRow}>
        <span style={filterValueLeft}>
          <CalendarIcon />
          <input type="date" value={fromValue} onChange={(event) => onFromChange(event.target.value)} style={dateInput}/>
        </span>
        <span style={dateRangeDivider}>to</span>
        <input type="date" value={toValue} onChange={(event) => onToChange(event.target.value)} style={dateInput}/>
        {(fromValue || toValue) ? (<button type="button" style={tinyGhostButton} onClick={onClear}>Clear</button>) : null}
      </div>
    </div>);
}
function HighlightCard({ item }) {
    const color = item.tone === "high" ? "#ff4d4f" : item.tone === "medium" ? "#ea7d00" : "#2f9f45";
    return (<div style={highlightCard}>
      <div style={highlightIconWrap}>
        <AlertIcon tone={color}/>
      </div>
      <div>
        <div style={{ ...highlightTitle, color }}>{item.title}</div>
        <div style={highlightDesc}>{item.description}</div>
        <div style={highlightNote}>{item.note}</div>
      </div>
    </div>);
}
function TrendPanel({ data }) {
    if (data.length === 0) {
        return <div style={emptyBox}>No trend data available yet.</div>;
    }
    const max = Math.max(...data.flatMap((item) => [item.all, item.high, item.impact]), 1);
    const allPoints = buildPolyline(data.map((item) => item.all), max);
    const highPoints = buildPolyline(data.map((item) => item.high), max);
    const impactPoints = buildPolyline(data.map((item) => item.impact), max);
    return (<div style={trendShell}>
      <svg viewBox="0 0 100 60" preserveAspectRatio="none" style={trendSvg}>
        {[15, 30, 45].map((line) => (<line key={line} x1="0" y1={line} x2="100" y2={line} stroke="rgba(148,163,184,0.18)" strokeWidth="0.4"/>))}
        <polyline fill="none" stroke="#ff3b30" strokeWidth="1.4" points={allPoints}/>
        <polyline fill="none" stroke="#ff9d22" strokeWidth="1.4" points={highPoints}/>
        <polyline fill="none" stroke="#2b8ef9" strokeWidth="1.4" points={impactPoints}/>
      </svg>
      <div style={trendAxis}>
        {data.map((item) => (<span key={item.label}>{item.label}</span>))}
      </div>
      <div style={trendLegend}>
        <div style={trendLegendRow}><span style={{ ...trendSwatch, background: "#ff3b30" }}></span>All Anomalies</div>
        <div style={trendLegendRow}><span style={{ ...trendSwatch, background: "#ff9d22" }}></span>High Risk</div>
        <div style={trendLegendRow}><span style={{ ...trendSwatch, background: "#2b8ef9" }}></span>Financial Impact (Cr)</div>
      </div>
    </div>);
}
function InlineSelect({ value, onChange, options, }) {
    return (<div style={inlineSelectWrap}>
      <select value={value} onChange={(event) => onChange(event.target.value)} style={inlineSelectInput}>
        {options.map((option) => (<option key={option.value} value={option.value}>
            {option.label}
          </option>))}
      </select>
    </div>);
}
function ToggleSwitch({ checked, onChange, label }) {
    return (<button type="button" style={toggleRow} onClick={onChange}>
      <span style={{ ...toggleTrack, justifyContent: checked ? "flex-end" : "flex-start", background: checked ? "#2b8ef9" : "#cfd9e8" }}>
        <span style={toggleThumb}></span>
      </span>
      <span>{label}</span>
    </button>);
}
function Bulb({ tone, size }) {
    const cls = tone === "red" ? bulbRed :
        tone === "orange" ? bulbOrange :
            bulbBlue;
    const iconSize = Math.round(size * 0.45);
    return (<div style={{ ...bulbBase, ...cls, width: size, height: size }}>
      <span style={bulbHighlight}></span>
      <svg width={iconSize} height={iconSize} viewBox="0 0 34 34" fill="none" aria-hidden="true">
        <circle cx="17" cy="13" r="5" fill="rgba(255,255,255,0.95)"/>
        <path d="M7 27c1.6-4.6 5-7 10-7s8.4 2.4 10 7" fill="rgba(255,255,255,0.95)"/>
      </svg>
    </div>);
}
function LegendLine({ color, label, value }) {
    return (<div style={legendLine}>
      <span style={{ ...legendDot, background: color }}></span>
      <span style={legendLineLabel}>{label}</span>
      <span style={legendLineValue}>{value}</span>
    </div>);
}
function LegendChip({ color, label }) {
    return (<span style={legendChip}>
      <span style={{ ...legendChipDot, background: color }}></span>
      {label}
    </span>);
}
function EntityChip({ tone, label }) {
    return (<span style={entityChip}>
      <span style={{ ...entityChipRing, color: tone, borderColor: tone }}>
        <UserMiniIcon />
      </span>
      <span>{label}</span>
    </span>);
}
function ContextRow({ label, value }) {
    return (<div style={contextRow}>
      <span style={contextLabel}>{label}</span>
      <span style={contextValue}>{value}</span>
    </div>);
}
function ActionButton({ children }) {
    return (<button type="button" style={actionButton}>{children}</button>);
}
/* ===================== Helpers ===================== */
function buildSparkData(rows) {
    if (rows.length === 0)
        return [];
    const grouped = new Map();
    for (const row of rows) {
        const parsed = parseDisplayDate(row.detectedOn);
        const groupLabel = parsed ? formatShortDate(parsed) : row.detectedOn || "Unknown";
        const sortKey = parsed ? parsed.getTime() : Number.MAX_SAFE_INTEGER;
        const current = grouped.get(groupLabel) || { label: groupLabel, all: 0, high: 0, impact: 0, sortKey };
        current.all += 1;
        current.high += row.riskLabel === "High" ? 1 : 0;
        current.impact += row.amount / 10000000;
        current.sortKey = Math.min(current.sortKey, sortKey);
        grouped.set(groupLabel, current);
    }
    return Array.from(grouped.values())
        .sort((left, right) => left.sortKey - right.sortKey)
        .slice(-5)
        .map((item) => ({
        label: item.label,
        all: item.all,
        high: item.high,
        impact: Number(item.impact.toFixed(2)),
    }));
}
function buildPolyline(values, max) {
    return values
        .map((value, index) => {
        const x = (index / Math.max(values.length - 1, 1)) * 100;
        const y = 55 - (value / max) * 45;
        return `${x},${y}`;
    })
        .join(" ");
}
function buildConicGradient(data, colors) {
    const total = data.reduce((sum, item) => sum + item.count, 0);
    if (total === 0)
        return "";
    let current = 0;
    const segments = data.map((item, index) => {
        const start = current;
        const end = current + (item.count / total) * 100;
        current = end;
        return `${colors[index % colors.length]} ${start}% ${end}%`;
    });
    return `conic-gradient(${segments.join(", ")})`;
}
function curvePath(x1, y1, x2, y2, curvature = 0.55) {
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    const dx = x2 - x1;
    const dy = y2 - y1;
    // perpendicular offset for the control point
    const offset = curvature * 0.25;
    const cx = mx - dy * offset;
    const cy = my + dx * offset;
    return `M${x1},${y1} Q${cx},${cy} ${x2},${y2}`;
}
function gradId(tone) {
    if (tone === "red")
        return "gradRed";
    if (tone === "orange")
        return "gradOrange";
    return "gradBlue";
}
function formatReason(value) {
    if (!value)
        return "No reason";
    return value.replace(/^OUTLIER::/i, "").replace(/_/g, " ").replace(/\s+/g, " ").trim();
}
function simplifyReason(reason) {
    const normalized = reason.toLowerCase();
    if (normalized.includes("duplicate"))
        return "Duplicate Claim";
    if (normalized.includes("vendor"))
        return "Fake Vendor";
    if (normalized.includes("gst"))
        return "GST Wrong";
    if (normalized.includes("payment"))
        return "Over Payment";
    if (normalized.includes("mismatch"))
        return "GST Wrong";
    return capitalize(reason);
}
function severityScore(row) {
    const score = Number(row?.ml_score ?? row?.reasons_json?.if_score ?? 0);
    const safeScore = Number.isFinite(score) ? score : 0;
    return row?.rule_flag ? Math.max(safeScore, 0.95) : safeScore;
}
function joinReasons(row) {
    const reasons = row?.reasons_json?.reason_list || [];
    if (!reasons.length)
        return "";
    return reasons.slice(0, 2).map((item) => simplifyReason(formatReason(item))).join(", ");
}
function resolvePayloadValue(payload, aliases) {
    for (const alias of aliases) {
        const directValue = payload[alias];
        if (directValue != null && String(directValue).trim()) {
            return String(directValue);
        }
        const dottedMatch = Object.entries(payload).find(([key, value]) => {
            const plainKey = key.split(".").pop()?.toLowerCase();
            return plainKey === alias.toLowerCase() && value != null && String(value).trim();
        });
        if (dottedMatch) {
            return String(dottedMatch[1]);
        }
    }
    return "";
}
function inferEntityType(payload) {
    if (resolvePayloadValue(payload, ["vendor_name", "vendor", "supplier_name"]))
        return "vendor";
    if (resolvePayloadValue(payload, ["office_name", "office", "branch"]))
        return "office";
    return "bill";
}
function formatIndianNumber(value) {
    return Number(value || 0).toLocaleString("en-IN");
}
function formatIndianCurrency(value) {
    return `₹ ${formatIndianNumber(value)}`;
}
function formatIndianCurrencyCompact(value) {
    const numeric = Number(value || 0);
    if (numeric >= 10000000)
        return `₹ ${(numeric / 10000000).toFixed(2)} Cr`;
    if (numeric >= 100000)
        return `₹ ${(numeric / 100000).toFixed(2)} L`;
    return formatIndianCurrency(numeric);
}
function parseDisplayDate(value) {
    if (!value || value === "Unavailable")
        return null;
    const normalized = String(value).trim();
    const parsed = new Date(normalized);
    if (!Number.isNaN(parsed.getTime())) {
        return parsed;
    }
    const dayFirstMatch = normalized.match(/^(\d{2})[/-](\d{2})[/-](\d{4})$/);
    if (dayFirstMatch) {
        const [, day, month, year] = dayFirstMatch;
        const fallback = new Date(`${year}-${month}-${day}T00:00:00`);
        if (!Number.isNaN(fallback.getTime())) {
            return fallback;
        }
    }
    return null;
}
function formatShortDate(date) {
    return date.toLocaleDateString("en-GB", { day: "2-digit", month: "short" });
}
function mapReviewStatus(value, feedback) {
    const normalizedFeedback = String(feedback || "").toLowerCase();
    if (normalizedFeedback === "accept")
        return "Approved";
    if (normalizedFeedback === "reject" || normalizedFeedback === "maybe")
        return "Under Review";
    const normalized = (value || "").toUpperCase();
    if (normalized === "REVIEWED")
        return "Under Review";
    return "Pending";
}
function formatPercent(value, total) {
    if (!total)
        return "0.00%";
    return `${((value / total) * 100).toFixed(2)}%`;
}
function capitalize(value) {
    return value.charAt(0).toUpperCase() + value.slice(1);
}
function truncateText(value, limit) {
    if (value.length <= limit)
        return value;
    return `${value.slice(0, limit - 1)}...`;
}
function riskScorePill(risk) {
    if (risk === "High")
        return riskHighPill;
    if (risk === "Medium")
        return riskMediumPill;
    return riskLowPill;
}
function statusPill(status) {
    if (status === "Pending")
        return statusNew;
    if (status === "Under Review")
        return statusReview;
    return statusApproved;
}
function getLayout(width) {
    if (width < 860) {
        return {
            contentColumns: "1fr",
            toolbarColumns: "1fr",
        };
    }
    if (width < 1260) {
        return {
            contentColumns: "280px minmax(0, 1fr)",
            toolbarColumns: "repeat(2, minmax(180px, 1fr))",
        };
    }
    return {
        contentColumns: "280px minmax(0, 1fr) 280px",
        toolbarColumns: "repeat(4, minmax(150px, 1fr))",
    };
}
/* ===================== Icons ===================== */
function CalendarIcon() {
    return (<svg width="14" height="14" viewBox="0 0 18 18" fill="none" aria-hidden="true">
      <rect x="2.25" y="3.5" width="13.5" height="12" rx="2" stroke="#5c6b87" strokeWidth="1.5"/>
      <path d="M5.5 2.25V5M12.5 2.25V5M2.25 7.25H15.75" stroke="#5c6b87" strokeWidth="1.5" strokeLinecap="round"/>
    </svg>);
}
function BellIcon() {
    return (<svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
      <path d="M11 3.5a4.5 4.5 0 0 0-4.5 4.5v2.2c0 .68-.24 1.34-.68 1.86L4.6 13.5c-.65.76-.11 1.92.88 1.92h11.04c.99 0 1.53-1.16.88-1.92l-1.22-1.44a2.88 2.88 0 0 1-.68-1.86V8A4.5 4.5 0 0 0 11 3.5Z" stroke="#11284d" strokeWidth="1.6"/>
      <path d="M8.8 17.3a2.46 2.46 0 0 0 4.4 0" stroke="#11284d" strokeWidth="1.6" strokeLinecap="round"/>
    </svg>);
}
function ChevronDownIcon() {
    return (<svg width="12" height="12" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <path d="M3 5.25 7 9l4-3.75" stroke="#233657" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>);
}
function DocumentIcon({ tone }) {
    return (<svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M7 3h8l4 4v14H7V3Z" stroke={tone} strokeWidth="1.6"/>
      <path d="M15 3v4h4M10 11h6M10 14h6M10 17h4" stroke={tone} strokeWidth="1.6" strokeLinecap="round"/>
    </svg>);
}
function AlertIcon({ tone }) {
    return (<svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M10.7 4.3c.6-1 2-1 2.6 0l8 13.9c.6 1-.1 2.3-1.3 2.3H4c-1.2 0-1.9-1.3-1.3-2.3l8-13.9Z" stroke={tone} strokeWidth="1.6"/>
      <path d="M12 9v5M12 16.5h.01" stroke={tone} strokeWidth="1.8" strokeLinecap="round"/>
    </svg>);
}
function RupeeIcon({ tone }) {
    return (<svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="9" stroke={tone} strokeWidth="1.6"/>
      <path d="M8 8.5h8M8 11.5h8M8 8.5c4 0 5.5 1.5 5.5 3.4 0 2-1.7 3.5-4.2 3.5H8l6.7 6.5" stroke={tone} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>);
}
function ShieldIcon({ tone }) {
    return (<svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 3 20 6v6.4c0 4.6-3 7.7-8 9.4-5-1.7-8-4.8-8-9.4V6l8-3Z" stroke={tone} strokeWidth="1.6"/>
      <path d="m9 12 2 2 4-4" stroke={tone} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>);
}
function SearchIcon() {
    return (<svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <circle cx="7" cy="7" r="4.8" stroke="#7b8aa6" strokeWidth="1.4"/>
      <path d="m10.6 10.6 3 3" stroke="#7b8aa6" strokeWidth="1.4" strokeLinecap="round"/>
    </svg>);
}
function FilterIcon() {
    return (<svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M2.5 3h11l-4.1 4.7v4l-2.8 1.3V7.7L2.5 3Z" stroke="#233657" strokeWidth="1.3" strokeLinejoin="round"/>
    </svg>);
}
function UserMiniIcon() {
    return (<svg width="11" height="11" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <circle cx="7" cy="4.4" r="2.3" fill="currentColor"/>
      <path d="M2.5 11.6c.84-2.2 2.4-3.4 4.5-3.4 2.1 0 3.66 1.2 4.5 3.4" fill="currentColor"/>
    </svg>);
}
function EyeIcon() {
    return (<svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M1.5 8s2.2-4 6.5-4 6.5 4 6.5 4-2.2 4-6.5 4S1.5 8 1.5 8Z" stroke="#22365a" strokeWidth="1.3"/>
      <circle cx="8" cy="8" r="2" stroke="#22365a" strokeWidth="1.3"/>
    </svg>);
}
function FileIcon() {
    return (<svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M4 2.5h5l3 3V13.5H4v-11Z" stroke="#22365a" strokeWidth="1.3"/>
      <path d="M9 2.5v3h3M6 8h4M6 10.5h4" stroke="#22365a" strokeWidth="1.3" strokeLinecap="round"/>
    </svg>);
}
function FlagIcon() {
    return (<svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M4 13.5v-11M4 3.5h7l-1.2 2L11 7.5H4" stroke="#22365a" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>);
}
/* ===================== Styles ===================== */
const pageStyles = `
  @keyframes tulipFloat {
    0% { transform: translate(-50%, -50%) translateY(0); }
    50% { transform: translate(-50%, -50%) translateY(-4px); }
    100% { transform: translate(-50%, -50%) translateY(0); }
  }
`;
const page = {
    display: "grid",
    gap: 14,
    color: "#10284a",
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
};
/* Toolbar */
const toolbar = {
    display: "grid",
    gridTemplateColumns: "280px minmax(0, 1fr) 220px",
    gap: 14,
    alignItems: "stretch",
    background: "#fff",
    border: "1px solid #e3ebf5",
    borderRadius: 16,
    padding: 8,
    boxShadow: "0 4px 18px rgba(16,40,74,0.05)",
};
const brandBlock = { display: "flex", alignItems: "center", gap: 12, padding: "6px 10px" };
const brandLogo = { width: 38, height: 38, display: "grid", placeItems: "center" };
const brandTitle = { fontSize: 19, fontWeight: 800, color: "#0f1f3d", lineHeight: 1, letterSpacing: "-0.01em" };
const brandSubtitle = { fontSize: 11, color: "#4f6485", marginTop: 4 };
const toolbarFilters = { display: "grid", gap: 10 };
const filterShell = {
    border: "1px solid #dde6f2",
    borderRadius: 10,
    padding: "6px 10px",
    background: "#fff",
    display: "grid",
    gap: 3,
};
const filterLabel = { fontSize: 11, fontWeight: 600, color: "#1f2e4d" };
const filterValueRow = {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
    fontSize: 13,
    color: "#233657",
};
const filterValueLeft = { display: "inline-flex", alignItems: "center", gap: 8, minWidth: 0 };
const selectInput = {
    width: "100%",
    border: "none",
    outline: "none",
    background: "transparent",
    color: "#233657",
    fontSize: 13,
    appearance: "none",
};
const dateRangeRow = {
    display: "flex",
    alignItems: "center",
    gap: 4,
    flexWrap: "wrap",
};
const dateRangeDivider = { fontSize: 11, color: "#5b6e8e", fontWeight: 600 };
const dateInput = {
    border: "none",
    outline: "none",
    background: "transparent",
    color: "#233657",
    fontSize: 12,
    minWidth: 108,
};
const tinyGhostButton = {
    border: "1px solid #dde6f2",
    borderRadius: 8,
    background: "#fff",
    color: "#233657",
    fontSize: 11,
    padding: "4px 8px",
    cursor: "pointer",
};
const profileWrap = {
    position: "relative",
    display: "flex",
    alignItems: "center",
    justifyContent: "flex-end",
    gap: 10,
    padding: "4px 10px",
};
const profileButton = {
    border: "none",
    background: "transparent",
    display: "flex",
    alignItems: "center",
    gap: 12,
    cursor: "pointer",
    padding: 0,
};
const bellWrap = { position: "relative", width: 30, height: 30, display: "grid", placeItems: "center" };
const bellBadge = {
    position: "absolute",
    top: -2,
    right: -2,
    minWidth: 16,
    height: 16,
    padding: "0 4px",
    borderRadius: 999,
    background: "#ff3040",
    color: "#fff",
    fontSize: 10,
    fontWeight: 700,
    display: "grid",
    placeItems: "center",
    border: "1.5px solid #fff",
};
const profileText = { display: "grid", gap: 2, lineHeight: 1.2 };
const profileName = { fontSize: 13, fontWeight: 600, color: "#11284d" };
const profileMeta = { fontSize: 11, color: "#5b6e8e" };
const profileMenu = {
    position: "absolute",
    top: "calc(100% + 8px)",
    right: 0,
    width: 220,
    background: "#fff",
    border: "1px solid #e3ebf5",
    borderRadius: 14,
    boxShadow: "0 12px 30px rgba(16,40,74,0.14)",
    padding: 12,
    display: "grid",
    gap: 10,
    zIndex: 20,
};
const profileMenuTitle = { fontSize: 12, fontWeight: 800, color: "#0f1f3d" };
const profileMenuRow = {
    display: "flex",
    justifyContent: "space-between",
    gap: 10,
    fontSize: 12,
    color: "#415776",
};
/* Grid */
const contentGrid = { display: "grid", gap: 14, alignItems: "start" };
const railColumn = { display: "grid", gap: 14, alignSelf: "start" };
const centerColumn = { display: "grid", gap: 14 };
/* Cards */
const panelCard = {
    background: "#fff",
    border: "1px solid #e3ebf5",
    borderRadius: 14,
    boxShadow: "0 4px 16px rgba(16,40,74,0.04)",
    padding: 14,
    display: "grid",
    gap: 12,
};
const panelHeading = {
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: "0.06em",
    color: "#4f6485",
};
/* Stats */
const summaryStack = { display: "grid", gap: 10 };
const statCard = {
    display: "grid",
    gridTemplateColumns: "44px 1fr",
    gap: 10,
    alignItems: "center",
    border: "1px solid #ecf1f8",
    borderRadius: 12,
    padding: "10px 12px",
    background: "#fdfdff",
};
const statIconWrap = {
    width: 40,
    height: 40,
    borderRadius: 10,
    display: "grid",
    placeItems: "center",
};
const statLabel = { fontSize: 11, color: "#5b6e8e", lineHeight: 1.2 };
const statValue = { fontSize: 20, fontWeight: 700, lineHeight: 1.1, marginTop: 2, letterSpacing: "-0.01em" };
const statSubtext = { fontSize: 10, color: "#6b7d9c", marginTop: 2 };
/* Donut */
const donutColumn = {
    display: "grid",
    justifyItems: "center",
    gap: 14,
};
const donut = {
    width: 132,
    height: 132,
    borderRadius: "50%",
    display: "grid",
    placeItems: "center",
    position: "relative",
};
const donutCore = {
    position: "relative",
    zIndex: 2,
    width: 88,
    height: 88,
    borderRadius: "50%",
    background: "#fff",
    boxShadow: "inset 0 0 0 1px #eef2f9",
    display: "grid",
    placeItems: "center",
    textAlign: "center",
};
const donutLegendStack = {
    display: "grid",
    gap: 9,
    width: "100%",
    maxHeight: 210,
    overflowY: "auto",
    paddingRight: 4,
};
const donutCoreValue = { fontSize: 18, fontWeight: 800, color: "#0f1f3d" };
const donutCoreLabel = { fontSize: 11, color: "#5b6e8e" };
const legendLine = {
    display: "grid",
    gridTemplateColumns: "10px minmax(0, 1fr) auto",
    gap: 10,
    alignItems: "center",
    fontSize: 12,
    color: "#233657",
};
const legendDot = { width: 10, height: 10, borderRadius: 3 };
const legendLineLabel = { minWidth: 0, lineHeight: 1.35, whiteSpace: "normal", overflowWrap: "anywhere" };
const legendLineValue = { color: "#2f4368", fontWeight: 700, whiteSpace: "nowrap", fontVariantNumeric: "tabular-nums" };
/* Trend */
const trendShell = { display: "grid", gap: 6 };
const trendSvg = { width: "100%", height: 120, background: "#fcfdff", borderRadius: 8 };
const trendAxis = {
    display: "grid",
    gridTemplateColumns: "repeat(5, 1fr)",
    gap: 2,
    fontSize: 10,
    color: "#5b6e8e",
};
const trendLegend = { display: "grid", gap: 5, marginTop: 6 };
const trendLegendRow = { display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "#233657" };
const trendSwatch = { width: 14, height: 2, borderRadius: 2 };
/* Network */
const networkHead = {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 12,
    flexWrap: "wrap",
};
const networkControls = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" };
const ctrlButton = {
    height: 30,
    padding: "0 10px",
    border: "1px solid #dde6f2",
    borderRadius: 8,
    background: "#fff",
    color: "#233657",
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    fontSize: 12,
    cursor: "pointer",
};
const inlineSelectWrap = {
    height: 30,
    minWidth: 148,
    border: "1px solid #dde6f2",
    borderRadius: 8,
    background: "#fff",
    padding: "0 8px",
    display: "flex",
    alignItems: "center",
};
const inlineSelectInput = {
    width: "100%",
    border: "none",
    outline: "none",
    background: "transparent",
    color: "#233657",
    fontSize: 12,
};
const toggleRow = {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    border: "none",
    background: "transparent",
    color: "#233657",
    fontSize: 12,
    cursor: "pointer",
};
const toggleTrack = {
    width: 30,
    height: 16,
    borderRadius: 999,
    padding: 2,
    display: "flex",
    transition: "background 0.2s",
};
const toggleThumb = { width: 12, height: 12, borderRadius: 999, background: "#fff" };
const networkCanvas = {
    position: "relative",
    height: 480,
    borderRadius: 14,
    overflow: "hidden",
    background: "radial-gradient(ellipse at 50% 55%, rgba(255,160,40,0.10) 0%, rgba(255,255,255,0) 28%), linear-gradient(180deg, #ffffff 0%, #fbfdff 100%)",
    border: "1px solid #eef2f9",
};
const gridFloor = {
    position: "absolute",
    inset: 0,
    backgroundImage: "linear-gradient(rgba(160,180,210,0.30) 1px, transparent 1px), linear-gradient(90deg, rgba(160,180,210,0.30) 1px, transparent 1px)",
    backgroundSize: "48px 48px",
    transform: "perspective(900px) rotateX(70deg) scale(1.5) translateY(80px)",
    transformOrigin: "center bottom",
    opacity: 0.55,
    WebkitMaskImage: "linear-gradient(180deg, transparent 0%, #000 25%, #000 100%)",
    maskImage: "linear-gradient(180deg, transparent 0%, #000 25%, #000 100%)",
};
const netSvg = { position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" };
/* Bulb nodes */
const nodeWrap = {
    position: "absolute",
    transform: "translate(-50%, -50%)",
    zIndex: 5,
    display: "grid",
    justifyItems: "center",
    gap: 4,
};
const bulbBase = {
    borderRadius: "50%",
    display: "grid",
    placeItems: "center",
    position: "relative",
    isolation: "isolate",
};
const bulbHighlight = {
    position: "absolute",
    top: "8%",
    left: "18%",
    width: "38%",
    height: "32%",
    borderRadius: "50%",
    background: "radial-gradient(circle at 30% 30%, rgba(255,255,255,0.9), rgba(255,255,255,0) 65%)",
    filter: "blur(1px)",
    pointerEvents: "none",
    zIndex: 2,
};
const bulbRed = {
    background: "radial-gradient(circle at 32% 28%, #ffc7cb 0%, #ff7780 22%, #f33741 60%, #c8121b 100%)",
    boxShadow: "0 12px 22px rgba(220,30,40,0.32), inset 0 -6px 14px rgba(120,0,10,0.4), inset 0 4px 8px rgba(255,255,255,0.35)",
};
const bulbOrange = {
    background: "radial-gradient(circle at 32% 28%, #ffe2b8 0%, #ffba66 22%, #ff9520 60%, #d56a00 100%)",
    boxShadow: "0 12px 22px rgba(220,120,0,0.32), inset 0 -6px 14px rgba(140,55,0,0.4), inset 0 4px 8px rgba(255,255,255,0.35)",
};
const bulbBlue = {
    background: "radial-gradient(circle at 32% 28%, #cee6ff 0%, #74b6ff 22%, #2b8ef9 60%, #1561c2 100%)",
    boxShadow: "0 12px 22px rgba(20,90,200,0.32), inset 0 -6px 14px rgba(10,40,100,0.4), inset 0 4px 8px rgba(255,255,255,0.35)",
};
const nodeLabelBlock = {
    marginTop: 4,
    textAlign: "center",
    color: "#11284d",
    fontSize: 11,
    lineHeight: 1.25,
    maxWidth: 130,
};
const nodeLabelKind = { fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", color: "#2c405f" };
const nodeLabelName = { fontSize: 11, fontWeight: 600, color: "#11284d" };
const nodeLabelMeta = { fontSize: 11, color: "#2c405f", marginTop: 1 };
const approverLabel = {
    position: "absolute",
    transform: "translate(-50%, 8px)",
    width: 80,
    textAlign: "center",
    color: "#2c405f",
    fontSize: 9,
    lineHeight: 1.2,
    pointerEvents: "none",
    zIndex: 4,
};
const approverLabelKind = { fontWeight: 700, letterSpacing: "0.06em" };
const centerCard = {
    position: "absolute",
    left: "50%",
    top: "calc(50% + 56px)",
    transform: "translateX(-50%)",
    minWidth: 168,
    padding: "8px 14px 10px",
    borderRadius: 10,
    background: "#fff",
    border: "1px solid #e3ebf5",
    boxShadow: "0 10px 24px rgba(16,40,74,0.14)",
    textAlign: "center",
    zIndex: 6,
};
const centerCardKind = { fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", color: "#11284d" };
const centerCardName = { fontSize: 12, color: "#233657", marginTop: 4 };
const centerCardScore = { fontSize: 11, color: "#f33741", fontWeight: 700, marginTop: 2 };
/* Network footer */
const netFoot = { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 };
const legendBox = {
    border: "1px solid #e3ebf5",
    borderRadius: 10,
    padding: "8px 12px",
    display: "flex",
    gap: 12,
    alignItems: "center",
    flexWrap: "wrap",
};
const legendBoxLabel = { fontSize: 11, fontWeight: 700, color: "#4f6485" };
const legendChip = { display: "inline-flex", alignItems: "center", gap: 6, fontSize: 11, color: "#233657" };
const legendChipDot = { width: 10, height: 10, borderRadius: 2 };
const entityChip = { display: "inline-flex", alignItems: "center", gap: 6, fontSize: 11, color: "#233657" };
const entityChipRing = {
    width: 16,
    height: 16,
    borderRadius: 999,
    border: "1.5px solid",
    display: "grid",
    placeItems: "center",
};
/* Table */
const tableHeadBar = {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 12,
    flexWrap: "wrap",
};
const tableTitle = { fontSize: 13, fontWeight: 700, color: "#0f1f3d", letterSpacing: "0.04em" };
const tableActions = { display: "flex", gap: 10, alignItems: "center" };
const searchBox = {
    width: 240,
    height: 28,
    padding: "0 10px",
    border: "1px solid #dde6f2",
    borderRadius: 8,
    display: "flex",
    alignItems: "center",
    gap: 6,
    color: "#8896ad",
    fontSize: 11,
};
const searchInput = {
    width: "100%",
    border: "none",
    outline: "none",
    background: "transparent",
    color: "#233657",
    fontSize: 11,
};
const filterBtn = {
    height: 28,
    padding: "0 10px",
    border: "1px solid #dde6f2",
    borderRadius: 8,
    background: "#fff",
    color: "#233657",
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    fontSize: 11,
    cursor: "pointer",
};
const filterMenuWrap = { position: "relative" };
const filterMenu = {
    position: "absolute",
    top: "calc(100% + 8px)",
    right: 0,
    width: 250,
    background: "#fff",
    border: "1px solid #e3ebf5",
    borderRadius: 14,
    boxShadow: "0 12px 30px rgba(16,40,74,0.14)",
    padding: 12,
    display: "grid",
    gap: 10,
    zIndex: 20,
};
const filterMenuGrid = { display: "grid", gap: 10 };
const clearFilterButton = {
    border: "1px solid #dde6f2",
    borderRadius: 10,
    background: "#f8fbff",
    color: "#22365a",
    fontSize: 12,
    fontWeight: 700,
    padding: "10px 12px",
    cursor: "pointer",
};
const scrollingTableWrapper = {
    overflowX: "auto",
    overflowY: "auto",
    maxHeight: 420,
    border: "1px solid #eef2f9",
    borderRadius: 12,
};
const reportTable = { width: "100%", borderCollapse: "collapse", minWidth: 900 };
const tableHeadCell = {
    textAlign: "left",
    padding: "10px 14px",
    fontSize: 11,
    fontWeight: 600,
    color: "#4f6485",
    background: "#f7faff",
    borderTop: "1px solid #ecf1f8",
    borderBottom: "1px solid #ecf1f8",
};
const tableBodyRow = { borderBottom: "1px solid #f1f5fb" };
const tableBodyCell = { padding: "12px 14px", fontSize: 12, color: "#233657", verticalAlign: "middle" };
const emptyTableCell = { padding: 24, textAlign: "center", color: "#64748b" };
const errorBanner = {
    borderRadius: 18,
    border: "1px solid #fecaca",
    background: "#fef2f2",
    color: "#b91c1c",
    padding: "14px 16px",
    boxShadow: "0 8px 18px rgba(148, 163, 184, 0.12)",
};
const scrollHighlightsWrap = {
    maxHeight: 360,
    overflowY: "auto",
    paddingRight: 4,
};
const riskHighPill = {
    display: "inline-flex",
    alignItems: "center",
    gap: 4,
    padding: "3px 8px",
    borderRadius: 6,
    background: "#fff0f1",
    color: "#f33741",
    fontWeight: 700,
    fontSize: 11,
};
const riskMediumPill = { ...riskHighPill, background: "#fff4e8", color: "#ea7d00" };
const riskLowPill = { ...riskHighPill, background: "#edf9ef", color: "#2f9f45" };
const statusNew = {
    display: "inline-flex",
    padding: "3px 8px",
    borderRadius: 4,
    background: "#fff0f1",
    color: "#f33741",
    fontSize: 11,
};
const statusReview = { ...statusNew, background: "#fff4e8", color: "#ea7d00" };
const statusApproved = { ...statusNew, background: "#edf9ef", color: "#2f9f45" };
const actionRow = { display: "flex", gap: 6 };
const actionButton = {
    width: 26,
    height: 26,
    borderRadius: 6,
    border: "1px solid #e1e8f3",
    background: "#fff",
    display: "grid",
    placeItems: "center",
    cursor: "pointer",
};
/* Highlights */
const highlightsStack = { display: "grid", gap: 8 };
const highlightCard = {
    display: "grid",
    gridTemplateColumns: "28px 1fr",
    gap: 10,
    alignItems: "start",
    border: "1px solid #ecf1f8",
    borderRadius: 10,
    padding: "10px 12px",
    background: "#fdfdff",
};
const highlightIconWrap = { width: 24, height: 24, display: "grid", placeItems: "center", marginTop: 2 };
const highlightTitle = { fontSize: 12, fontWeight: 700, lineHeight: 1.2 };
const highlightDesc = { fontSize: 11, color: "#233657", marginTop: 4, lineHeight: 1.4 };
const highlightNote = { fontSize: 11, color: "#4f6485", marginTop: 4 };
const viewAllButton = {
    height: 32,
    borderRadius: 8,
    background: "#eaf3ff",
    color: "#1668d8",
    border: "1px solid #c6dffb",
    fontWeight: 600,
    fontSize: 12,
    cursor: "pointer",
};
const emptyBox = {
    borderRadius: 10,
    border: "1px dashed #dbe5f1",
    padding: 14,
    color: "#607393",
    fontSize: 12,
};
/* Run context */
const contextStack = { display: "grid", gap: 8 };
const contextRow = {
    display: "flex",
    justifyContent: "space-between",
    gap: 10,
    paddingBottom: 8,
    borderBottom: "1px solid #eef2f8",
};
const contextLabel = { fontSize: 12, color: "#607393" };
const contextValue = { fontSize: 12, color: "#152a54", fontWeight: 600, textAlign: "right" };
const tableTags = { display: "flex", flexWrap: "wrap", gap: 8 };
const contextTag = {
    padding: "5px 10px",
    borderRadius: 999,
    background: "#f2f7ff",
    color: "#145ed1",
    fontSize: 11,
    border: "1px solid #cfe0fa",
};
const emptyTagText = { fontSize: 12, color: "#607393" };
