export function displayValue(value) {
  if (value === null || value === undefined || value === "") {
    return "NA";
  }
  return String(value);
}

export function hydrateLatestDataset(latestWorkbenchRun) {
  if (!latestWorkbenchRun?.datasetTable) {
    return null;
  }

  const latestSelectedTablesKey = (latestWorkbenchRun?.selectedTables || []).join("\u001f");
  return {
    dataset_table: latestWorkbenchRun.datasetTable,
    run_id: latestWorkbenchRun?.runId ?? null,
    selected_tables: latestSelectedTablesKey ? latestSelectedTablesKey.split("\u001f") : [],
    run_name: latestWorkbenchRun?.runName || "Latest workbench run",
    total_rows: latestWorkbenchRun?.totalRows || 0,
    final_anomaly_count: latestWorkbenchRun?.finalAnomalyCount || 0,
  };
}
