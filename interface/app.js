const METRIC_ORDER = ["Output Relevancy", "Completeness"];

const state = {
  examples: [],
  filteredExamples: [],
  selectedKey: null,
  fileName: "",
  warnings: []
};

const refs = {
  fileInput: document.getElementById("fileInput"),
  orchFilter: document.getElementById("orchFilter"),
  intendedFilter: document.getElementById("intendedFilter"),
  relevancyScoreFilter: document.getElementById("relevancyScoreFilter"),
  completenessScoreFilter: document.getElementById("completenessScoreFilter"),
  rowSelect: document.getElementById("rowSelect"),
  rowCountNote: document.getElementById("rowCountNote"),
  fileStatus: document.getElementById("fileStatus"),
  warningBox: document.getElementById("warningBox"),
  selectedMeta: document.getElementById("selectedMeta"),
  queryBlock: document.getElementById("queryBlock"),
  inputBlock: document.getElementById("inputBlock"),
  routeGrid: document.getElementById("routeGrid"),
  relevancyBlock: document.getElementById("relevancyBlock"),
  completenessBlock: document.getElementById("completenessBlock"),
  scoreStrip: document.getElementById("scoreStrip"),
  overallNotesBlock: document.getElementById("overallNotesBlock")
};

refs.fileInput.addEventListener("change", async (event) => {
  const [file] = event.target.files || [];
  if (!file) {
    return;
  }

  try {
    const text = await file.text();
    loadData(text, file.name);
  } catch (error) {
    showFatal(`Could not read file: ${error.message}`);
  }
});

refs.orchFilter.addEventListener("change", () => {
  filterExamples();
  renderStatus();
  renderFilterControls();
  renderSelectedExample();
});

refs.intendedFilter.addEventListener("change", () => {
  filterExamples();
  renderStatus();
  renderFilterControls();
  renderSelectedExample();
});

refs.relevancyScoreFilter.addEventListener("change", () => {
  filterExamples();
  renderStatus();
  renderFilterControls();
  renderSelectedExample();
});

refs.completenessScoreFilter.addEventListener("change", () => {
  filterExamples();
  renderStatus();
  renderFilterControls();
  renderSelectedExample();
});

refs.rowSelect.addEventListener("change", () => {
  state.selectedKey = refs.rowSelect.value || null;
  renderSelectedExample();
});

initialize();

function initialize() {
  renderFilterControls();
  renderSelectedExample();
}

function loadData(text, fileName) {
  const parsed = parseRecords(text, fileName);
  state.examples = buildExamples(parsed.rows);
  state.fileName = fileName;
  state.warnings = parsed.warnings;

  filterExamples();
  renderStatus();
  renderWarnings();
  renderFilterControls();
  renderSelectedExample();
}

function parseRecords(text, fileName) {
  const trimmed = text.trim();
  if (!trimmed) {
    throw new Error("The selected file is empty.");
  }

  const warnings = [];
  const extension = (fileName.split(".").pop() || "").toLowerCase();

  if (extension === "jsonl") {
    return { rows: filterUnsupportedRows(parseJsonLines(trimmed, warnings), warnings), warnings };
  }

  try {
    return { rows: filterUnsupportedRows(normalizeParsedJson(parseJsonWithRelaxedNumbers(trimmed)), warnings), warnings };
  } catch (jsonError) {
    try {
      return { rows: filterUnsupportedRows(parseJsonLines(trimmed, warnings), warnings), warnings };
    } catch {
      throw new Error(`Could not parse file as JSON or JSONL. ${jsonError.message}`);
    }
  }
}

function parseJsonLines(text, warnings) {
  const rows = [];
  const lines = text.split(/\r?\n/);

  lines.forEach((line, index) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }

    try {
      rows.push(parseJsonWithRelaxedNumbers(trimmed));
    } catch (error) {
      warnings.push(`Skipped line ${index + 1}: ${error.message}`);
    }
  });

  if (!rows.length) {
    throw new Error("No valid JSON objects were found.");
  }

  return rows;
}

function parseJsonWithRelaxedNumbers(text) {
  try {
    return JSON.parse(text);
  } catch (error) {
    const normalized = normalizeNonStandardJsonNumbers(text);
    if (normalized === text) {
      throw error;
    }

    return JSON.parse(normalized);
  }
}

function normalizeNonStandardJsonNumbers(text) {
  return text
    .replace(/(:\s*)NaN(?=\s*[,}\]])/g, "$1null")
    .replace(/(:\s*)Infinity(?=\s*[,}\]])/g, "$1null")
    .replace(/(:\s*)-Infinity(?=\s*[,}\]])/g, "$1null");
}

function normalizeParsedJson(parsed) {
  if (Array.isArray(parsed)) {
    return parsed;
  }

  if (Array.isArray(parsed.rows)) {
    return parsed.rows;
  }

  if (Array.isArray(parsed.examples)) {
    return parsed.examples;
  }

  return [parsed];
}

function filterUnsupportedRows(rows, warnings) {
  const filteredRows = rows.filter((row) => !shouldSkipRow(row));
  const skippedCount = rows.length - filteredRows.length;

  if (!filteredRows.length) {
    throw new Error("No supported rows were found after skipping rows where route_orch is ERROR.");
  }

  return filteredRows;
}

function shouldSkipRow(row) {
  return normalizeRouteValue(firstText(row?.route_orch, "")) === "ERROR";
}

function buildExamples(rows) {
  return rows
    .map((row, index) => finalizeExample(row, index))
    .sort((left, right) => compareExampleOrder(left, right));
}

function finalizeExample(row, index) {
  return {
    key: buildExampleKey(row, index),
    rows: [row],
    representative: row,
    displayId: firstText(row.row_id, row.row_id_previous_folder, row.example_id, String(index + 1)),
    routeOrchValue: normalizeRouteValue(firstText(row.route_orch, getRouteName(row))),
    routeIntendedValue: normalizeRouteValue(firstText(row.route_intended, getRouteName(row))),
    outputRelevancyScore: firstText(row.eval_output_relevancy_score, ""),
    completenessScore: firstText(row.eval_completeness_score, ""),
    query: firstText(row.query, ""),
    inputPreview: firstText(row.input_preview, row.input, "")
  };
}

function buildExampleKey(row, index) {
  return [
    firstText(row.folder_source, "unknown-folder"),
    firstText(row.dataset_source, "unknown-dataset"),
    firstText(row.row_id, row.row_id_previous_folder, row.example_id, String(index + 1)),
    String(index)
  ].join("::");
}

function compareExampleOrder(left, right) {
  const leftNumber = Number(left.representative.row_id ?? left.representative.row_id_previous_folder);
  const rightNumber = Number(right.representative.row_id ?? right.representative.row_id_previous_folder);

  if (!Number.isNaN(leftNumber) && !Number.isNaN(rightNumber)) {
    return leftNumber - rightNumber;
  }

  return String(left.displayId).localeCompare(String(right.displayId), undefined, { numeric: true });
}

function filterExamples() {
  const orchFilter = refs.orchFilter.value || "ALL";
  const intendedFilter = refs.intendedFilter.value || "ALL";
  const relevancyScoreFilter = refs.relevancyScoreFilter.value || "ALL";
  const completenessScoreFilter = refs.completenessScoreFilter.value || "ALL";

  state.filteredExamples = state.examples.filter((example) => {
    const orchMatch = orchFilter === "ALL" || example.routeOrchValue === orchFilter;
    const intendedMatch = intendedFilter === "ALL" || example.routeIntendedValue === intendedFilter;
    const relevancyMatch = relevancyScoreFilter === "ALL" || example.outputRelevancyScore === relevancyScoreFilter;
    const completenessMatch = completenessScoreFilter === "ALL" || example.completenessScore === completenessScoreFilter;
    return orchMatch && intendedMatch && relevancyMatch && completenessMatch;
  });

  if (!state.filteredExamples.some((example) => example.key === state.selectedKey)) {
    state.selectedKey = state.filteredExamples[0]?.key || null;
  }
}

function renderStatus() {
  if (!state.examples.length) {
    refs.fileStatus.textContent = "No file loaded yet.";
    return;
  }

  refs.fileStatus.textContent = `${state.fileName} loaded - ${state.filteredExamples.length} visible row${state.filteredExamples.length === 1 ? "" : "s"} out of ${state.examples.length}.`;
}

function renderWarnings() {
  if (!state.warnings.length) {
    refs.warningBox.classList.add("hidden");
    refs.warningBox.textContent = "";
    return;
  }

  refs.warningBox.classList.remove("hidden");
  refs.warningBox.textContent = state.warnings.slice(0, 6).join(" ");
}

function renderFilterControls() {
  renderRouteFilter(refs.orchFilter, state.examples.map((example) => example.routeOrchValue), "All orchestration routes");
  renderRouteFilter(refs.intendedFilter, state.examples.map((example) => example.routeIntendedValue), "All intended routes");
  renderRouteFilter(refs.relevancyScoreFilter, state.examples.map((example) => example.outputRelevancyScore), "All output relevancy scores");
  renderRouteFilter(refs.completenessScoreFilter, state.examples.map((example) => example.completenessScore), "All completeness scores");
  refs.rowCountNote.textContent = `${state.filteredExamples.length} row${state.filteredExamples.length === 1 ? "" : "s"} available`;

  refs.rowSelect.textContent = "";
  if (!state.filteredExamples.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No rows match current filters";
    refs.rowSelect.appendChild(option);
    refs.rowSelect.value = "";
    return;
  }

  state.filteredExamples.forEach((example) => {
    const option = document.createElement("option");
    option.value = example.key;
    option.textContent = buildRowOptionLabel(example);
    refs.rowSelect.appendChild(option);
  });

  refs.rowSelect.value = state.selectedKey || state.filteredExamples[0].key;
}

function renderRouteFilter(select, values, allLabel) {
  const currentValue = select.value || "ALL";
  const options = ["ALL", ...uniqueValues(values)].filter(Boolean);

  select.textContent = "";
  options.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value === "ALL" ? allLabel : value;
    select.appendChild(option);
  });

  select.value = options.includes(currentValue) ? currentValue : "ALL";
}

function buildRowOptionLabel(example) {
  const orch = firstText(example.routeOrchValue, "-");
  const intended = firstText(example.routeIntendedValue, "-");
  return `${example.displayId} | orch ${orch} | intended ${intended}`;
}

function renderSelectedExample() {
  const example = state.examples.find((item) => item.key === state.selectedKey) || null;
  if (!example) {
    renderNoSelection();
    return;
  }

  renderUserRequest(example);
  renderRoutes(example);
  renderScoring(example);
  renderOverallSummary(example);
}

function renderNoSelection() {
  refs.selectedMeta.textContent = "";
  setFormattedContent(refs.queryBlock, "Select a file and choose a row to inspect.");
  setFormattedContent(refs.inputBlock, "");
  refs.routeGrid.textContent = "";
  refs.routeGrid.appendChild(buildPlaceholder("Route outputs will appear here."));
  refs.relevancyBlock.textContent = "";
  refs.completenessBlock.textContent = "";
  refs.relevancyBlock.appendChild(buildPlaceholder("Output Relevancy details will appear here."));
  refs.completenessBlock.appendChild(buildPlaceholder("Completeness details will appear here."));
  refs.scoreStrip.textContent = "";
  setFormattedContent(refs.overallNotesBlock, "No evaluation summary is available yet.");
}

function renderUserRequest(example) {
  const row = example.representative;
  refs.selectedMeta.textContent = "";
  refs.selectedMeta.appendChild(buildPill(`row ${example.displayId}`, "example-id"));

  if (example.routeOrchValue) {
    refs.selectedMeta.appendChild(buildPill(`orch ${example.routeOrchValue}`, "pill"));
  }

  if (example.routeIntendedValue) {
    refs.selectedMeta.appendChild(buildPill(`intended ${example.routeIntendedValue}`, "pill"));
  }

  setFormattedContent(refs.queryBlock, firstText(row.query, "No query available."));
  setFormattedContent(refs.inputBlock, firstText(row.input, row.input_preview, "No input available."));
}

function renderRoutes(example) {
  refs.routeGrid.textContent = "";
  refs.routeGrid.appendChild(buildRouteCard(example.representative));
}

function buildRouteCard(row) {
  const route = normalizeRouteValue(firstText(getRouteName(row), row?.route_orch, row?.route_intended, "UNKNOWN"));
  const routeData = extractRouteData(row);

  const card = document.createElement("section");
  card.className = "route-card";

  const header = document.createElement("div");
  header.className = "route-card-head";

  const topLine = document.createElement("div");
  topLine.className = "route-card-top";
  topLine.appendChild(buildPill(route, "example-id"));
  topLine.appendChild(buildPill(routeData.routeLabel || route, "pill"));

  const title = document.createElement("h3");
  title.textContent = routeLabel(route);

  header.append(topLine, title);

  const body = document.createElement("div");
  body.className = "route-card-body";

  if (!routeData.available) {
    body.appendChild(buildMutedBlock("No data for this route is present in the selected row."));
    card.append(header, body);
    return card;
  }

  if (routeData.responseText) {
    const responseBlock = document.createElement("div");
    responseBlock.className = "route-copy";
    setFormattedContent(responseBlock, routeData.responseText);
    body.appendChild(responseBlock);
  }

  if (routeData.suggestions.length) {
    routeData.suggestions.forEach((suggestion, index) => {
      body.appendChild(buildSuggestionCard(suggestion, index + 1));
    });
  }

  if (!routeData.responseText && !routeData.suggestions.length) {
    body.appendChild(buildMutedBlock("This route has no response or suggestion payload in the selected row."));
  }

  card.append(header, body);
  return card;
}

function buildSuggestionCard(suggestion, index) {
  const card = document.createElement("article");
  card.className = "item-card";

  const title = document.createElement("h4");
  title.textContent = `${suggestion.explanation || "Suggestion"} (${index})`;

  const transformed = document.createElement("div");
  transformed.className = "small-copy rich-content";
  setFormattedContent(
    transformed,
    firstText(suggestion.transformed_text, suggestion.response, suggestion.revision, "No transformed text available.")
  );

  card.append(title, transformed);
  return card;
}

function renderScoring(example) {
  const evalRow = pickEvaluationRow(example);
  const rubrics = getRubrics(evalRow);
  const verdicts = getVerdicts(evalRow);
  const rubricGroups = rubrics.length ? buildRubricGroups(rubrics) : [];

  refs.relevancyBlock.textContent = "";
  refs.completenessBlock.textContent = "";

  if (!rubricGroups.length) {
    refs.relevancyBlock.appendChild(buildPlaceholder("No Output Relevancy payload is available in the selected row."));
    refs.completenessBlock.appendChild(buildPlaceholder("No Completeness payload is available in the selected row."));
    return;
  }

  const groupsByMetric = new Map(rubricGroups.map((group) => [normalizeComparisonText(group.name), group]));
  renderScoringMetricColumn(refs.relevancyBlock, groupsByMetric.get(normalizeComparisonText("Output Relevancy")), verdicts, "Output Relevancy");
  renderScoringMetricColumn(refs.completenessBlock, groupsByMetric.get(normalizeComparisonText("Completeness")), verdicts, "Completeness");
}

function renderScoringMetricColumn(container, group, verdicts, label) {
  container.textContent = "";
  if (!group) {
    container.appendChild(buildPlaceholder(`No ${label} payload is available in the selected row.`));
    return;
  }

  const stack = document.createElement("div");
  stack.className = "metric-stack";
  stack.appendChild(buildScoringMetricCard(group, verdicts));
  container.appendChild(stack);
}

function buildRubricGroups(rubrics) {
  const groups = new Map();

  rubrics.forEach((rubric) => {
    const metricName = getMetricName(rubric);
    if (!groups.has(metricName)) {
      groups.set(metricName, {
        name: metricName,
        description: firstText(rubric.description, ""),
        importance: firstText(rubric.metric_importance, rubric.importance, rubric.weight, "Unspecified"),
        maxScore: null,
        items: []
      });
    }

    const group = groups.get(metricName);
    group.description = firstText(group.description, rubric.description, "");
    group.importance = firstText(group.importance, rubric.metric_importance, "Unspecified");

    const rubricMaxScore = getMaxScoreFromLevels(rubric.levels);
    if (rubricMaxScore !== null) {
      group.maxScore = group.maxScore === null ? rubricMaxScore : Math.max(group.maxScore, rubricMaxScore);
    }

    const evaluationItems = ensureArray(rubric.evaluation_items);
    evaluationItems.forEach((item) => {
      const itemName = firstText(item.item, "Unnamed item");
      group.items.push({
        rawName: itemName,
        criterion: cleanupItemLabel(itemName),
        description: firstText(item.description, ""),
        weight: firstText(item.importance, rubric.metric_importance, "Unspecified")
      });
    });
  });

  const orderedGroups = Array.from(groups.values()).sort((left, right) => {
    const leftIndex = METRIC_ORDER.indexOf(left.name);
    const rightIndex = METRIC_ORDER.indexOf(right.name);

    if (leftIndex !== -1 || rightIndex !== -1) {
      return (leftIndex === -1 ? Number.MAX_SAFE_INTEGER : leftIndex) - (rightIndex === -1 ? Number.MAX_SAFE_INTEGER : rightIndex);
    }

    return left.name.localeCompare(right.name);
  });

  orderedGroups.forEach((group) => {
    const prefix = metricPrefix(group.name);
    group.description = firstText(group.description, inferMetricDescription(group.name, group.items));
    group.items = group.items.map((item, index) => ({
      ...item,
      key: normalizeComparisonText(firstText(item.rawName, item.criterion)),
      label: `${prefix}-${index + 1}`
    }));
  });

  return orderedGroups;
}

function buildScoringMetricCard(group, verdicts) {
  const card = document.createElement("article");
  card.className = "score-card";

  const metricVerdict = matchMetricVerdict(group, verdicts);
  const verdictItems = group.items.map((item) => ({
    item,
    verdict: matchVerdict(item, verdicts, metricVerdict)
  }));

  const scores = verdictItems
    .map((entry) => Number(entry.verdict?.score))
    .filter((score) => !Number.isNaN(score));
  const metricScore = Number(metricVerdict?.score);
  const averageScore = !Number.isNaN(metricScore)
    ? metricScore
    : scores.length
      ? scores.reduce((sum, score) => sum + score, 0) / scores.length
      : null;
  const summaryText = firstText(metricVerdict?.reasoning, summarizeVerdicts(verdictItems.map((entry) => entry.verdict).filter(Boolean)), "No metric-level reasoning is available for this metric.");
  const scoreLabel = formatScoreValue(averageScore, group.maxScore);

  const definition = document.createElement("p");
  definition.className = "metric-definition";
  definition.textContent = group.description;

  const metricMeta = document.createElement("div");
  metricMeta.className = "metric-meta";

  const metricMetaRow = document.createElement("div");
  metricMetaRow.className = "metric-meta-row";
  metricMetaRow.appendChild(buildPill(scoreLabel === null ? "score n/a" : `score ${scoreLabel}`, "score-chip"));
  if (group.maxScore !== null) {
    metricMetaRow.appendChild(buildPill(`max ${group.maxScore}`, "pill"));
  }

  const metricReasoning = document.createElement("p");
  metricReasoning.className = "metric-reasoning";
  metricReasoning.textContent = summaryText;

  metricMeta.append(metricMetaRow, metricReasoning);

  const itemGrid = document.createElement("div");
  itemGrid.className = "item-grid compact";

  verdictItems.forEach(({ item, verdict }) => {
    const itemCard = document.createElement("article");
    itemCard.className = "item-card";

    const title = document.createElement("h4");
    title.textContent = `${item.label} ${item.criterion}`;

    const metaRow = document.createElement("div");
    metaRow.className = "meta-row";
    metaRow.appendChild(buildPill(`importance ${item.weight}`, "pill"));
    metaRow.appendChild(buildPill(formatScoreValue(verdict?.score, group.maxScore) || "n/a", "score-chip"));

    const reasoning = document.createElement("p");
    reasoning.className = "small-copy";
    reasoning.textContent = firstText(verdict?.reasoning, "No verdict reasoning is available for this item.");

    itemCard.append(title, metaRow, reasoning);
    itemGrid.appendChild(itemCard);
  });

  card.append(definition, metricMeta, itemGrid);
  return card;
}

function renderOverallSummary(example) {
  const evalRow = pickEvaluationRow(example);
  const rubrics = getRubrics(evalRow);
  const verdicts = getVerdicts(evalRow);
  const groups = rubrics.length ? buildRubricGroups(rubrics) : [];

  refs.scoreStrip.textContent = "";
  groups.forEach((group) => {
    const score = computeGroupScore(group, verdicts);
    refs.scoreStrip.appendChild(buildPill(`${group.name}: ${formatScoreValue(score, group.maxScore) || "n/a"}`, "score-chip"));
  });

  const directMetricScores = getDirectMetricScores(evalRow);
  directMetricScores.forEach(({ label, score }) => {
    if (!groups.some((group) => normalizeComparisonText(group.name) === normalizeComparisonText(label))) {
      refs.scoreStrip.appendChild(buildPill(`${label}: ${score}`, "score-chip"));
    }
  });

  const overallScore = firstText(evalRow?.eval_score, "");
  if (overallScore) {
    refs.scoreStrip.appendChild(buildPill(`overall eval: ${overallScore}`, "pill"));
  }

  setFormattedContent(
    refs.overallNotesBlock,
    firstText(
      evalRow?.eval_overall_notes,
      evalRow?.overall_notes,
      "No overall evaluation notes are available for the selected row."
    )
  );
}

function pickEvaluationRow(example) {
  return example.representative || null;
}

function extractRouteData(row) {
  if (!row) {
    return { available: false, routeLabel: "", reasoning: "", responseText: "", suggestions: [] };
  }

  const route = getRouteName(row);
  const output = ensureObject(row.output);
  const suggestions = firstNonEmptyArray([
    output?.suggestions,
    row.suggestions,
    row[`${route.toLowerCase()}_suggestions`],
    row[`${camelRoute(route)}Suggestions`]
  ]);

  const responseText = firstText(
    output?.response,
    row.response,
    row.returned_final_text,
    row[`${route.toLowerCase()}_response`],
    row[`${camelRoute(route)}Response`],
    ""
  );

  const reasoning = firstText(
    output?.reasoning,
    row.reasoning,
    row[`${route.toLowerCase()}_reasoning`],
    ""
  );

  return {
    available: Boolean(responseText || suggestions.length || reasoning),
    routeLabel: normalizeRouteValue(firstText(output?.route, row.route_orch, row.route_intended, row.route, route)),
    reasoning,
    responseText,
    suggestions
  };
}

function getRubrics(row) {
  return ensureArray(row?.eval_rubrics_json);
}

function getVerdicts(row) {
  return ensureArray(row?.eval_verdicts_json);
}

function firstNonEmptyArray(values) {
  for (const value of values) {
    const array = ensureArray(value);
    if (array.length) {
      return array;
    }
  }

  return [];
}

function ensureArray(value) {
  if (Array.isArray(value)) {
    return value;
  }

  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return [];
    }

    try {
      const parsed = JSON.parse(trimmed);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  return [];
}

function ensureObject(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value;
  }

  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }

    try {
      const parsed = JSON.parse(trimmed);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
    } catch {
      return null;
    }
  }

  return null;
}

function getRouteName(row) {
  if (!row) {
    return "";
  }

  const output = ensureObject(row.output);
  return normalizeRouteValue(firstText(output?.route, row.route_orch, row.route_intended, row.route, ""));
}

function getMetricName(rubric) {
  return cleanupItemLabel(firstText(rubric.metric, "Other"));
}

function inferMetricDescription(metricName, items) {
  const lowerName = metricName.toLowerCase();
  if (lowerName.includes("relevancy")) {
    return "Assesses whether the output is relevant, tailored to the request, and contextually appropriate for the selected row.";
  }
  if (lowerName.includes("complete")) {
    return "Assesses whether the output covers the needed structural elements and organizes them into a coherent whole.";
  }

  const descriptions = uniqueValues(items.map((item) => item.description)).slice(0, 2);
  return descriptions.join(" ") || "Metric description is not available in the uploaded payload.";
}

function metricPrefix(metricName) {
  const upper = metricName
    .split(/\s+/)
    .map((part) => part[0] || "")
    .join("")
    .toUpperCase();
  return upper || "IT";
}

function matchMetricVerdict(group, verdicts) {
  const metricKey = normalizeComparisonText(group.name);
  return (
    verdicts.find((verdict) => normalizeComparisonText(firstText(verdict.metric_name, "")) === metricKey) || null
  );
}

function matchVerdict(item, verdicts, metricVerdict = null) {
  const itemKey = normalizeComparisonText(firstText(item.rawName, item.criterion));
  const metricItems = ensureArray(metricVerdict?.evaluation_items);

  const metricItemVerdict = metricItems.find((entry) => {
    const verdictKey = normalizeComparisonText(firstText(entry.item_name, ""));
    return verdictKey === itemKey || verdictKey.includes(itemKey) || itemKey.includes(verdictKey);
  });

  if (metricItemVerdict) {
    return metricItemVerdict;
  }

  return null;
}

function summarizeVerdicts(verdicts) {
  const snippets = verdicts
    .map((verdict) => firstSentence(firstText(verdict.reasoning, "")))
    .filter(Boolean)
    .slice(0, 2);
  return snippets.join(" ");
}

function computeGroupScore(group, verdicts) {
  const metricVerdict = matchMetricVerdict(group, verdicts);
  const metricScore = Number(metricVerdict?.score);
  if (!Number.isNaN(metricScore)) {
    return metricScore;
  }

  const scores = group.items
    .map((item) => Number(matchVerdict(item, verdicts, metricVerdict)?.score))
    .filter((score) => !Number.isNaN(score));

  if (!scores.length) {
    return null;
  }

  return scores.reduce((sum, score) => sum + score, 0) / scores.length;
}

function hasEvaluationData(row) {
  return Boolean(
    getRubrics(row).length ||
      getVerdicts(row).length ||
      row?.eval_overall_notes ||
      row?.eval_output_relevancy_score !== undefined ||
      row?.eval_completeness_score !== undefined ||
      row?.eval_score !== undefined
  );
}

function getDirectMetricScores(row) {
  return [
    { label: "Output Relevancy", score: firstText(row?.eval_output_relevancy_score, "") },
    { label: "Completeness", score: firstText(row?.eval_completeness_score, "") }
  ].filter((entry) => entry.score !== "");
}

function getMaxScoreFromLevels(levels) {
  const scores = ensureArray(levels)
    .map((level) => Number(level?.score))
    .filter((score) => !Number.isNaN(score));

  return scores.length ? Math.max(...scores) : null;
}

function formatScoreValue(score, maxScore = null) {
  const numeric = Number(score);
  if (Number.isNaN(numeric)) {
    return null;
  }

  const formatted = Number.isInteger(numeric) ? String(numeric) : numeric.toFixed(2);
  return maxScore === null ? formatted : `${formatted} / ${maxScore}`;
}

function cleanupItemLabel(value) {
  return firstText(value)
    .replace(/^[-*]\s*/, "")
    .replace(/^\*\*(.*?)\*\*:\s*/, "$1: ")
    .replace(/\*\*/g, "")
    .replace(/`/g, "")
    .trim();
}

function normalizeComparisonText(value) {
  return cleanupItemLabel(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function uniqueValues(values) {
  return [...new Set(values.filter((value) => value !== undefined && value !== null && value !== ""))];
}

function firstText(...values) {
  for (const value of values) {
    if (value === undefined || value === null) {
      continue;
    }

    const stringValue = String(value).trim();
    if (stringValue) {
      return stringValue;
    }
  }

  return "";
}

function firstSentence(text) {
  const trimmed = firstText(text);
  if (!trimmed) {
    return "";
  }

  const match = trimmed.match(/.*?[.!?](\s|$)/);
  return match ? match[0].trim() : trimmed;
}

function normalizeRouteValue(value) {
  return firstText(value).toUpperCase();
}

function setFormattedContent(element, value) {
  const normalized = normalizeDisplayText(value);
  element.textContent = "";
  element.classList.add("rich-content");

  if (!normalized) {
    return;
  }

  if (looksLikeMarkup(normalized)) {
    element.innerHTML = sanitizeHtml(normalized);
    return;
  }

  element.innerHTML = markdownToHtml(normalized);
}

function normalizeDisplayText(value) {
  const text = firstText(value);
  if (!text) {
    return "";
  }

  return text
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .replace(/\\t/g, "    ")
    .replace(/\r\n/g, "\n")
    .trim();
}

function looksLikeMarkup(text) {
  return /<\/?[a-z][\s\S]*>/i.test(text);
}

function sanitizeHtml(html) {
  const parser = new DOMParser();
  const documentRoot = parser.parseFromString(`<div>${html}</div>`, "text/html");
  const container = document.createElement("div");
  const source = documentRoot.body.firstElementChild;
  sanitizeNodes(source, container);
  return container.innerHTML;
}

function sanitizeNodes(sourceParent, targetParent) {
  const allowedTags = new Set(["p", "br", "strong", "b", "em", "i", "ul", "ol", "li", "code", "pre", "blockquote", "table", "thead", "tbody", "tr", "th", "td", "h1", "h2", "h3", "h4", "h5", "h6"]);

  Array.from(sourceParent.childNodes).forEach((node) => {
    if (node.nodeType === Node.TEXT_NODE) {
      targetParent.appendChild(document.createTextNode(node.textContent || ""));
      return;
    }

    if (node.nodeType !== Node.ELEMENT_NODE) {
      return;
    }

    const tag = node.tagName.toLowerCase();
    if (!allowedTags.has(tag)) {
      sanitizeNodes(node, targetParent);
      return;
    }

    const nextNode = document.createElement(tag);
    targetParent.appendChild(nextNode);
    if (tag !== "br") {
      sanitizeNodes(node, nextNode);
    }
  });
}

function markdownToHtml(markdown) {
  const lines = markdown.split("\n");
  const blocks = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      const depth = headingMatch[1].length;
      blocks.push(`<h${depth}>${formatInlineMarkdown(headingMatch[2])}</h${depth}>`);
      index += 1;
      continue;
    }

    if (/^[-*]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index])) {
        items.push(`<li>${formatInlineMarkdown(lines[index].replace(/^[-*]\s+/, ""))}</li>`);
        index += 1;
      }
      blocks.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^\d+\.\s+/.test(lines[index])) {
        items.push(`<li>${formatInlineMarkdown(lines[index].replace(/^\d+\.\s+/, ""))}</li>`);
        index += 1;
      }
      blocks.push(`<ol>${items.join("")}</ol>`);
      continue;
    }

    const paragraph = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^(#{1,6})\s+/.test(lines[index]) &&
      !/^[-*]\s+/.test(lines[index]) &&
      !/^\d+\.\s+/.test(lines[index])
    ) {
      paragraph.push(formatInlineMarkdown(lines[index]));
      index += 1;
    }

    blocks.push(`<p>${paragraph.join("<br>")}</p>`);
  }

  return blocks.join("");
}

function formatInlineMarkdown(text) {
  let escaped = escapeHtml(text);
  escaped = escaped.replace(/`([^`]+)`/g, "<code>$1</code>");
  escaped = escaped.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  escaped = escaped.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  escaped = escaped.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  escaped = escaped.replace(/_([^_]+)_/g, "<em>$1</em>");
  return escaped;
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function routeLabel(route) {
  return route.replace(/_/g, " ");
}

function camelRoute(route) {
  return route
    .toLowerCase()
    .split("_")
    .map((part, index) => (index === 0 ? part : part[0].toUpperCase() + part.slice(1)))
    .join("");
}

function buildPill(text, className) {
  const pill = document.createElement("span");
  pill.className = className;
  pill.textContent = text;
  return pill;
}

function buildMutedBlock(text) {
  const block = document.createElement("div");
  block.className = "note-box muted";
  setFormattedContent(block, text);
  return block;
}

function buildPlaceholder(text) {
  const block = document.createElement("div");
  block.className = "placeholder";
  block.textContent = text;
  return block;
}

function showFatal(message) {
  state.examples = [];
  state.filteredExamples = [];
  state.selectedKey = null;
  refs.fileStatus.textContent = message;
  refs.warningBox.classList.add("hidden");
  renderFilterControls();
  renderSelectedExample();
}
