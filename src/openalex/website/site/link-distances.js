(() => {
  "use strict";

  const svg = document.querySelector("#link-scatter");
  const details = document.querySelector("#link-details");
  const tooltip = document.querySelector("#link-tooltip");
  const summary = document.querySelector("#summary");
  const search = document.querySelector("#cluster-search");
  const tabs = [...document.querySelectorAll("[data-measure]")];
  const typeButtons = [...document.querySelectorAll("[data-cluster-type]")];
  const NS = "http://www.w3.org/2000/svg";
  const state = {
    rows: [],
    measure: "new",
    points: [],
    hovered: null,
    query: "",
  };
  const measures = {
    new: {
      distance: "average_new_link_distance",
      connected: "new_link_connected_count",
      disconnected: "new_link_disconnected_count",
      distribution: "new_link_distance_distribution",
      baselineDistribution: "new_link_baseline_distance_distribution",
      prefix: "new_link",
      yearly: "new_link_distance_by_year",
      label: "Average first-link graph distance",
      description: "first coauthorship links measured in the prior graph",
    },
    "first-semantic": {
      distance: "average_first_semantic_link_distance",
      distribution: "first_semantic_link_distance_distribution",
      baselineDistribution: "first_semantic_link_baseline_distance_distribution",
      yearly: "first_semantic_link_distance_by_year",
      prefix: "first_semantic_link",
      label: "Average first-link semantic distance",
      description: "first coauthorship links using leave-cluster-out author embeddings",
      semantic: true,
    },
    semantic: {
      distance: "average_semantic_link_distance",
      distribution: "semantic_link_distance_distribution",
      baselineDistribution: "semantic_link_baseline_distance_distribution",
      yearly: "semantic_link_distance_by_year",
      prefix: "semantic_link",
      label: "Average all-link semantic distance",
      description: "all coauthor-pair occurrences using leave-cluster-out author embeddings",
      semantic: true,
    },
    all: {
      distance: "average_all_link_distance",
      connected: "all_link_connected_count",
      disconnected: "all_link_disconnected_count",
      distribution: "all_link_distance_distribution",
      baselineDistribution: "all_link_baseline_distance_distribution",
      prefix: "all_link",
      yearly: "all_link_distance_by_year",
      label: "Average all-link graph distance",
      description: "all coauthor-pair observations on cluster papers measured in the graph",
    },
    diversity: {
      distance: "rao_stirling_diversity",
      distribution: "topic_distribution",
      label: "Rao–Stirling topic diversity",
      description: "paper-topic diversity using c-TF-IDF cosine dissimilarity",
      diversity: true,
    },
    "author-diversity": {
      distance: "author_rao_stirling_diversity",
      distribution: "author_topic_distribution",
      label: "Author primary-topic diversity",
      description: "fractionally weighted author primary-topic diversity",
      diversity: true,
      authorDiversity: true,
    },
    "within-collaboration-diversity": {
      distance: "author_within_collaboration_diversity",
      xDistance: "author_rao_stirling_diversity",
      distribution: "author_topic_distribution",
      label: "Mean within-collaboration diversity",
      xLabel: "Overall author diversity",
      description: "within-collaboration diversity compared with overall author diversity",
      diversity: true,
      authorDiversity: true,
      diversityComparison: true,
    },
  };
  const clusterMethodology = [
    ["Cluster size", "Number of distinct papers containing at least one keyword in the cluster. Scatter plots using cluster size place it on a base-10 logarithmic x-axis."],
    ["Share of papers by year", "Cluster papers in year y divided by all analyzed corpus papers published in year y."],
  ];
  const graphInferenceMethodology = [
    ["Disconnected share", "Disconnected observations / all eligible observations. Disconnected pairs are excluded from finite graph-distance means."],
    ["Matched baseline", "The same disconnected proportion in the year-matched reference population."],
    ["Disconnect Δ", "Selected-cluster disconnected share − matched-baseline disconnected share."],
    ["Disconnect 95% CI", "Year-stratified nonparametric bootstrap interval for Disconnect Δ."],
    ["Disconnect p and q", "Year-stratified permutation-test p-value and Benjamini–Hochberg FDR-adjusted q-value across clusters."],
    ["Mean-hop Δ", "Mean finite graph distance in the selected cluster − mean finite distance in its year-matched baseline."],
    ["Mean-hop 95% CI", "Year-stratified nonparametric bootstrap interval for Mean-hop Δ."],
    ["Wasserstein-1", "Earth-mover distance between the selected and year-matched finite graph-distance distributions; larger values mean a greater distributional shift."],
    ["Wasserstein 95% CI", "Year-stratified nonparametric bootstrap interval for Wasserstein-1."],
    ["Distance p and q", "Permutation-test p-value for Wasserstein-1 and its Benjamini–Hochberg FDR-adjusted q-value."],
  ];
  const leaveClusterOutMethodology = [
    ["Leave-cluster-out author profile", "For author a and cluster c: eₐ,₋c = [Σₚ∉c (1/nₚ)eₚ] / [Σₚ∉c (1/nₚ)], where eₚ is the paper embedding and nₚ its author count. All embedding-covered cluster papers are removed."],
    ["Cosine distance", "d(a,b|c) = 1 − cos(eₐ,₋c, eᵦ,₋c), clipped to [0,2]. Zero means aligned profiles; larger values mean more semantically different profiles."],
    ["Embedding coverage", "Covered / (covered + without-profile), where a record is covered only if both authors retain a leave-cluster-out profile."],
  ];
  const methodologies = {
    new: [
      ...clusterMethodology,
      ["First link", "One unordered author pair in its first observed coauthorship year. It is attributed to a cluster only when exactly that cluster appears on every paper creating the link in that year."],
      ["Graph distance", "Exact unweighted shortest-path length between the two authors in the cumulative coauthorship graph through year y−1."],
      ["Mean distance", "Inverse-inclusion-weighted mean graph distance among connected, uniquely attributed first links."],
      ["Connected and disconnected", "Exact first-link population counts by whether a path exists in the graph through year y−1."],
      ["Reference distribution", "First links without a unique cluster, post-stratified to the selected cluster’s first-link year distribution."],
      ...graphInferenceMethodology,
    ],
    "first-semantic": [
      ...clusterMethodology,
      ["First link", "One uniquely attributed unordered author pair in its first observed coauthorship year; repeated papers do not create repeated records."],
      ...leaveClusterOutMethodology,
      ["Mean cosine distance", "Inverse-inclusion-weighted mean leave-cluster-out cosine distance across covered first links."],
      ["Covered first links", "Exact number of uniquely attributed first links for which both authors retain a leave-cluster-out profile."],
      ["First links without profiles", "Exact number excluded because at least one author has no remaining profile."],
      ["Sampled first links", "Number retained by the independent per-cluster, per-year simple random sample, capped by --cluster-sample."],
      ["Reference distribution", "Covered first links without a unique cluster, sampled separately for each cluster and year using that cluster’s exclusion, then matched to the selected year distribution."],
      ["Histogram and yearly mean", "Histogram bars use cosine-distance bins of width 0.05 on [0,2]; yearly points are sampling-weighted means."],
    ],
    all: [
      ...clusterMethodology,
      ["All-link observation", "One unordered coauthor-pair occurrence on one cluster paper. Repeated collaborations on different papers remain separate observations."],
      ["Graph distance", "Existing links are assigned distance 1; new connected pairs use exact shortest-path distance in the graph through year y−1."],
      ["Mean distance", "Reservoir-weighted mean graph distance over connected all-link observations."],
      ["Connected and disconnected", "Exact occurrence counts by whether a path exists before the paper year."],
      ["Existing links", "Occurrences whose author pair had already collaborated before the paper year."],
      ["Repeat share", "Existing-link occurrences / all eligible all-link occurrences."],
      ["Repeat-share Δ and 95% CI", "Selected repeat share − year-matched baseline repeat share, with a year-stratified bootstrap interval."],
      ["Reference distribution", "All-link occurrences from other clusters, excluding the selected cluster and post-stratified to its occurrence-year distribution."],
      ...graphInferenceMethodology,
    ],
    semantic: [
      ...clusterMethodology,
      ["All-link observation", "One unordered author-pair occurrence on one cluster paper. A pair appearing on f qualifying papers contributes f records."],
      ...leaveClusterOutMethodology,
      ["Mean cosine distance", "Mean of the cluster’s uniform global reservoir of covered pair occurrences; the reservoir is capped by --cluster-sample."],
      ["Covered occurrences", "Exact number of pair occurrences for which both authors retain a leave-cluster-out profile."],
      ["Occurrences without profiles", "Exact number excluded because at least one author has no remaining profile."],
      ["Sampled occurrences", "Number retained in the cluster’s uniform global reservoir."],
      ["Reference distribution", "Embedding-covered pair occurrences sampled from all analyzed papers separately for each cluster and year, using the same cluster exclusion and matched to the selected year mix."],
      ["Histogram and yearly mean", "Histogram bars use cosine-distance bins of width 0.05 on [0,2]; yearly points summarize sampled covered occurrences accepted in that year."],
    ],
    diversity: [
      ...clusterMethodology,
      ["Paper topic", "Hard full-corpus BERTopic classifier assignment. Classifier confidence is not used; missing assignments and topic −1 outliers are excluded."],
      ["Topic share", "Assigned cluster papers in topic i / all assigned, non-outlier cluster papers."],
      ["Topic dissimilarity", "dᵢⱼ = 1 − cosine(vᵢ,vⱼ), where vᵢ and vⱼ are normalized topic c-TF-IDF vectors."],
      ["Rao–Stirling diversity", "D = ΣᵢΣⱼ pᵢpⱼdᵢⱼ = 1 − ‖Σᵢpᵢvᵢ‖². It is the scatter y-variable."],
      ["Assigned, missing, and outlier papers", "Exact cluster-paper counts in the diversity distribution, without an assignment, and assigned topic −1, respectively."],
      ["Topic coverage", "Assigned non-outlier papers / all cluster papers represented by assigned, missing, or outlier counts."],
      ["Paper-topic distribution", "Bars show topic shares, ordered by paper frequency."],
    ],
    "author-diversity": [
      ...clusterMethodology,
      ["Leave-cluster-out primary topic", "For each author, subtract every current-cluster paper’s 1/n fractional topic weight from the full profile, then choose the remaining topic with greatest weight."],
      ["Fractional paper weight", "Each authorship on an n-author cluster paper contributes 1/n, so every paper contributes total weight 1, including single-author papers."],
      ["Author topic share", "Fractional weight assigned to primary topic i / all fractional weight with a valid primary topic."],
      ["Author Rao–Stirling diversity", "D = ΣᵢΣⱼ pᵢpⱼ(1 − cosine(vᵢ,vⱼ)) over the pooled fractional primary-topic distribution. It is the scatter y-variable."],
      ["Involved and classified authors", "Distinct authors on cluster papers and the subset retaining a valid leave-cluster-out primary topic."],
      ["Missing primary topics", "Distinct involved authors with no valid non-outlier topic after cluster subtraction."],
      ["Classified weight", "Fractional paper weight contributed by authors with valid primary topics."],
      ["Weighted coverage", "Classified fractional weight / total fractional paper weight."],
      ["Author primary-topic distribution", "Bars show pooled fractional primary-topic shares, ordered by weight."],
    ],
    "within-collaboration-diversity": [
      ...clusterMethodology,
      ["Overall author diversity (x)", "The pooled author Rao–Stirling diversity: Doverall = ΣᵢΣⱼ pᵢpⱼdᵢⱼ using leave-cluster-out primary topics and 1/n fractional authorship weights."],
      ["Within-collaboration diversity (paper)", "For paper p, Dp = ΣᵢΣⱼ pᵢ|p pⱼ|p dᵢⱼ across covered coauthors’ primary topics, with covered coauthors weighted equally within that paper."],
      ["Mean within-collaboration diversity (y)", "Dwithin = (1/M)ΣₚDp over the M eligible multi-author papers. Every eligible paper has equal weight, regardless of team size."],
      ["Overall − within", "Doverall − Dwithin. With complete topic coverage this is the between-collaboration component; incomplete coverage can break the exact decomposition."],
      ["Diagonal reference", "The line y = x. Points near it have similar within and overall diversity; points below it have a larger between-collaboration component."],
      ["Multi-author collaborations", "Cluster papers with at least two authors."],
      ["Eligible collaborations", "Multi-author cluster papers with at least two covered leave-cluster-out primary topics."],
      ["Insufficient topic coverage", "Multi-author papers with fewer than two covered primary topics, excluded from Dwithin."],
      ["Collaboration coverage", "Eligible collaborations / all multi-author collaborations."],
      ["Author primary-topic distribution", "The detail bars show the pooled fractional distribution used for the x-variable."],
    ],
  };

  function methodologyPanel(measure) {
    const section = document.createElement("section");
    section.className = "methodology";
    const heading = document.createElement("h3");
    heading.textContent = "Formula and methodology";
    const definitions = document.createElement("dl");
    for (const [termText, detailText] of methodologies[measure] || []) {
      const term = document.createElement("dt");
      term.textContent = termText;
      const detail = document.createElement("dd");
      detail.textContent = detailText;
      definitions.append(term, detail);
    }
    section.append(heading, definitions);
    return section;
  }

  function svgNode(name, attrs = {}) {
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    return node;
  }

  function formatPercent(share) {
    const percent = 100 * Number(share || 0);
    const digits = percent >= 10 ? 1 : percent >= 1 ? 2 : 3;
    return `${percent.toFixed(digits)}%`;
  }

  function formatDistance(value) {
    if (value == null) return "—";
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
  }

  function formatDiversity(value) {
    if (value == null) return "—";
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 });
  }

  function formatProbability(value) {
    return value == null ? "—" : formatPercent(value);
  }

  function formatPValue(value) {
    if (value == null) return "—";
    return Number(value) < 0.001 ? "<0.001" : Number(value).toFixed(3);
  }

  function formatInterval(low, high, formatter = formatDistance) {
    return low == null || high == null ? "—" : `[${formatter(low)}, ${formatter(high)}]`;
  }

  function selectedTypes() {
    return new Set(
      typeButtons
        .filter((button) => button.getAttribute("aria-pressed") === "true")
        .map((button) => button.dataset.clusterType),
    );
  }

  function typeFilterIsOpen() {
    return selectedTypes().size === typeButtons.length;
  }

  function matchesQuery(row) {
    if (!state.query) return true;
    const haystack = [
      row.label,
      ...(row.keywords || []),
      String(row.cluster_id),
      `cluster ${row.cluster_id}`,
    ].join(" ").toLocaleLowerCase();
    return haystack.includes(state.query);
  }

  function matchesType(row) {
    if (typeFilterIsOpen()) return true;
    return selectedTypes().has(row.cluster_type);
  }

  function filteredRows() {
    return state.rows.filter((row) => matchesQuery(row) && matchesType(row));
  }

  function updateSummary() {
    const visible = filteredRows().length;
    const narrowed = Boolean(state.query) || !typeFilterIsOpen();
    const count = narrowed
      ? `${visible.toLocaleString()} of ${state.rows.length.toLocaleString()} clusters`
      : `${state.rows.length.toLocaleString()} clusters`;
    summary.textContent = `${count} · ${measures[state.measure].description}`;
  }

  function clearSelection() {
    state.hovered = null;
    tooltip.hidden = true;
    details.innerHTML = `
      <div class="empty">
        <p class="eyebrow">Event cluster</p>
        <h2>Explore the points</h2>
        <p>Each visible point is one matching cluster. Hover it to see its temporal curves and detailed distribution.</p>
      </div>
    `;
    details.append(methodologyPanel(state.measure));
  }

  function lineChart(values) {
    const chart = svgNode("svg", {
      class: "line-chart link-curve",
      viewBox: "0 0 340 180",
      role: "img",
    });
    if (!values.length) {
      const message = svgNode("text", { class: "axis-label", x: 16, y: 30 });
      message.textContent = "No yearly observations";
      chart.append(message);
      return chart;
    }
    const padding = { left: 52, right: 12, top: 16, bottom: 30 };
    const years = values.map((item) => Number(item.year));
    const shares = values.map((item) => Number(item.share) || 0);
    const minYear = Math.min(...years);
    const maxYear = Math.max(...years);
    const maxShare = Math.max(0, ...shares);
    const scale = maxShare || 1;
    const x = (year) => padding.left + ((year - minYear) / Math.max(1, maxYear - minYear)) * (340 - padding.left - padding.right);
    const y = (share) => 180 - padding.bottom - (share / scale) * (180 - padding.top - padding.bottom);
    chart.append(
      svgNode("path", {
        class: "axis",
        d: `M ${padding.left} ${padding.top} V ${180 - padding.bottom} H ${340 - padding.right}`,
      }),
      svgNode("path", {
        class: "series",
        d: values.map((item, index) => `${index ? "L" : "M"} ${x(Number(item.year))} ${y(Number(item.share) || 0)}`).join(" "),
      }),
    );
    for (const item of values) {
      const point = svgNode("circle", {
        class: "series-point",
        cx: x(Number(item.year)),
        cy: y(Number(item.share) || 0),
        r: 3.5,
      });
      const title = svgNode("title");
      title.textContent = `${item.year}: ${formatPercent(item.share)} of papers (${Number(item.papers).toLocaleString()})`;
      point.append(title);
      chart.append(point);
    }
    const start = svgNode("text", { class: "axis-label", x: padding.left, y: 172 });
    start.textContent = String(minYear);
    const finish = svgNode("text", { class: "axis-label end", x: 340 - padding.right, y: 172 });
    finish.textContent = String(maxYear);
    const maximum = svgNode("text", { class: "axis-label", x: 4, y: padding.top + 4 });
    maximum.textContent = formatPercent(maxShare);
    chart.append(start, finish, maximum);
    chart.setAttribute(
      "aria-label",
      `Share of papers by year: ${values.map((item) => `${item.year} ${formatPercent(item.share)}`).join(", ")}`,
    );
    return chart;
  }

  function distanceYearChart(values) {
    const chart = svgNode("svg", {
      class: "line-chart link-curve",
      viewBox: "0 0 340 180",
      role: "img",
    });
    if (!values.length) {
      const message = svgNode("text", { class: "axis-label", x: 16, y: 30 });
      message.textContent = "No yearly distance observations";
      chart.append(message);
      return chart;
    }
    const padding = { left: 52, right: 12, top: 16, bottom: 30 };
    const years = values.map((item) => Number(item.year));
    const distances = values.map((item) => Number(item.distance) || 0);
    const minYear = Math.min(...years);
    const maxYear = Math.max(...years);
    const maxDistance = Math.max(0, ...distances);
    const scale = maxDistance || 1;
    const x = (year) => padding.left + ((year - minYear) / Math.max(1, maxYear - minYear)) * (340 - padding.left - padding.right);
    const y = (distance) => 180 - padding.bottom - (distance / scale) * (180 - padding.top - padding.bottom);
    chart.append(
      svgNode("path", {
        class: "axis",
        d: `M ${padding.left} ${padding.top} V ${180 - padding.bottom} H ${340 - padding.right}`,
      }),
      svgNode("path", {
        class: "series",
        d: values.map((item, index) => `${index ? "L" : "M"} ${x(Number(item.year))} ${y(Number(item.distance) || 0)}`).join(" "),
      }),
    );
    for (const item of values) {
      const point = svgNode("circle", {
        class: "series-point",
        cx: x(Number(item.year)),
        cy: y(Number(item.distance) || 0),
        r: 3.5,
      });
      const title = svgNode("title");
      title.textContent = `${item.year}: mean distance ${formatDistance(item.distance)}`;
      point.append(title);
      chart.append(point);
    }
    const start = svgNode("text", { class: "axis-label", x: padding.left, y: 172 });
    start.textContent = String(minYear);
    const finish = svgNode("text", { class: "axis-label end", x: 340 - padding.right, y: 172 });
    finish.textContent = String(maxYear);
    const maximum = svgNode("text", { class: "axis-label", x: 4, y: padding.top + 4 });
    maximum.textContent = formatDistance(maxDistance);
    chart.append(start, finish, maximum);
    chart.setAttribute(
      "aria-label",
      `Mean connected distance by year: ${values.map((item) => `${item.year} ${formatDistance(item.distance)}`).join(", ")}`,
    );
    return chart;
  }

  function orderedTopics(values, valueField = "papers") {
    return [...(values || [])].sort(
      (left, right) => Number(right[valueField]) - Number(left[valueField])
        || Number(left.topic_id) - Number(right.topic_id),
    );
  }

  function topicTopWord(item) {
    const label = String(item.label || "").replace(/^-?\d+[_\s-]+/, "");
    return label.split("_").find((word) => word.length) || `T${item.topic_id}`;
  }

  function topicDistributionChart(row) {
    const config = measures[state.measure];
    const valueField = config.authorDiversity ? "weight" : "papers";
    const values = orderedTopics(row[config.distribution], valueField);
    const wrapper = document.createElement("div");
    wrapper.className = "topic-histogram-scroll";
    const tooltip = document.createElement("div");
    tooltip.className = "topic-tooltip";
    tooltip.hidden = true;
    const width = Math.max(340, 54 + values.length * 38);
    const height = 250;
    const chart = svgNode("svg", {
      class: "line-chart topic-detail-chart",
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
    });
    chart.style.width = `${width}px`;
    if (!values.length) {
      const message = svgNode("text", { class: "axis-label", x: 16, y: 30 });
      message.textContent = "No topic assignments";
      chart.append(message);
      wrapper.append(chart, tooltip);
      return wrapper;
    }
    const padding = { left: 46, right: 12, top: 16, bottom: 92 };
    const baseline = height - padding.bottom;
    const maxShare = Math.max(...values.map((item) => Number(item.share))) || 1;
    const step = (width - padding.left - padding.right) / values.length;
    const barWidth = Math.max(5, Math.min(24, step * 0.68));
    const x = (index) => padding.left + step * (index + 0.5);
    const y = (share) => baseline
      - (Number(share) / maxShare) * (height - padding.top - padding.bottom);
    chart.append(svgNode("path", {
      class: "axis",
      d: `M ${padding.left} ${padding.top} V ${baseline} H ${width - padding.right}`,
    }));
    values.forEach((item, index) => {
      const bar = svgNode("rect", {
        class: "topic-bar",
        x: x(index) - barWidth / 2,
        y: y(item.share),
        width: barWidth,
        height: Math.max(0, baseline - y(item.share)),
      });
      const hit = svgNode("rect", {
        class: "topic-hit",
        x: padding.left + step * index,
        y: padding.top,
        width: step,
        height: baseline - padding.top,
      });
      const detail = config.authorDiversity
        ? `${item.label}: ${Number(item.authors).toLocaleString()} authors, ${Number(item.weight).toLocaleString(undefined, { maximumFractionDigits: 2 })} fractional paper weight (${formatPercent(item.share)})`
        : `${item.label}: ${Number(item.papers).toLocaleString()} papers (${formatPercent(item.share)})`;
      hit.addEventListener("pointerenter", (event) => showTopicTooltip(wrapper, tooltip, event, detail));
      hit.addEventListener("pointermove", (event) => showTopicTooltip(wrapper, tooltip, event, detail));
      hit.addEventListener("pointerleave", () => {
        tooltip.hidden = true;
      });
      const label = svgNode("text", {
        class: "axis-label",
        x: x(index),
        y: baseline + 12,
        "text-anchor": "end",
        transform: `rotate(-45 ${x(index)} ${baseline + 12})`,
      });
      label.textContent = topicTopWord(item);
      chart.append(bar, hit, label);
    });
    const maximum = svgNode("text", {
      class: "axis-label",
      x: 3,
      y: padding.top + 4,
    });
    maximum.textContent = formatPercent(maxShare);
    const xLabel = svgNode("text", {
      class: "axis-label",
      x: (padding.left + width - padding.right) / 2,
      y: height - 8,
      "text-anchor": "middle",
    });
    xLabel.textContent = config.authorDiversity
      ? "Primary topics, ordered by fractional author weight"
      : "Topics, ordered by frequency";
    chart.append(maximum, xLabel);
    chart.setAttribute(
      "aria-label",
      `${config.authorDiversity ? "Author primary-topic" : "Topic"} distribution for ${row.label}, ordered by frequency: ${values.map(
        (item) => `${item.label} ${formatPercent(item.share)}`,
      ).join(", ")}`,
    );
    wrapper.append(chart, tooltip);
    return wrapper;
  }

  function showTopicTooltip(frame, tooltip, event, text) {
    const bounds = frame.getBoundingClientRect();
    tooltip.hidden = false;
    tooltip.textContent = text;
    tooltip.style.left = `${event.clientX - bounds.left + frame.scrollLeft + 12}px`;
    tooltip.style.top = `${event.clientY - bounds.top - 28}px`;
  }

  function distributionChart(row) {
    const config = measures[state.measure];
    if (config.diversity) return topicDistributionChart(row);
    if (config.semantic) return semanticDistributionChart(row, config);
    const selected = distributionMap(row[config.distribution], state.measure);
    const baseline = distributionMap(row[config.baselineDistribution], state.measure);
    const maximumDistance = Math.max(0, ...selected.keys(), ...baseline.keys());
    const selectedTotal = [...selected.values()].reduce((total, count) => total + count, 0);
    const baselineTotal = [...baseline.values()].reduce((total, count) => total + count, 0);
    const selectedShare = (distance) => (selectedTotal ? (selected.get(distance) || 0) / selectedTotal : 0);
    const baselineShare = (distance) => (baselineTotal ? (baseline.get(distance) || 0) / baselineTotal : 0);
    const chart = svgNode("svg", {
      class: "line-chart distance-chart",
      viewBox: "0 0 340 190",
      role: "img",
    });
    if (!selectedTotal) {
      const message = svgNode("text", { class: "axis-label", x: 16, y: 30 });
      message.textContent = "No connected distance observations";
      chart.append(message);
      return chart;
    }
    const padding = { left: 46, right: 12, top: 16, bottom: 34 };
    const values = Array.from({ length: maximumDistance + 1 }, (_, distance) => distance);
    const maxShare = Math.max(
      0,
      ...values.map(selectedShare),
      ...values.map(baselineShare),
    ) || 1;
    const plotWidth = 340 - padding.left - padding.right;
    const step = plotWidth / Math.max(1, values.length);
    const x = (distance) => padding.left + step * (distance + 0.5);
    const y = (share) => 190 - padding.bottom - (share / maxShare) * (190 - padding.top - padding.bottom);
    chart.append(
      svgNode("path", {
        class: "axis",
        d: `M ${padding.left} ${padding.top} V ${190 - padding.bottom} H ${340 - padding.right}`,
      }),
    );
    const barWidth = Math.max(2, Math.min(20, step * 0.58));
    for (const distance of values) {
      const share = selectedShare(distance);
      chart.append(
        svgNode("rect", {
          class: "distance-bar",
          x: x(distance) - barWidth / 2,
          y: y(share),
          width: barWidth,
          height: Math.max(0, 190 - padding.bottom - y(share)),
        }),
      );
      if (values.length <= 14 || distance % Math.ceil(values.length / 10) === 0) {
        const label = svgNode("text", {
          class: "axis-label",
          x: x(distance),
          y: 176,
          "text-anchor": "middle",
        });
        label.textContent = String(distance);
        chart.append(label);
      }
    }
    const baselinePath = values
      .map((distance, index) => `${index ? "L" : "M"} ${x(distance)} ${y(baselineShare(distance))}`)
      .join(" ");
    chart.append(svgNode("path", { class: "baseline-series", d: baselinePath }));
    for (const distance of values) {
      chart.append(
        svgNode("circle", {
          class: "baseline-point",
          cx: x(distance),
          cy: y(baselineShare(distance)),
          r: 2.5,
        }),
      );
    }
    const maximum = svgNode("text", { class: "axis-label", x: 3, y: padding.top + 4 });
    maximum.textContent = formatPercent(maxShare);
    const xLabel = svgNode("text", {
      class: "axis-label",
      x: (padding.left + 340 - padding.right) / 2,
      y: 189,
      "text-anchor": "middle",
    });
    xLabel.textContent = "Network distance";
    chart.append(maximum, xLabel);
    chart.setAttribute(
      "aria-label",
      `Connected-distance distribution for ${row.label}, compared with ${baselineLabel()}`,
    );
    return chart;
  }

  function semanticDistributionChart(row, config) {
    const bins = row[config.distribution] || [];
    const baselineBins = row[config.baselineDistribution] || [];
    const chart = svgNode("svg", {
      class: "line-chart distance-chart",
      viewBox: "0 0 340 190",
      role: "img",
    });
    const total = bins.reduce((sum, item) => sum + Number(item[2] || 0), 0);
    const baselineTotal = baselineBins.reduce((sum, item) => sum + Number(item[2] || 0), 0);
    if (!total) {
      const message = svgNode("text", { class: "axis-label", x: 16, y: 30 });
      message.textContent = "No semantic distance observations";
      chart.append(message);
      return chart;
    }
    const padding = { left: 46, right: 12, top: 16, bottom: 34 };
    const plotWidth = 340 - padding.left - padding.right;
    const shares = bins.map((item) => Number(item[2]) / total);
    const baselineByBin = new Map(
      baselineBins.map((item) => [
        Math.round(Number(item[0]) * 20),
        baselineTotal ? Number(item[2]) / baselineTotal : 0,
      ]),
    );
    const maxShare = Math.max(...shares, ...baselineByBin.values()) || 1;
    const x = (distance) => padding.left + (Number(distance) / 2) * plotWidth;
    const y = (share) => 190 - padding.bottom - (share / maxShare) * (190 - padding.top - padding.bottom);
    chart.append(
      svgNode("path", {
        class: "axis",
        d: `M ${padding.left} ${padding.top} V ${190 - padding.bottom} H ${340 - padding.right}`,
      }),
    );
    bins.forEach((item, index) => {
      const left = Number(item[0]);
      const right = Number(item[1]);
      const share = shares[index];
      chart.append(
        svgNode("rect", {
          class: "distance-bar",
          x: x(left),
          y: y(share),
          width: Math.max(1, x(right) - x(left) - 0.5),
          height: Math.max(0, 190 - padding.bottom - y(share)),
        }),
      );
    });
    const baselinePath = Array.from({ length: 40 }, (_, index) => {
      const center = (index + 0.5) / 20;
      return `${index ? "L" : "M"} ${x(center)} ${y(baselineByBin.get(index) || 0)}`;
    }).join(" ");
    chart.append(svgNode("path", { class: "baseline-series", d: baselinePath }));
    for (let index = 0; index < 40; index += 1) {
      chart.append(
        svgNode("circle", {
          class: "baseline-point",
          cx: x((index + 0.5) / 20),
          cy: y(baselineByBin.get(index) || 0),
          r: 2,
        }),
      );
    }
    for (const distance of [0, 0.5, 1, 1.5, 2]) {
      const label = svgNode("text", {
        class: "axis-label",
        x: x(distance),
        y: 176,
        "text-anchor": "middle",
      });
      label.textContent = formatDistance(distance);
      chart.append(label);
    }
    const maximum = svgNode("text", { class: "axis-label", x: 3, y: padding.top + 4 });
    maximum.textContent = formatPercent(maxShare);
    const xLabel = svgNode("text", {
      class: "axis-label",
      x: (padding.left + 340 - padding.right) / 2,
      y: 189,
      "text-anchor": "middle",
    });
    xLabel.textContent = "Cosine distance";
    chart.append(maximum, xLabel);
    chart.setAttribute(
      "aria-label",
      `Semantic-distance distribution for ${row.label}, compared with ${baselineLabel()}`,
    );
    return chart;
  }

  function distributionMap(pairs, measure) {
    const counts = new Map();
    for (const [distance, count] of pairs || []) {
      const shown = measure === "all" && Number(distance) === 0 ? 1 : Number(distance);
      counts.set(shown, (counts.get(shown) || 0) + Number(count));
    }
    return counts;
  }

  function baselineLabel() {
    if (state.measure === "first-semantic") {
      return "Year-matched first links outside clusters (same cluster excluded)";
    }
    if (state.measure === "semantic") {
      return "Year-matched random paper links (same cluster excluded)";
    }
    return state.measure === "all"
      ? "Year-matched links outside this cluster"
      : "Year-matched new links outside clusters";
  }

  function show(row) {
    state.hovered = row;
    details.replaceChildren();
    const config = measures[state.measure];
    const eyebrow = document.createElement("p");
    eyebrow.className = "eyebrow";
    eyebrow.textContent = `Cluster ${row.cluster_id}`;
    const heading = document.createElement("h2");
    heading.textContent = row.label;
    const type = document.createElement("p");
    type.className = "cluster-type";
    type.dataset.type = row.cluster_type || "";
    type.textContent = row.cluster_type || "Unclassified";
    const stats = document.createElement("div");
    stats.className = "link-stats";
    const values = [["Cluster size", `${Number(row.paper_count).toLocaleString()} papers`]];
    if (config.diversityComparison) {
      const collaborations = Number(row.author_collaboration_paper_count || 0);
      const eligible = Number(row.author_within_collaboration_paper_count || 0);
      const missing = Number(row.author_within_collaboration_missing_paper_count || 0);
      const overall = row[config.xDistance];
      const within = row[config.distance];
      values.push(
        ["Overall author diversity", formatDiversity(overall)],
        ["Mean within-collaboration diversity", formatDiversity(within)],
        ["Overall − within", formatDiversity(
          overall == null || within == null ? null : Number(overall) - Number(within),
        )],
        ["Multi-author collaborations", collaborations.toLocaleString()],
        ["Eligible collaborations", eligible.toLocaleString()],
        ["Insufficient topic coverage", missing.toLocaleString()],
        ["Collaboration coverage", formatProbability(collaborations ? eligible / collaborations : null)],
      );
    } else if (config.authorDiversity) {
      const authors = Number(row.author_count || 0);
      const covered = Number(row.author_primary_topic_count || 0);
      const missing = Number(row.author_primary_topic_missing_count || 0);
      const totalWeight = Number(row.author_weight || 0);
      const coveredWeight = Number(row.author_primary_topic_weight || 0);
      values.push(
        ["Rao–Stirling diversity", formatDiversity(row[config.distance])],
        ["Involved authors", authors.toLocaleString()],
        ["Classified authors", covered.toLocaleString()],
        ["Missing primary topics", missing.toLocaleString()],
        ["Fractional paper weight", formatDiversity(totalWeight)],
        ["Classified weight", formatDiversity(coveredWeight)],
        ["Weighted coverage", formatProbability(totalWeight ? coveredWeight / totalWeight : null)],
      );
    } else if (config.diversity) {
      const covered = Number(row.topic_observation_count || 0);
      const missing = Number(row.topic_missing_count || 0);
      const outliers = Number(row.topic_outlier_count || 0);
      const total = covered + missing + outliers;
      values.push(
        ["Rao–Stirling diversity", formatDiversity(row[config.distance])],
        ["Assigned papers", covered.toLocaleString()],
        ["Missing topics", missing.toLocaleString()],
        ["Topic outliers", outliers.toLocaleString()],
        ["Topic coverage", formatProbability(total ? covered / total : null)],
      );
    } else if (config.semantic) {
      const covered = Number(row[`${config.prefix}_observation_count`] || 0);
      const missing = Number(row[`${config.prefix}_missing_embedding_count`] || 0);
      const coverage = covered + missing ? covered / (covered + missing) : null;
      const firstLinks = state.measure === "first-semantic";
      values.push(
        ["Mean cosine distance", formatDistance(row[config.distance])],
        [firstLinks ? "Covered first links" : "Covered occurrences", covered.toLocaleString()],
        [firstLinks ? "First links without profiles" : "Occurrences without profiles", missing.toLocaleString()],
        ["Embedding coverage", formatProbability(coverage)],
        [firstLinks ? "Sampled first links" : "Sampled occurrences", Number(row[`${config.prefix}_distance_sample_count`] || 0).toLocaleString()],
      );
    } else {
      values.push(
        ["Mean distance", formatDistance(row[config.distance])],
        ["Connected", Number(row[config.connected]).toLocaleString()],
        ["Disconnected", Number(row[config.disconnected]).toLocaleString()],
        ["Disconnected share", formatProbability(row[`${config.prefix}_disconnection_probability`])],
        ["Matched baseline", formatProbability(row[`${config.prefix}_baseline_disconnection_probability`])],
        ["Disconnect Δ", formatProbability(row[`${config.prefix}_disconnection_risk_difference`])],
        ["Disconnect 95% CI", formatInterval(
          row[`${config.prefix}_disconnection_ci_low`],
          row[`${config.prefix}_disconnection_ci_high`],
          formatProbability,
        )],
        ["Disconnect p", formatPValue(row[`${config.prefix}_disconnection_p_value`])],
        ["Disconnect q", formatPValue(row[`${config.prefix}_disconnection_q_value`])],
        ["Mean-hop Δ", formatDistance(row[`${config.prefix}_mean_distance_shift`])],
        ["Mean-hop 95% CI", formatInterval(
          row[`${config.prefix}_mean_distance_shift_ci_low`],
          row[`${config.prefix}_mean_distance_shift_ci_high`],
        )],
        ["Wasserstein-1", formatDistance(row[`${config.prefix}_wasserstein_distance`])],
        ["Wasserstein 95% CI", formatInterval(
          row[`${config.prefix}_wasserstein_ci_low`],
          row[`${config.prefix}_wasserstein_ci_high`],
        )],
        ["Distance p", formatPValue(row[`${config.prefix}_wasserstein_p_value`])],
        ["Distance q", formatPValue(row[`${config.prefix}_wasserstein_q_value`])],
      );
    }
    if (state.measure === "all") {
      values.push(["Existing links", Number(row.all_link_existing_count).toLocaleString()]);
      values.push(["Repeat share", formatProbability(row.all_link_repeat_probability)]);
      values.push(["Repeat-share Δ", formatProbability(row.all_link_repeat_risk_difference)]);
      values.push(["Repeat Δ 95% CI", formatInterval(
        row.all_link_repeat_risk_difference_ci_low,
        row.all_link_repeat_risk_difference_ci_high,
        formatProbability,
      )]);
    }
    for (const [label, value] of values) {
      const item = document.createElement("div");
      const term = document.createElement("span");
      term.textContent = label;
      const number = document.createElement("strong");
      number.textContent = value;
      item.append(term, number);
      stats.append(item);
    }
    const chartTitle = document.createElement("h3");
    chartTitle.textContent = "Share of papers by year";
    const yearlyDistanceTitle = document.createElement("h3");
    yearlyDistanceTitle.textContent = config.semantic
      ? "Mean semantic distance by year"
      : "Mean distance by year";
    const distanceTitle = document.createElement("h3");
    distanceTitle.textContent = config.authorDiversity
      ? "Author primary-topic distribution"
      : config.diversity
        ? "Paper-topic distribution"
      : config.semantic
        ? "Semantic-distance distribution"
        : "Connected-distance distribution";
    const legend = document.createElement("div");
    legend.className = "distance-legend";
    legend.innerHTML = config.semantic
      ? `<span><i class="cluster-swatch"></i>Selected cluster</span><span><i class="baseline-swatch"></i>${baselineLabel()}</span>`
      : `<span><i class="cluster-swatch"></i>Selected cluster</span><span><i class="baseline-swatch"></i>${baselineLabel()}</span>`;
    const detailNodes = [
      eyebrow,
      heading,
      type,
      stats,
      chartTitle,
      lineChart(row.yearly || []),
    ];
    if (!config.diversity) {
      detailNodes.push(
        yearlyDistanceTitle,
        distanceYearChart(row[config.yearly] || []),
      );
    }
    detailNodes.push(distanceTitle);
    if (!config.diversity) detailNodes.push(legend);
    detailNodes.push(distributionChart(row));
    details.append(...detailNodes);
    if (row.keywords && row.keywords.length) {
      const wordTitle = document.createElement("h3");
      wordTitle.textContent = "Included keywords";
      const words = document.createElement("div");
      words.className = "tags";
      for (const keyword of row.keywords) {
        const tag = document.createElement("span");
        tag.textContent = keyword;
        words.append(tag);
      }
      details.append(wordTitle, words);
    }
    details.append(methodologyPanel(state.measure));
    for (const point of svg.querySelectorAll(".scatter-point")) {
      const active = Number(point.dataset.clusterId) === row.cluster_id;
      point.classList.toggle("active", active);
      point.setAttribute("r", active ? "7" : "5");
    }
  }

  function nearest(clientX, clientY) {
    const rect = svg.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    let result = null;
    let best = 18 * 18;
    for (const point of state.points) {
      const distance = (point.x - x) ** 2 + (point.y - y) ** 2;
      if (distance < best) {
        best = distance;
        result = point;
      }
    }
    return result;
  }

  function observableRows() {
    const config = measures[state.measure];
    return state.rows.filter(
      (row) => row.paper_count > 0
        && Number.isFinite(row[config.distance])
        && (!config.xDistance || Number.isFinite(row[config.xDistance])),
    );
  }

  function draw() {
    const rect = svg.getBoundingClientRect();
    const width = Math.max(520, rect.width);
    const height = Math.max(380, rect.height);
    const padding = { left: 72, right: 28, top: 26, bottom: 62 };
    const config = measures[state.measure];
    const comparison = Boolean(config.diversityComparison);
    const domain = observableRows();
    const rows = filteredRows().filter(
      (row) => row.paper_count > 0
        && Number.isFinite(row[config.distance])
        && (!config.xDistance || Number.isFinite(row[config.xDistance])),
    );
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.replaceChildren();
    state.points = [];
    if (!domain.length) {
      const message = svgNode("text", { class: "plot-empty", x: width / 2, y: height / 2 });
      message.textContent = `No clusters have ${config.description} observations`;
      svg.append(message);
      return;
    }

    const minLog = comparison
      ? 0
      : Math.log10(Math.min(...domain.map((row) => row.paper_count)));
    const maxLog = comparison
      ? 1
      : Math.log10(Math.max(...domain.map((row) => row.paper_count)));
    const minDistance = comparison
      ? 0
      : Math.min(0, ...domain.map((row) => row[config.distance]));
    const maxDistance = comparison
      ? 1
      : Math.max(...domain.map((row) => row[config.distance]));
    const x = comparison
      ? (value) => padding.left + Number(value) * (width - padding.left - padding.right)
      : (count) => padding.left + ((Math.log10(count) - minLog) / Math.max(0.1, maxLog - minLog)) * (width - padding.left - padding.right);
    const verticalSpan = comparison
      ? 1
      : config.diversity
      ? Math.max(0.05, maxDistance - minDistance)
      : Math.max(1, maxDistance - minDistance);
    const y = (distance) => height - padding.bottom - ((distance - minDistance) / verticalSpan) * (height - padding.top - padding.bottom);

    const grid = svgNode("g", { class: "plot-grid" });
    const axes = svgNode("g");
    for (let index = 0; index <= 5; index += 1) {
      const value = minDistance + ((maxDistance - minDistance) * index) / 5;
      const py = y(value);
      grid.append(svgNode("line", { x1: padding.left, y1: py, x2: width - padding.right, y2: py }));
      const label = svgNode("text", { class: "plot-tick", x: padding.left - 10, y: py + 4, "text-anchor": "end" });
      label.textContent = config.diversity ? formatDiversity(value) : formatDistance(value);
      axes.append(label);
    }
    const xTicks = comparison
      ? Array.from({ length: 6 }, (_, index) => index / 5)
      : [];
    if (!comparison) {
      const firstPower = Math.ceil(minLog);
      const lastPower = Math.floor(maxLog);
      for (let power = firstPower; power <= lastPower; power += 1) xTicks.push(10 ** power);
      const smallest = Math.min(...domain.map((row) => row.paper_count));
      if (!xTicks.length || xTicks[0] > smallest * 2) xTicks.unshift(smallest);
    }
    for (const value of xTicks) {
      const px = x(value);
      grid.append(svgNode("line", { x1: px, y1: padding.top, x2: px, y2: height - padding.bottom }));
      const label = svgNode("text", { class: "plot-tick", x: px, y: height - padding.bottom + 22, "text-anchor": "middle" });
      label.textContent = comparison ? formatDiversity(value) : Number(value).toLocaleString();
      axes.append(label);
    }
    axes.append(
      svgNode("path", {
        class: "axis",
        d: `M ${padding.left} ${padding.top} V ${height - padding.bottom} H ${width - padding.right}`,
      }),
    );
    const xLabel = svgNode("text", { class: "plot-axis-title", x: (padding.left + width - padding.right) / 2, y: height - 14, "text-anchor": "middle" });
    xLabel.textContent = config.xLabel || "Cluster size (papers, log scale)";
    const yLabel = svgNode("text", {
      class: "plot-axis-title",
      x: 18,
      y: (padding.top + height - padding.bottom) / 2,
      transform: `rotate(-90 18 ${(padding.top + height - padding.bottom) / 2})`,
      "text-anchor": "middle",
    });
    yLabel.textContent = config.label;
    axes.append(xLabel, yLabel);
    if (comparison) {
      axes.append(
        svgNode("line", {
          class: "comparison-line",
          x1: x(0),
          y1: y(0),
          x2: x(1),
          y2: y(1),
        }),
      );
    }

    const marks = svgNode("g");
    if (!rows.length) {
      const message = svgNode("text", { class: "plot-empty", x: width / 2, y: height / 2 });
      message.textContent = `No matching clusters have ${config.description} observations`;
      marks.append(message);
    }
    for (const row of rows) {
      const xValue = config.xDistance ? row[config.xDistance] : row.paper_count;
      const point = { row, x: x(xValue), y: y(row[config.distance]) };
      state.points.push(point);
      const circle = svgNode("circle", {
        class: `scatter-point${row === state.hovered ? " active" : ""}`,
        cx: point.x,
        cy: point.y,
        r: row === state.hovered ? 7 : 5,
        tabindex: 0,
        "data-cluster-id": row.cluster_id,
        "data-type": row.cluster_type || "",
        "aria-label": comparison
          ? `${row.label}, ${row.cluster_type || "unclassified"}: overall author diversity ${formatDiversity(xValue)}, ${config.label} ${formatDiversity(row[config.distance])}`
          : `${row.label}, ${row.cluster_type || "unclassified"}: ${Number(row.paper_count).toLocaleString()} papers, ${config.label} ${config.diversity ? formatDiversity(row[config.distance]) : formatDistance(row[config.distance])}`,
      });
      circle.addEventListener("focus", () => {
        show(row);
      });
      circle.addEventListener("pointerenter", () => {
        show(row);
      });
      marks.append(circle);
    }
    svg.append(grid, axes, marks);
  }

  function updateHover(event) {
    const point = nearest(event.clientX, event.clientY);
    if (!point) {
      tooltip.hidden = true;
      return;
    }
    if (point.row !== state.hovered) {
      show(point.row);
    }
    tooltip.hidden = false;
    tooltip.textContent = point.row.cluster_type
      ? `${point.row.label} · ${point.row.cluster_type}`
      : point.row.label;
    tooltip.style.left = `${point.x + 12}px`;
    tooltip.style.top = `${point.y - 10}px`;
  }

  svg.addEventListener("pointermove", updateHover);
  svg.addEventListener("pointerleave", () => {
    tooltip.hidden = true;
  });
  window.addEventListener("resize", draw);
  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      state.measure = tab.dataset.measure;
      for (const button of tabs) button.setAttribute("aria-pressed", String(button === tab));
      updateSummary();
      if (
        state.hovered
        && Number.isFinite(state.hovered[measures[state.measure].distance])
      ) {
        show(state.hovered);
      } else {
        clearSelection();
      }
      draw();
    });
  }
  search.addEventListener("input", () => {
    state.query = search.value.trim().toLocaleLowerCase();
    if (state.hovered && (!matchesQuery(state.hovered) || !matchesType(state.hovered))) clearSelection();
    updateSummary();
    draw();
  });
  for (const button of typeButtons) {
    button.addEventListener("click", () => {
      const pressed = button.getAttribute("aria-pressed") === "true";
      button.setAttribute("aria-pressed", String(!pressed));
      if (state.hovered && !matchesType(state.hovered)) clearSelection();
      updateSummary();
      draw();
    });
  }

  async function fetchCompressedJson(url) {
    if (typeof DecompressionStream === "undefined") {
      throw new Error("This browser does not support gzip decompression");
    }
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    if (!response.body) throw new Error("The data response has no body");
    const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
    return new Response(stream).json();
  }

  fetchCompressedJson("data.json.gz")
    .then((data) => {
      const rowsByCluster = new Map();
      for (const row of data.link_distances || []) {
        rowsByCluster.set(Number(row.cluster_id), { ...row });
      }
      for (const row of data.cluster_topics || []) {
        const clusterId = Number(row.cluster_id);
        rowsByCluster.set(clusterId, {
          ...(rowsByCluster.get(clusterId) || {}),
          ...row,
        });
      }
      state.rows = [...rowsByCluster.values()];
      if (!state.rows.length) {
        summary.textContent = "No link or cluster-topic summary was supplied.";
        draw();
        return;
      }
      if (!(data.link_distances || []).length && (data.cluster_topics || []).length) {
        state.measure = "diversity";
        for (const button of tabs) {
          button.setAttribute(
            "aria-pressed",
            String(button.dataset.measure === state.measure),
          );
        }
      }
      updateSummary();
      clearSelection();
      draw();
    })
    .catch((error) => {
      summary.textContent = `Unable to load cluster measures: ${error.message}`;
    });
})();
