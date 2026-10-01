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
      label: "Average first-link distance",
      description: "first coauthorship links",
    },
    all: {
      distance: "average_all_link_distance",
      connected: "all_link_connected_count",
      disconnected: "all_link_disconnected_count",
      distribution: "all_link_distance_distribution",
      baselineDistribution: "all_link_baseline_distance_distribution",
      prefix: "all_link",
      yearly: "all_link_distance_by_year",
      label: "Average distance across all paper links",
      description: "all coauthor-pair observations on cluster papers",
    },
  };

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
        <p>Each visible point is one matching cluster. Hover it to see its temporal curves and distance distribution.</p>
      </div>
    `;
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

  function distributionChart(row) {
    const config = measures[state.measure];
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

  function distributionMap(pairs, measure) {
    const counts = new Map();
    for (const [distance, count] of pairs || []) {
      const shown = measure === "all" && Number(distance) === 0 ? 1 : Number(distance);
      counts.set(shown, (counts.get(shown) || 0) + Number(count));
    }
    return counts;
  }

  function baselineLabel() {
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
    const values = [
      ["Cluster size", `${Number(row.paper_count).toLocaleString()} papers`],
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
    ];
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
    yearlyDistanceTitle.textContent = "Mean distance by year";
    const distanceTitle = document.createElement("h3");
    distanceTitle.textContent = "Connected-distance distribution";
    const legend = document.createElement("div");
    legend.className = "distance-legend";
    legend.innerHTML = `<span><i class="cluster-swatch"></i>Selected cluster</span><span><i class="baseline-swatch"></i>${baselineLabel()}</span>`;
    details.append(
      eyebrow,
      heading,
      type,
      stats,
      chartTitle,
      lineChart(row.yearly || []),
      yearlyDistanceTitle,
      distanceYearChart(row[config.yearly] || []),
      distanceTitle,
      legend,
      distributionChart(row),
    );
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
    return state.rows.filter((row) => row.paper_count > 0 && Number.isFinite(row[config.distance]));
  }

  function draw() {
    const rect = svg.getBoundingClientRect();
    const width = Math.max(520, rect.width);
    const height = Math.max(380, rect.height);
    const padding = { left: 72, right: 28, top: 26, bottom: 62 };
    const config = measures[state.measure];
    const domain = observableRows();
    const rows = filteredRows().filter((row) => row.paper_count > 0 && Number.isFinite(row[config.distance]));
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.replaceChildren();
    state.points = [];
    if (!domain.length) {
      const message = svgNode("text", { class: "plot-empty", x: width / 2, y: height / 2 });
      message.textContent = "No clusters have connected distance observations";
      svg.append(message);
      return;
    }

    const minLog = Math.log10(Math.min(...domain.map((row) => row.paper_count)));
    const maxLog = Math.log10(Math.max(...domain.map((row) => row.paper_count)));
    const minDistance = Math.min(0, ...domain.map((row) => row[config.distance]));
    const maxDistance = Math.max(...domain.map((row) => row[config.distance]));
    const x = (count) => padding.left + ((Math.log10(count) - minLog) / Math.max(0.1, maxLog - minLog)) * (width - padding.left - padding.right);
    const y = (distance) => height - padding.bottom - ((distance - minDistance) / Math.max(1, maxDistance - minDistance)) * (height - padding.top - padding.bottom);

    const grid = svgNode("g", { class: "plot-grid" });
    const axes = svgNode("g");
    for (let index = 0; index <= 5; index += 1) {
      const value = minDistance + ((maxDistance - minDistance) * index) / 5;
      const py = y(value);
      grid.append(svgNode("line", { x1: padding.left, y1: py, x2: width - padding.right, y2: py }));
      const label = svgNode("text", { class: "plot-tick", x: padding.left - 10, y: py + 4, "text-anchor": "end" });
      label.textContent = formatDistance(value);
      axes.append(label);
    }
    const firstPower = Math.ceil(minLog);
    const lastPower = Math.floor(maxLog);
    const xTicks = [];
    for (let power = firstPower; power <= lastPower; power += 1) xTicks.push(10 ** power);
    const smallest = Math.min(...domain.map((row) => row.paper_count));
    if (!xTicks.length || xTicks[0] > smallest * 2) xTicks.unshift(smallest);
    for (const value of xTicks) {
      const px = x(value);
      grid.append(svgNode("line", { x1: px, y1: padding.top, x2: px, y2: height - padding.bottom }));
      const label = svgNode("text", { class: "plot-tick", x: px, y: height - padding.bottom + 22, "text-anchor": "middle" });
      label.textContent = Number(value).toLocaleString();
      axes.append(label);
    }
    axes.append(
      svgNode("path", {
        class: "axis",
        d: `M ${padding.left} ${padding.top} V ${height - padding.bottom} H ${width - padding.right}`,
      }),
    );
    const xLabel = svgNode("text", { class: "plot-axis-title", x: (padding.left + width - padding.right) / 2, y: height - 14, "text-anchor": "middle" });
    xLabel.textContent = "Cluster size (papers, log scale)";
    const yLabel = svgNode("text", {
      class: "plot-axis-title",
      x: 18,
      y: (padding.top + height - padding.bottom) / 2,
      transform: `rotate(-90 18 ${(padding.top + height - padding.bottom) / 2})`,
      "text-anchor": "middle",
    });
    yLabel.textContent = config.label;
    axes.append(xLabel, yLabel);

    const marks = svgNode("g");
    if (!rows.length) {
      const message = svgNode("text", { class: "plot-empty", x: width / 2, y: height / 2 });
      message.textContent = "No matching clusters have connected distance observations";
      marks.append(message);
    }
    for (const row of rows) {
      const point = { row, x: x(row.paper_count), y: y(row[config.distance]) };
      state.points.push(point);
      const circle = svgNode("circle", {
        class: `scatter-point${row === state.hovered ? " active" : ""}`,
        cx: point.x,
        cy: point.y,
        r: row === state.hovered ? 7 : 5,
        tabindex: 0,
        "data-cluster-id": row.cluster_id,
        "data-type": row.cluster_type || "",
        "aria-label": `${row.label}, ${row.cluster_type || "unclassified"}: ${Number(row.paper_count).toLocaleString()} papers, mean distance ${formatDistance(row[config.distance])}`,
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
      if (state.hovered) show(state.hovered);
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

  fetch("data.json")
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      if (!data.link_distances) {
        summary.textContent = "No link summary was supplied. Pass --new-link-visualizations-dir when building the site.";
        draw();
        return;
      }
      state.rows = data.link_distances;
      updateSummary();
      draw();
    })
    .catch((error) => {
      summary.textContent = `Unable to load link distances: ${error.message}`;
    });
})();
