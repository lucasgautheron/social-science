(() => {
  "use strict";

  const summary = document.querySelector("#summary");
  const trends = document.querySelector("#trends");
  const NS = "http://www.w3.org/2000/svg";
  const measures = [
    {
      key: "collaborator_topic_count",
      expected: "expected_collaborator_topic_count",
      title: "Collaborators' primary topics",
      format: (value) => Number(value).toFixed(1),
    },
    {
      key: "out_of_topic_share",
      expected: "expected_out_of_topic_share",
      title: "Out-of-topic collaborators",
      format: (value) => `${(Number(value) * 100).toFixed(1)}%`,
    },
    {
      key: "rao_stirling_diversity",
      expected: "expected_rao_stirling_diversity",
      title: "Collaborator topic diversity",
      format: (value) => Number(value).toFixed(3),
    },
  ];

  function svgNode(name, attributes = {}) {
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    return node;
  }

  function formatCount(value) {
    return Number(value || 0).toLocaleString();
  }

  function chart(data, measure) {
    const width = 640;
    const height = 280;
    const padding = { top: 28, right: 108, bottom: 36, left: 52 };
    const observed = data.years.filter((row) => row[measure.key] != null);
    const expected = data.years.filter((row) => row[measure.expected] != null);
    const values = [
      ...observed.map((row) => Number(row[measure.key])),
      ...expected.map((row) => Number(row[measure.expected])),
    ];
    const years = data.years.map((row) => Number(row.year));
    const yearMin = Math.min(...years);
    const yearMax = Math.max(...years);
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      min = 0;
      max = 1;
    }
    if (min === max) {
      min -= 1;
      max += 1;
    }
    const pad = (max - min) * 0.14;
    min -= pad;
    max += pad;
    const x = (year) => (
      padding.left
      + ((year - yearMin) / Math.max(1, yearMax - yearMin))
      * (width - padding.left - padding.right)
    );
    const y = (value) => (
      padding.top
      + (1 - (value - min) / (max - min)) * (height - padding.top - padding.bottom)
    );
    const svg = svgNode("svg", {
      class: "trend-chart",
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
    });
    svg.append(svgNode("line", {
      class: "trend-axis",
      x1: padding.left,
      y1: height - padding.bottom,
      x2: width - padding.right,
      y2: height - padding.bottom,
    }));
    const breakYear = Number(data.break_year);
    if (breakYear >= yearMin && breakYear <= yearMax) {
      const breakX = x(breakYear);
      svg.append(svgNode("line", {
        class: "trend-break",
        x1: breakX,
        y1: padding.top,
        x2: breakX,
        y2: height - padding.bottom,
      }));
      const marker = svgNode("text", {
        class: "trend-label break",
        x: breakX + 4,
        y: padding.top - 8,
      });
      marker.textContent = String(breakYear);
      svg.append(marker);
    }
    if (expected.length) {
      svg.append(svgNode("polyline", {
        class: "trend-expected",
        points: expected.map((row) => `${x(row.year)},${y(row[measure.expected])}`).join(" "),
      }));
    }
    if (observed.length) {
      svg.append(svgNode("polyline", {
        class: "trend-observed",
        points: observed.map((row) => `${x(row.year)},${y(row[measure.key])}`).join(" "),
      }));
    }
    for (const row of observed) {
      const point = svgNode("circle", {
        class: "trend-point",
        cx: x(row.year),
        cy: y(row[measure.key]),
        r: 3.5,
      });
      const title = svgNode("title");
      title.textContent = [
        `${row.year}: ${measure.format(row[measure.key])}`,
        `${formatCount(row.eligible_authors)} eligible authors`,
        `${formatCount(row.authors_without_primary_topic)} without a primary topic`,
        `${formatCount(row.authors_without_classified_collaborator)} without a classified collaborator`,
      ].join("; ");
      point.append(title);
      svg.append(point);
    }
    const start = svgNode("text", {
      class: "axis-label",
      x: padding.left,
      y: height - 12,
    });
    start.textContent = String(yearMin);
    const finish = svgNode("text", {
      class: "axis-label end",
      x: width - padding.right,
      y: height - 12,
    });
    finish.textContent = String(yearMax);
    svg.append(start, finish);
    const last = observed.at(-1);
    const lastExpected = expected.at(-1);
    if (last) {
      const label = svgNode("text", {
        class: "trend-label observed",
        x: x(last.year) + 8,
        y: y(last[measure.key]) - 6,
      });
      label.textContent = `Observed ${measure.format(last[measure.key])}`;
      svg.append(label);
    }
    if (lastExpected) {
      const label = svgNode("text", {
        class: "trend-label expected",
        x: x(lastExpected.year) + 8,
        y: y(lastExpected[measure.expected]) + 14,
      });
      label.textContent = `Expected ${measure.format(lastExpected[measure.expected])}`;
      svg.append(label);
    }
    const described = observed
      .map((row) => `${row.year} ${measure.format(row[measure.key])}`)
      .join(", ");
    svg.setAttribute("aria-label", `${measure.title}: ${described}`);
    return svg;
  }

  function render(data) {
    trends.replaceChildren();
    if (!data || !Array.isArray(data.years) || !data.years.length) {
      summary.textContent = "Global topic trends were not included in this website build.";
      return;
    }
    const first = data.years[0];
    const last = data.years.at(-1);
    summary.textContent = [
      `${first.year}–${last.year}.`,
      `Eligible authors in ${last.year}: ${formatCount(last.eligible_authors)}.`,
      `Without a leave-year-out primary topic: ${formatCount(last.authors_without_primary_topic)}.`,
      `Without a classified collaborator: ${formatCount(last.authors_without_classified_collaborator)}.`,
    ].join(" ");
    for (const measure of measures) {
      const panel = document.createElement("section");
      panel.className = "card trend-panel";
      const title = document.createElement("h2");
      title.textContent = measure.title;
      panel.append(title, chart(data, measure));
      trends.append(panel);
    }
  }

  async function fetchCompressedJson(url) {
    if (typeof DecompressionStream === "undefined") {
      throw new Error("This browser does not support gzip decompression");
    }
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(stream).text());
  }

  fetchCompressedJson("data.json.gz")
    .then((payload) => render(payload.global_trends))
    .catch((error) => {
      summary.textContent = `Could not load global trends: ${error.message}`;
    });
})();
