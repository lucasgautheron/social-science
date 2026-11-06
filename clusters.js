(() => {
  "use strict";

  const list = document.querySelector("#clusters");
  const summary = document.querySelector("#summary");
  const NS = "http://www.w3.org/2000/svg";

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

  function orderedTopics(values, valueField = "papers") {
    return [...(values || [])].sort(
      (left, right) => Number(right[valueField]) - Number(left[valueField])
        || Number(left.topic_id) - Number(right.topic_id),
    );
  }

  function topicTopWord(item) {
    if (item.topic_id == null) return "Other";
    if (item.top_word) return String(item.top_word);
    const label = String(item.label || "").replace(/^-?\d+[_\s-]+/, "");
    return label.split("_").find((word) => word.length) || `T${item.topic_id}`;
  }

  function topicHistogram(values, valueField = "papers", unit = "papers") {
    const ordered = orderedTopics(values, valueField);
    const width = 480;
    const height = 180;
    const frame = document.createElement("div");
    frame.className = "topic-chart-frame";
    const tooltip = document.createElement("div");
    tooltip.className = "topic-tooltip";
    tooltip.hidden = true;
    const chart = svgNode("svg", {
      class: "line-chart cluster-topic-chart",
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
    });
    frame.append(chart, tooltip);
    if (!ordered.length) {
      const message = svgNode("text", { class: "axis-label", x: 8, y: 24 });
      message.textContent = "No topic assignments";
      chart.append(message);
      return frame;
    }
    const shown = ordered.slice(0, 10);
    if (ordered.length > shown.length) {
      shown.push({
        topic_id: null,
        label: `${ordered.length - shown.length} other topics`,
        [valueField]: ordered.slice(shown.length).reduce(
          (total, item) => total + Number(item[valueField]),
          0,
        ),
      });
    }
    const padding = { left: 40, right: 10, top: 12, bottom: 72 };
    const maximum = Math.max(...shown.map((item) => Number(item[valueField]))) || 1;
    const step = (width - padding.left - padding.right) / shown.length;
    const barWidth = Math.max(4, Math.min(22, step * 0.65));
    const baseline = height - padding.bottom;
    const y = (count) => baseline
      - (Number(count) / maximum) * (height - padding.top - padding.bottom);
    chart.append(svgNode("path", {
      class: "axis",
      d: `M ${padding.left} ${padding.top} V ${baseline} H ${width - padding.right}`,
    }));
    shown.forEach((item, index) => {
      const x = padding.left + step * (index + 0.5);
      const bar = svgNode("rect", {
        class: "topic-bar",
        x: x - barWidth / 2,
        y: y(item[valueField]),
        width: barWidth,
        height: Math.max(0, baseline - y(item[valueField])),
      });
      const hit = svgNode("rect", {
        class: "topic-hit",
        x: padding.left + step * index,
        y: padding.top,
        width: step,
        height: baseline - padding.top,
      });
      const detail = `${item.label}: ${Number(item[valueField]).toLocaleString(
        undefined,
        { maximumFractionDigits: 2 },
      )} ${unit}`;
      hit.addEventListener("pointerenter", (event) => showTopicTooltip(frame, tooltip, event, detail));
      hit.addEventListener("pointermove", (event) => showTopicTooltip(frame, tooltip, event, detail));
      hit.addEventListener("pointerleave", () => {
        tooltip.hidden = true;
      });
      const label = svgNode("text", {
        class: "axis-label",
        x,
        y: baseline + 12,
        "text-anchor": "end",
        transform: `rotate(-45 ${x} ${baseline + 12})`,
      });
      label.textContent = topicTopWord(item);
      chart.append(bar, hit, label);
    });
    const maximumLabel = svgNode("text", {
      class: "axis-label",
      x: 3,
      y: padding.top + 4,
    });
    maximumLabel.textContent = Number(maximum).toLocaleString(
      undefined,
      { maximumFractionDigits: 2 },
    );
    chart.append(maximumLabel);
    chart.setAttribute(
      "aria-label",
      `Topics ordered by frequency: ${ordered.map(
        (item) => `${item.label} ${item[valueField]}`,
      ).join(", ")}`,
    );
    return frame;
  }

  function showTopicTooltip(frame, tooltip, event, text) {
    const bounds = frame.getBoundingClientRect();
    tooltip.hidden = false;
    tooltip.textContent = text;
    tooltip.style.left = `${event.clientX - bounds.left + 12}px`;
    tooltip.style.top = `${event.clientY - bounds.top - 28}px`;
  }

  function lineChart(values) {
    const chart = svgNode("svg", {
      class: "line-chart cluster-chart",
      viewBox: "0 0 360 110",
      role: "img",
    });
    if (!values.length) {
      const message = svgNode("text", { class: "axis-label", x: 8, y: 24 });
      message.textContent = "No yearly observations";
      chart.append(message);
      return chart;
    }
    const padding = { left: 48, right: 12, top: 12, bottom: 24 };
    const years = values.map((item) => item.year);
    const shares = values.map((item) => Number(item.share) || 0);
    const minYear = Math.min(...years);
    const maxYear = Math.max(...years);
    const maxShare = Math.max(0, ...shares);
    const scale = maxShare > 0 ? maxShare : 1;
    const x = (year) => padding.left + ((year - minYear) / Math.max(1, maxYear - minYear)) * (360 - padding.left - padding.right);
    const y = (share) => 110 - padding.bottom - (share / scale) * (110 - padding.top - padding.bottom);
    chart.append(
      svgNode("path", {
        class: "axis",
        d: `M ${padding.left} ${padding.top} V ${110 - padding.bottom} H ${360 - padding.right}`,
      }),
    );
    const path = values
      .map((item, index) => `${index ? "L" : "M"} ${x(item.year)} ${y(Number(item.share) || 0)}`)
      .join(" ");
    chart.append(svgNode("path", { class: "series", d: path }));
    for (const item of values) {
      const point = svgNode("circle", {
        class: "series-point",
        cx: x(item.year),
        cy: y(Number(item.share) || 0),
        r: 3,
      });
      const title = svgNode("title");
      title.textContent = `${item.year}: ${formatPercent(item.share)} of documents (${Number(item.papers).toLocaleString()})`;
      point.append(title);
      chart.append(point);
    }
    const start = svgNode("text", { class: "axis-label", x: padding.left, y: 104 });
    start.textContent = String(minYear);
    const finish = svgNode("text", { class: "axis-label end", x: 360 - padding.right, y: 104 });
    finish.textContent = String(maxYear);
    const maximum = svgNode("text", { class: "axis-label", x: 4, y: padding.top + 4 });
    maximum.textContent = formatPercent(maxShare);
    chart.append(start, finish, maximum);
    const label = values.map((item) => `${item.year} ${formatPercent(item.share)}`).join(", ");
    chart.setAttribute("aria-label", `Share of documents by year: ${label}`);
    return chart;
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
      const clusters = data.cluster_list;
      if (!clusters) {
        summary.textContent = "No clusters were supplied. Pass --clusters-dir when building the site.";
        return;
      }
      const total = Number(data.meta.total_documents || 0);
      summary.textContent = `${clusters.length.toLocaleString()} clusters · ${total.toLocaleString()} documents`;
      const maximum = Math.max(1, ...clusters.map((item) => item.papers));
      for (const item of clusters) {
        const row = document.createElement("li");
        row.className = "keyword-row";
        const header = document.createElement("div");
        header.className = "keyword-header";
        const name = document.createElement("strong");
        name.textContent = item.keywords.join(" · ");
        const value = document.createElement("span");
        value.textContent = `${Number(item.papers).toLocaleString()} papers · ${formatPercent(item.share)}`;
        header.append(name, value);
        const track = document.createElement("div");
        track.className = "bar-track";
        const bar = document.createElement("div");
        bar.className = "bar";
        bar.style.width = `${(100 * item.papers) / maximum}%`;
        track.append(bar);
        row.append(header, track, lineChart(item.yearly || []));
        if (Array.isArray(item.topic_distribution)) {
          const topicHeader = document.createElement("div");
          topicHeader.className = "cluster-topic-header";
          const title = document.createElement("strong");
          title.textContent = "Paper topics";
          const diversity = document.createElement("span");
          diversity.textContent = item.rao_stirling_diversity == null
            ? "Rao–Stirling —"
            : `Rao–Stirling ${Number(item.rao_stirling_diversity).toFixed(3)}`;
          topicHeader.append(title, diversity);
          row.append(topicHeader, topicHistogram(item.topic_distribution));
        }
        if (Array.isArray(item.author_topic_distribution)) {
          const authorHeader = document.createElement("div");
          authorHeader.className = "cluster-topic-header";
          const title = document.createElement("strong");
          title.textContent = "Author primary topics";
          const diversity = document.createElement("span");
          diversity.textContent = item.author_rao_stirling_diversity == null
            ? "Rao–Stirling —"
            : `Rao–Stirling ${Number(item.author_rao_stirling_diversity).toFixed(3)}`;
          authorHeader.append(title, diversity);
          row.append(
            authorHeader,
            topicHistogram(
              item.author_topic_distribution,
              "weight",
              "fractional paper weight",
            ),
          );
        }
        list.append(row);
      }
    })
    .catch((error) => {
      summary.textContent = `Unable to load clusters: ${error.message}`;
    });
})();
