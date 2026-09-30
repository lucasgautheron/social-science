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

  fetch("data.json")
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
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
        list.append(row);
      }
    })
    .catch((error) => {
      summary.textContent = `Unable to load clusters: ${error.message}`;
    });
})();
