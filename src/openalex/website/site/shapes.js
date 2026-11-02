(() => {
  "use strict";

  const summary = document.querySelector("#summary");
  const scatter = document.querySelector("#shape-scatter");
  const scatterTooltip = document.querySelector("#shape-tooltip");
  const legend = document.querySelector("#shape-legend");
  const gallery = document.querySelector("#shape-gallery");
  const NS = "http://www.w3.org/2000/svg";
  const CLASSES = [
    "stable",
    "steady rise",
    "accelerating rise",
    "saturating rise",
    "rise then fall",
    "fall then rise",
    "steady decline",
    "accelerating decline",
    "flattening decline",
    "shock: transient",
    "shock: lasting",
    "abrupt drop",
  ];
  const COLORS = {
    stable: "#9e9e9e",
    "steady rise": "#2e7d32",
    "accelerating rise": "#66bb6a",
    "saturating rise": "#a5d6a7",
    "rise then fall": "#8e24aa",
    "fall then rise": "#ce93d8",
    "steady decline": "#1565c0",
    "accelerating decline": "#42a5f5",
    "flattening decline": "#90caf9",
    "shock: transient": "#ef6c00",
    "shock: lasting": "#c62828",
    "abrupt drop": "#5d4037",
  };

  function svgNode(name, attributes = {}) {
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    return node;
  }

  function isAbrupt(name) {
    return name.startsWith("shock") || name.startsWith("abrupt");
  }

  function scaledActivity(activity) {
    const values = activity.map(Number);
    const mean = values.reduce((total, value) => total + value, 0) / values.length;
    const centered = values.map((value) => value - mean);
    const variance = centered.reduce((total, value) => total + value * value, 0) / centered.length;
    const scale = Math.max(Math.sqrt(variance), 0.5);
    return centered.map((value) => value / scale);
  }

  function seriesRange(activity) {
    const values = activity.map(Number);
    return Math.max(...values) - Math.min(...values);
  }

  function clusterText(row) {
    const keywords = row.keywords || [];
    const extra = keywords.slice(1).join(", ");
    if (!extra) return row.label;
    const text = `${row.label} · ${extra}`;
    return text.length > 96 ? `${row.label} · ${extra.slice(0, 72)}…` : text;
  }

  function median(values) {
    const sorted = values.filter((value) => value != null).sort((left, right) => left - right);
    if (!sorted.length) return null;
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  function pathFrom(points) {
    return points
      .map((point, index) => `${index ? "L" : "M"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
      .join(" ");
  }

  function pointerPosition(node, event) {
    const point = node.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const matrix = node.getScreenCTM();
    if (!matrix) return null;
    return point.matrixTransform(matrix.inverse());
  }

  function valueAt(points, x) {
    if (!points.length || x < points[0].x || x > points[points.length - 1].x) return null;
    let high = 1;
    while (high < points.length - 1 && points[high].x < x) high += 1;
    const left = points[high - 1];
    const right = points[Math.min(high, points.length - 1)];
    if (right.x === left.x) return left.y;
    const weight = (x - left.x) / (right.x - left.x);
    return left.y + weight * (right.y - left.y);
  }

  function placeTooltip(tooltip, frame, event, text) {
    const bounds = frame.getBoundingClientRect();
    tooltip.hidden = false;
    tooltip.textContent = text;
    const left = event.clientX - bounds.left + 12;
    const top = event.clientY - bounds.top - 30;
    tooltip.style.left = `${Math.max(4, Math.min(left, bounds.width - 8))}px`;
    tooltip.style.top = `${Math.max(4, top)}px`;
  }

  function drawScatter(series, years) {
    const width = 920;
    const height = 560;
    const padding = { left: 64, right: 24, top: 28, bottom: 58 };
    scatter.setAttribute("viewBox", `0 0 ${width} ${height}`);
    scatter.replaceChildren();
    const trends = series.map((row) => Number(row.trend));
    const bursts = series.map((row) => Number(row.burstiness));
    const xMin = Math.min(...trends);
    const xMax = Math.max(...trends);
    const yMin = Math.min(0, ...bursts);
    const yMax = Math.max(0.22, ...bursts);
    const xPad = Math.max((xMax - xMin) * 0.06, 0.05);
    const yPad = Math.max((yMax - yMin) * 0.08, 0.02);
    const x0 = xMin - xPad;
    const x1 = xMax + xPad;
    const y0 = yMin - yPad * 0.2;
    const y1 = yMax + yPad;
    const x = (value) => padding.left + ((value - x0) / (x1 - x0)) * (width - padding.left - padding.right);
    const y = (value) => height - padding.bottom - ((value - y0) / (y1 - y0)) * (height - padding.top - padding.bottom);
    scatter.append(
      svgNode("path", {
        class: "axis",
        d: `M ${padding.left} ${padding.top} V ${height - padding.bottom} H ${width - padding.right}`,
      }),
    );
    if (y0 <= 0.15 && 0.15 <= y1) {
      scatter.append(
        svgNode("line", {
          class: "shape-threshold",
          x1: padding.left,
          x2: width - padding.right,
          y1: y(0.15),
          y2: y(0.15),
        }),
      );
      const note = svgNode("text", { class: "shape-note", x: padding.left + 6, y: y(0.15) - 6 });
      note.textContent = "abrupt threshold";
      scatter.append(note);
    }
    const xLabel = svgNode("text", {
      class: "plot-axis-title",
      x: (padding.left + width - padding.right) / 2,
      y: height - 16,
      "text-anchor": "middle",
    });
    xLabel.textContent = `net trend ${years[0]}–${years[years.length - 1]} (log fold-change of the smooth component)`;
    const yLabel = svgNode("text", {
      class: "plot-axis-title",
      x: 16,
      y: (padding.top + height - padding.bottom) / 2,
      transform: `rotate(-90 16 ${(padding.top + height - padding.bottom) / 2})`,
      "text-anchor": "middle",
    });
    yLabel.textContent = "burstiness (variance share explained by a shock beyond the trend)";
    scatter.append(xLabel, yLabel);
    for (const row of series) {
      const point = svgNode("circle", {
        class: "shape-point",
        cx: x(Number(row.trend)),
        cy: y(Number(row.burstiness)),
        r: 4.5,
        fill: COLORS[row.sub] || "#9e9e9e",
        "data-type": row.sub,
        tabindex: 0,
      });
      const show = (event) => placeTooltip(
        scatterTooltip,
        scatter.parentElement,
        event,
        `${clusterText(row)} · ${row.sub}`,
      );
      point.addEventListener("pointerenter", show);
      point.addEventListener("pointermove", show);
      point.addEventListener("focus", (event) => {
        const bounds = point.getBoundingClientRect();
        placeTooltip(
          scatterTooltip,
          scatter.parentElement,
          { clientX: bounds.right, clientY: bounds.top },
          `${clusterText(row)} · ${row.sub}`,
        );
      });
      point.addEventListener("pointerleave", () => {
        scatterTooltip.hidden = true;
      });
      point.addEventListener("blur", () => {
        scatterTooltip.hidden = true;
      });
      scatter.append(point);
    }
    legend.replaceChildren();
    for (const name of CLASSES) {
      const count = series.filter((row) => row.sub === name).length;
      const item = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.style.background = COLORS[name];
      item.append(swatch, document.createTextNode(`${name} (${count})`));
      legend.append(item);
    }
  }

  function drawPanel(name, members, years, yLimit) {
    const panel = document.createElement("section");
    panel.className = "card shape-panel";
    const heading = document.createElement("h3");
    heading.textContent = `${name} (n=${members.length})`;
    const examples = document.createElement("p");
    examples.className = "examples";
    const named = [...members].sort((left, right) => seriesRange(right.activity) - seriesRange(left.activity));
    examples.textContent = named.slice(0, 4).map((row) => row.label).join(", ") || "No series";
    const frame = document.createElement("div");
    frame.className = "shape-frame";
    const chart = svgNode("svg", {
      class: "shape-chart",
      viewBox: "0 0 280 168",
      role: "img",
      "aria-label": `${name} series`,
    });
    const tooltip = document.createElement("div");
    tooltip.className = "tooltip";
    tooltip.hidden = true;
    frame.append(chart, tooltip);
    panel.append(heading, examples, frame);
    if (!members.length) return panel;

    const width = 280;
    const height = 168;
    const padding = { left: 28, right: 10, top: 10, bottom: 26 };
    const y = (value) => height - padding.bottom - ((value + yLimit) / (2 * yLimit)) * (height - padding.top - padding.bottom);
    const abrupt = isAbrupt(name);
    const span = years.length;
    let xValues;
    let curves;
    if (abrupt) {
      const relative = [];
      for (let offset = -(span - 1); offset < span; offset += 1) relative.push(offset);
      xValues = relative;
      curves = members.map((row) => {
        const onset = years.indexOf(Number(row.onset));
        const aligned = new Array(relative.length).fill(null);
        const start = span - 1 - (onset < 0 ? 0 : onset);
        row.scaled.forEach((value, index) => {
          const position = start + index;
          if (position >= 0 && position < aligned.length) aligned[position] = value;
        });
        return { row, values: aligned };
      });
    } else {
      xValues = years;
      curves = members.map((row) => ({ row, values: row.scaled }));
    }
    const xDomain = abrupt ? [-8, 10] : [xValues[0], xValues[xValues.length - 1]];
    const x = (value) => padding.left + ((value - xDomain[0]) / (xDomain[1] - xDomain[0] || 1)) * (width - padding.left - padding.right);
    const clipId = `shape-clip-${name.replace(/[^a-z]+/gi, "-")}`;
    const clip = svgNode("clipPath", { id: clipId });
    clip.append(svgNode("rect", {
      x: padding.left,
      y: padding.top,
      width: width - padding.left - padding.right,
      height: height - padding.top - padding.bottom,
    }));
    const defs = svgNode("defs");
    defs.append(clip);
    chart.append(defs);
    const curvesLayer = svgNode("g", { "clip-path": `url(#${clipId})` });
    if (!abrupt && years.some((year) => year >= 2023) && years.some((year) => year < 2023)) {
      chart.append(svgNode("rect", {
        class: "shape-band",
        x: x(2022.5),
        y: padding.top,
        width: Math.max(0, x(2025.5) - x(2022.5)),
        height: height - padding.top - padding.bottom,
      }));
    }
    chart.append(svgNode("path", {
      class: "axis",
      d: `M ${padding.left} ${padding.top} V ${height - padding.bottom} H ${width - padding.right}`,
    }));
    const drawn = [];
    for (const curve of curves) {
      const points = [];
      curve.values.forEach((value, index) => {
        if (value == null) return;
        points.push({ x: x(xValues[index]), y: y(value) });
      });
      if (points.length < 2) continue;
      const path = svgNode("path", {
        class: "shape-curve member",
        d: pathFrom(points),
        stroke: COLORS[name],
      });
      curvesLayer.append(path);
      drawn.push({ row: curve.row, points, path });
    }
    const counts = xValues.map((_, index) => curves.filter((curve) => curve.values[index] != null).length);
    const required = Math.max(3, Math.floor(members.length / 4));
    const medianPoints = [];
    xValues.forEach((value, index) => {
      if (counts[index] < required) return;
      const center = median(curves.map((curve) => curve.values[index]));
      if (center == null) return;
      medianPoints.push({ x: x(value), y: y(center) });
    });
    if (medianPoints.length > 1) {
      curvesLayer.append(svgNode("path", { class: "shape-curve median", d: pathFrom(medianPoints) }));
    }
    chart.append(curvesLayer);
    if (abrupt) {
      chart.append(svgNode("line", {
        class: "shape-onset",
        x1: x(0),
        x2: x(0),
        y1: padding.top,
        y2: height - padding.bottom,
      }));
    }
    const tickValues = abrupt ? [-8, 0, 8] : [years[0], years[Math.floor((years.length - 1) / 2)], years[years.length - 1]];
    for (const tick of tickValues) {
      if (tick < xValues[0] || tick > xValues[xValues.length - 1]) continue;
      const label = svgNode("text", {
        class: "plot-tick",
        x: x(tick),
        y: height - 8,
        "text-anchor": "middle",
      });
      label.textContent = String(tick);
      chart.append(label);
    }
    let hovered = null;
    chart.addEventListener("pointermove", (event) => {
      const pointer = pointerPosition(chart, event);
      if (!pointer) return;
      let best = null;
      let bestDistance = 18;
      for (const curve of drawn) {
        const level = valueAt(curve.points, pointer.x);
        if (level == null) continue;
        const distance = Math.abs(level - pointer.y);
        if (distance < bestDistance) {
          best = curve;
          bestDistance = distance;
        }
      }
      if (!best) {
        if (hovered) hovered.path.classList.remove("hover");
        hovered = null;
        tooltip.hidden = true;
        return;
      }
      if (hovered && hovered !== best) hovered.path.classList.remove("hover");
      hovered = best;
      best.path.classList.add("hover");
      placeTooltip(tooltip, frame, event, `${clusterText(best.row)} · ${name}`);
    });
    chart.addEventListener("pointerleave", () => {
      if (hovered) hovered.path.classList.remove("hover");
      hovered = null;
      tooltip.hidden = true;
    });
    return panel;
  }

  function render(shapes) {
    if (!shapes || !shapes.series || !shapes.series.length) {
      summary.textContent = "Shape classes were not included in this website build.";
      scatter.closest("section").hidden = true;
      gallery.closest("section").hidden = true;
      return;
    }
    const years = shapes.years.map(Number);
    const series = shapes.series.map((row) => ({ ...row, scaled: scaledActivity(row.activity) }));
    const first = years[0];
    const last = years[years.length - 1];
    summary.textContent = `${series.length.toLocaleString()} keyword series · topic-adjusted log activity, ${first}–${last}`;
    drawScatter(series, years);
    let yLimit = 0.5;
    for (const row of series) {
      for (const value of row.scaled) yLimit = Math.max(yLimit, Math.abs(value));
    }
    gallery.replaceChildren();
    for (const name of CLASSES) {
      gallery.append(drawPanel(name, series.filter((row) => row.sub === name), years, yLimit));
    }
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
    .then((payload) => render(payload.shapes))
    .catch((error) => {
      summary.textContent = `Could not load shape classes: ${error.message}`;
    });
})();
