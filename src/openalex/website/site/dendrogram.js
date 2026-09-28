(() => {
  "use strict";

  const svg = document.querySelector("#tree");
  const details = document.querySelector("#details");
  const tooltip = document.querySelector("#tooltip");
  const summary = document.querySelector("#summary");
  const reset = document.querySelector("#reset");
  const NS = "http://www.w3.org/2000/svg";
  const state = {
    nodes: [],
    byId: new Map(),
    hovered: null,
    pinned: null,
    sx: 1,
    sy: 1,
    tx: 0,
    ty: 0,
    dragging: false,
    moved: 0,
    pointer: null,
  };

  function svgNode(name, attrs = {}) {
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    return node;
  }

  function bounds() {
    const rect = svg.getBoundingClientRect();
    return { width: Math.max(1, rect.width), height: Math.max(1, rect.height) };
  }

  function fit() {
    const { width, height } = bounds();
    if (!state.nodes.length) return;
    const xs = state.nodes.map((node) => node.x);
    const ys = state.nodes.map((node) => node.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    state.sx = (width - 96) / Math.max(maxX - minX, 0.05);
    state.sy = (height - 72) / Math.max(maxY - minY, 1);
    state.tx = 48 - minX * state.sx;
    state.ty = 36 - minY * state.sy;
    draw();
  }

  function position(node) {
    return { x: node.x * state.sx + state.tx, y: node.y * state.sy + state.ty };
  }

  function nearest(clientX, clientY) {
    const rect = svg.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    let result = null;
    let best = 14 * 14;
    for (const node of state.nodes) {
      const point = position(node);
      const distance = (point.x - x) ** 2 + (point.y - y) ** 2;
      if (distance < best) {
        best = distance;
        result = node;
      }
    }
    return result;
  }

  function draw() {
    const { width, height } = bounds();
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.replaceChildren();
    const group = svgNode("g", {
      transform: `translate(${state.tx} ${state.ty}) scale(${state.sx} ${state.sy})`,
    });
    const links = svgNode("g");
    const marks = svgNode("g");
    for (const node of state.nodes) {
      if (node.is_leaf) continue;
      const left = state.byId.get(node.left_id);
      const right = state.byId.get(node.right_id);
      if (!left || !right) continue;
      links.append(
        svgNode("path", {
          class: "tree-link",
          d: `M ${left.x} ${left.y} H ${node.x} V ${right.y} H ${right.x}`,
        }),
      );
    }
    for (const node of state.nodes) {
      const active = node === state.hovered || node === state.pinned;
      marks.append(
        svgNode("circle", {
          class: `tree-node${node.is_leaf ? " leaf" : ""}${active ? " active" : ""}`,
          cx: node.x,
          cy: node.y,
          r: 5 / Math.max(0.001, Math.min(state.sx, state.sy)),
        }),
      );
    }
    group.append(links, marks);
    svg.append(group);
    updateTooltip();
  }

  function updateTooltip() {
    const node = state.hovered || state.pinned;
    if (!node || state.dragging) {
      tooltip.hidden = true;
      return;
    }
    const point = position(node);
    tooltip.hidden = false;
    tooltip.textContent = node.is_leaf
      ? node.keyword
      : `${node.keywords.length.toLocaleString()} keywords`;
    tooltip.style.left = `${point.x + 12}px`;
    tooltip.style.top = `${point.y - 12}px`;
  }

  function lineChart(values) {
    const chart = svgNode("svg", { class: "line-chart", viewBox: "0 0 340 180" });
    if (!values.length) {
      const message = svgNode("text", { x: 16, y: 30 });
      message.textContent = "No yearly observations";
      chart.append(message);
      return chart;
    }
    const padding = { left: 44, right: 12, top: 16, bottom: 30 };
    const years = values.map((item) => item.year);
    const counts = values.map((item) => item.papers);
    const minYear = Math.min(...years);
    const maxYear = Math.max(...years);
    const maxCount = Math.max(1, ...counts);
    const x = (year) => padding.left + ((year - minYear) / Math.max(1, maxYear - minYear)) * (340 - padding.left - padding.right);
    const y = (count) => 180 - padding.bottom - (count / maxCount) * (180 - padding.top - padding.bottom);
    chart.append(
      svgNode("path", {
        class: "axis",
        d: `M ${padding.left} ${padding.top} V ${180 - padding.bottom} H ${340 - padding.right}`,
      }),
    );
    const path = values.map((item, index) => `${index ? "L" : "M"} ${x(item.year)} ${y(item.papers)}`).join(" ");
    chart.append(svgNode("path", { class: "series", d: path }));
    for (const item of values) {
      const point = svgNode("circle", {
        class: "series-point",
        cx: x(item.year),
        cy: y(item.papers),
        r: 3.5,
      });
      const title = svgNode("title");
      title.textContent = `${item.year}: ${Number(item.papers).toLocaleString()} papers (${(100 * item.share).toFixed(2)}%)`;
      point.append(title);
      chart.append(point);
    }
    const start = svgNode("text", { class: "axis-label", x: padding.left, y: 172 });
    start.textContent = String(minYear);
    const finish = svgNode("text", { class: "axis-label end", x: 340 - padding.right, y: 172 });
    finish.textContent = String(maxYear);
    const maximum = svgNode("text", { class: "axis-label", x: 4, y: padding.top + 4 });
    maximum.textContent = maxCount.toLocaleString();
    chart.append(start, finish, maximum);
    return chart;
  }

  function show(node) {
    if (!node) return;
    details.replaceChildren();
    const eyebrow = document.createElement("p");
    eyebrow.className = "eyebrow";
    eyebrow.textContent = node.is_leaf ? "Keyword" : "Keyword cluster";
    const heading = document.createElement("h2");
    heading.textContent = node.is_leaf ? node.keyword : `${node.keywords.length} keywords`;
    const chartTitle = document.createElement("h3");
    chartTitle.textContent = "Papers by year";
    const wordTitle = document.createElement("h3");
    wordTitle.textContent = "Included keywords";
    const words = document.createElement("div");
    words.className = "tags";
    for (const keyword of node.keywords) {
      const tag = document.createElement("span");
      tag.textContent = keyword;
      words.append(tag);
    }
    details.append(eyebrow, heading, chartTitle, lineChart(node.yearly || []), wordTitle, words);
  }

  svg.addEventListener("pointerdown", (event) => {
    state.dragging = true;
    state.moved = 0;
    state.pointer = { x: event.clientX, y: event.clientY };
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    if (state.dragging) {
      const dx = event.clientX - state.pointer.x;
      const dy = event.clientY - state.pointer.y;
      state.tx += dx;
      state.ty += dy;
      state.moved += Math.abs(dx) + Math.abs(dy);
      state.pointer = { x: event.clientX, y: event.clientY };
    } else {
      state.hovered = nearest(event.clientX, event.clientY);
      if (state.hovered) show(state.hovered);
    }
    draw();
  });
  svg.addEventListener("pointerup", (event) => {
    if (state.moved < 5) {
      state.pinned = nearest(event.clientX, event.clientY);
      if (state.pinned) show(state.pinned);
    }
    state.dragging = false;
    state.pointer = null;
    if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
    draw();
  });
  svg.addEventListener("pointerleave", () => {
    if (!state.dragging) {
      state.hovered = null;
      if (state.pinned) show(state.pinned);
      draw();
    }
  });
  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = svg.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const wx = (px - state.tx) / state.sx;
    const wy = (py - state.ty) / state.sy;
    const factor = Math.exp(-event.deltaY * 0.0012);
    state.sx = Math.max(0.01, Math.min(1000, state.sx * factor));
    state.sy = Math.max(0.01, Math.min(1000, state.sy * factor));
    state.tx = px - wx * state.sx;
    state.ty = py - wy * state.sy;
    draw();
  }, { passive: false });
  reset.addEventListener("click", fit);
  window.addEventListener("resize", fit);

  fetch("data.json")
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      state.nodes = data.dendrogram.nodes || [];
      state.byId = new Map(state.nodes.map((node) => [node.id, node]));
      summary.textContent = `${data.meta.selected_keywords.toLocaleString()} keywords · ${data.meta.clusters.toLocaleString()} clusters · ${(100 * data.meta.cluster_similarity).toFixed(0)}% similarity cut`;
      reset.disabled = false;
      fit();
    })
    .catch((error) => {
      summary.textContent = `Unable to load dendrogram: ${error.message}`;
    });
})();
