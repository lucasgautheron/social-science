(() => {
  "use strict";

  const canvas = document.querySelector("#graph");
  const panel = canvas.parentElement;
  const context = canvas.getContext("2d");
  const details = document.querySelector("#details");
  const tooltip = document.querySelector("#tooltip");
  const summary = document.querySelector("#summary");
  const reset = document.querySelector("#reset");
  const keywordMode = document.querySelector("#keyword-mode");
  const clusterMode = document.querySelector("#cluster-mode");
  const NS = "http://www.w3.org/2000/svg";
  const state = {
    graph: null,
    mode: "keyword",
    nodes: [],
    edges: [],
    maxPapers: 1,
    hovered: null,
    pinned: null,
    scale: 1,
    tx: 0,
    ty: 0,
    dragging: false,
    moved: 0,
    pointer: null,
    hitGrid: new Map(),
  };

  function svgNode(name, attrs = {}) {
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    return node;
  }

  function bounds() {
    const rect = canvas.getBoundingClientRect();
    return { width: Math.max(1, rect.width), height: Math.max(1, rect.height) };
  }

  function resizeCanvas() {
    const { width, height } = bounds();
    const ratio = window.devicePixelRatio || 1;
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { width, height };
  }

  function nodeRadius(node) {
    return Math.max(2.5, 19 * Math.sqrt(Number(node.papers || 0) / state.maxPapers));
  }

  function position(node) {
    return {
      x: node.x * state.scale + state.tx,
      y: node.y * state.scale + state.ty,
    };
  }

  function fit() {
    if (!state.nodes.length) return;
    const { width, height } = bounds();
    const xs = state.nodes.map((node) => node.x);
    const ys = state.nodes.map((node) => node.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const availableWidth = Math.max(1, width - 96);
    const availableHeight = Math.max(1, height - 96);
    state.scale = Math.min(
      availableWidth / Math.max(maxX - minX, 0.1),
      availableHeight / Math.max(maxY - minY, 0.1),
    );
    state.tx = width / 2 - ((minX + maxX) / 2) * state.scale;
    state.ty = height / 2 - ((minY + maxY) / 2) * state.scale;
    draw();
  }

  function gridKey(column, row) {
    return `${column}:${row}`;
  }

  function indexNode(node, x, y, radius) {
    const cell = 32;
    const minColumn = Math.floor((x - radius - 5) / cell);
    const maxColumn = Math.floor((x + radius + 5) / cell);
    const minRow = Math.floor((y - radius - 5) / cell);
    const maxRow = Math.floor((y + radius + 5) / cell);
    for (let column = minColumn; column <= maxColumn; column += 1) {
      for (let row = minRow; row <= maxRow; row += 1) {
        const key = gridKey(column, row);
        if (!state.hitGrid.has(key)) state.hitGrid.set(key, []);
        state.hitGrid.get(key).push({ node, x, y, radius });
      }
    }
  }

  function nearest(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const cell = 32;
    const candidates = state.hitGrid.get(
      gridKey(Math.floor(x / cell), Math.floor(y / cell)),
    ) || [];
    let result = null;
    let best = Infinity;
    for (const candidate of candidates) {
      const distance = (candidate.x - x) ** 2 + (candidate.y - y) ** 2;
      if (distance <= (candidate.radius + 5) ** 2 && distance < best) {
        best = distance;
        result = candidate.node;
      }
    }
    return result;
  }

  function draw() {
    const { width, height } = resizeCanvas();
    context.clearRect(0, 0, width, height);
    state.hitGrid.clear();
    if (!state.nodes.length) return;
    const maxWeight = Math.max(1, ...state.edges.map((edge) => Number(edge.weight)));
    context.lineCap = "round";
    for (const edge of state.edges) {
      const source = state.nodes[edge.source];
      const target = state.nodes[edge.target];
      if (!source || !target) continue;
      const start = position(source);
      const finish = position(target);
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(finish.x, finish.y);
      context.strokeStyle = "rgb(84 105 101 / 24%)";
      context.lineWidth = 0.5 + 2.5 * Math.sqrt(Number(edge.weight) / maxWeight);
      context.stroke();
    }
    for (const node of state.nodes) {
      const point = position(node);
      const radius = nodeRadius(node);
      context.beginPath();
      context.arc(point.x, point.y, radius, 0, Math.PI * 2);
      context.fillStyle = node.color;
      context.fill();
      context.strokeStyle = node === state.pinned || node === state.hovered
        ? "#102f2b"
        : "rgb(255 255 255 / 90%)";
      context.lineWidth = node === state.pinned || node === state.hovered ? 3 : 1.5;
      context.stroke();
      indexNode(node, point.x, point.y, radius);
    }
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
    tooltip.textContent = node.kind === "keyword"
      ? node.keyword
      : `Cluster ${node.group} · ${node.keyword_count.toLocaleString()} keywords`;
    tooltip.style.left = `${point.x + 12}px`;
    tooltip.style.top = `${point.y - 12}px`;
  }

  function formatPercent(share) {
    const percent = 100 * Number(share || 0);
    const digits = percent >= 10 ? 1 : percent >= 1 ? 2 : 3;
    return `${percent.toFixed(digits)}%`;
  }

  function lineChart(values) {
    const chart = svgNode("svg", { class: "line-chart", viewBox: "0 0 340 180" });
    if (!values.length) {
      const message = svgNode("text", { x: 16, y: 30 });
      message.textContent = "No yearly observations";
      chart.append(message);
      return chart;
    }
    const padding = { left: 52, right: 12, top: 16, bottom: 30 };
    const years = values.map((item) => item.year);
    const shares = values.map((item) => Number(item.share) || 0);
    const minYear = Math.min(...years);
    const maxYear = Math.max(...years);
    const maxShare = Math.max(0, ...shares);
    const scale = maxShare > 0 ? maxShare : 1;
    const x = (year) => padding.left + ((year - minYear) / Math.max(1, maxYear - minYear)) * (340 - padding.left - padding.right);
    const y = (share) => 180 - padding.bottom - (share / scale) * (180 - padding.top - padding.bottom);
    chart.append(
      svgNode("path", {
        class: "axis",
        d: `M ${padding.left} ${padding.top} V ${180 - padding.bottom} H ${340 - padding.right}`,
      }),
    );
    const path = values.map((item, index) => `${index ? "L" : "M"} ${x(item.year)} ${y(Number(item.share) || 0)}`).join(" ");
    chart.append(svgNode("path", { class: "series", d: path }));
    for (const item of values) {
      const point = svgNode("circle", {
        class: "series-point",
        cx: x(item.year),
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
    return chart;
  }

  function show(node) {
    if (!node) return;
    details.replaceChildren();
    const eyebrow = document.createElement("p");
    eyebrow.className = "eyebrow";
    eyebrow.textContent = node.kind === "keyword" ? "Keyword" : "Keyword cluster";
    const heading = document.createElement("h2");
    heading.textContent = node.kind === "keyword" ? node.keyword : `Cluster ${node.group}`;
    const frequency = document.createElement("p");
    frequency.className = "detail-stat";
    frequency.textContent = node.kind === "keyword"
      ? `${Number(node.papers).toLocaleString()} papers`
      : `${Number(node.papers).toLocaleString()} summed keyword frequency · ${Number(node.document_frequency).toLocaleString()} distinct papers`;
    const chartTitle = document.createElement("h3");
    chartTitle.textContent = "Share of papers by year";
    const wordTitle = document.createElement("h3");
    wordTitle.textContent = node.kind === "keyword" ? "Cluster" : "Included keywords";
    const words = document.createElement("div");
    words.className = "tags";
    if (node.kind === "keyword") {
      const tag = document.createElement("span");
      tag.textContent = `Cluster ${node.group}`;
      tag.style.borderColor = node.color;
      words.append(tag);
    } else {
      for (const keyword of node.keywords) {
        const tag = document.createElement("span");
        tag.textContent = keyword;
        words.append(tag);
      }
    }
    details.append(
      eyebrow,
      heading,
      frequency,
      chartTitle,
      lineChart(node.yearly || []),
      wordTitle,
      words,
    );
  }

  function updateSummary() {
    const counts = state.graph.counts;
    if (state.mode === "keyword") {
      summary.textContent = `${counts.displayed_keywords.toLocaleString()} of ${counts.original_keywords.toLocaleString()} keywords · ${counts.displayed_edges.toLocaleString()} of ${counts.positive_edges.toLocaleString()} positive-NPMI edges`;
    } else {
      summary.textContent = `${counts.displayed_clusters.toLocaleString()} clusters · ${counts.displayed_cluster_edges.toLocaleString()} of ${counts.positive_cluster_edges.toLocaleString()} positive-NPMI edges · hierarchy level ${state.graph.level}`;
    }
  }

  function setMode(mode) {
    state.mode = mode;
    state.nodes = state.graph[mode].nodes || [];
    state.edges = state.graph[mode].edges || [];
    state.maxPapers = Math.max(1, ...state.nodes.map((item) => Number(item.papers || 0)));
    state.hovered = null;
    state.pinned = null;
    keywordMode.setAttribute("aria-pressed", String(mode === "keyword"));
    clusterMode.setAttribute("aria-pressed", String(mode === "cluster"));
    updateSummary();
    fit();
  }

  canvas.addEventListener("pointerdown", (event) => {
    state.dragging = true;
    state.moved = 0;
    state.pointer = { x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (state.dragging) {
      const dx = event.clientX - state.pointer.x;
      const dy = event.clientY - state.pointer.y;
      state.tx += dx;
      state.ty += dy;
      state.moved += Math.abs(dx) + Math.abs(dy);
      state.pointer = { x: event.clientX, y: event.clientY };
    } else {
      state.hovered = nearest(event.clientX, event.clientY);
    }
    draw();
  });
  canvas.addEventListener("pointerup", (event) => {
    if (state.moved < 5) {
      state.pinned = nearest(event.clientX, event.clientY);
      if (state.pinned) show(state.pinned);
    }
    state.dragging = false;
    state.pointer = null;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    draw();
  });
  canvas.addEventListener("pointerleave", () => {
    if (!state.dragging) {
      state.hovered = null;
      draw();
    }
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const wx = (px - state.tx) / state.scale;
    const wy = (py - state.ty) / state.scale;
    const factor = Math.exp(-event.deltaY * 0.0012);
    state.scale = Math.max(10, Math.min(5000, state.scale * factor));
    state.tx = px - wx * state.scale;
    state.ty = py - wy * state.scale;
    draw();
  }, { passive: false });
  keywordMode.addEventListener("click", () => setMode("keyword"));
  clusterMode.addEventListener("click", () => setMode("cluster"));
  reset.addEventListener("click", fit);
  window.addEventListener("resize", fit);

  fetch("data.json")
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      if (!data.graph) {
        summary.textContent = "No blockmodel graph was supplied when this site was built.";
        panel.classList.add("graph-unavailable");
        keywordMode.disabled = true;
        clusterMode.disabled = true;
        details.innerHTML = '<div class="empty"><p class="eyebrow">Graph unavailable</p><h2>Build with cluster artifacts</h2><p>Pass <code>--clusters-dir output/event_clusters</code> to <code>openalex build-website</code>.</p></div>';
        return;
      }
      state.graph = data.graph;
      reset.disabled = false;
      setMode("keyword");
    })
    .catch((error) => {
      summary.textContent = `Unable to load graph: ${error.message}`;
    });
})();
