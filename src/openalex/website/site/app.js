(() => {
  "use strict";

  const list = document.querySelector("#keywords");
  const summary = document.querySelector("#summary");

  fetch("data.json")
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      const total = Number(data.meta.processed_papers || 0);
      summary.textContent = `${data.top_keywords.length.toLocaleString()} keywords · ${total.toLocaleString()} processed papers`;
      const maximum = Math.max(1, ...data.top_keywords.map((item) => item.papers));
      for (const item of data.top_keywords) {
        const row = document.createElement("li");
        row.className = "keyword-row";
        const header = document.createElement("div");
        header.className = "keyword-header";
        const name = document.createElement("strong");
        name.textContent = item.keyword;
        const value = document.createElement("span");
        value.textContent = `${Number(item.papers).toLocaleString()} papers · ${(100 * item.share).toFixed(2)}%`;
        header.append(name, value);
        const track = document.createElement("div");
        track.className = "bar-track";
        const bar = document.createElement("div");
        bar.className = "bar";
        bar.style.width = `${(100 * item.papers) / maximum}%`;
        track.append(bar);
        row.append(header, track);
        list.append(row);
      }
    })
    .catch((error) => {
      summary.textContent = `Unable to load keywords: ${error.message}`;
    });
})();
