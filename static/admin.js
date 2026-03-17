const PAGE_SIZE = 20;

const state = {
  token: window.localStorage.getItem("gateway_admin_token") || "",
  cardsOffset: 0,
  cardsTotal: 0,
  flashTimer: null,
  pollTimer: null,
};

document.addEventListener("DOMContentLoaded", () => {
  bindTabs();
  bindAuth();
  bindOverviewActions();
  bindMemoryActions();
  bindVectorActions();

  document.getElementById("tokenInput").value = state.token;
  refreshAll();
  state.pollTimer = window.setInterval(pollJobs, 5000);
});

function bindTabs() {
  document.querySelectorAll(".tab-btn").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(button.dataset.target).classList.add("active");
    });
  });
}

function bindAuth() {
  document.getElementById("saveTokenBtn").addEventListener("click", () => {
    state.token = document.getElementById("tokenInput").value.trim();
    window.localStorage.setItem("gateway_admin_token", state.token);
    flash("口令已保存，可以开始操作。", "success");
    refreshAll();
  });

  document.getElementById("clearTokenBtn").addEventListener("click", () => {
    state.token = "";
    document.getElementById("tokenInput").value = "";
    window.localStorage.removeItem("gateway_admin_token");
    flash("口令已清空。", "info");
    refreshAll();
  });
}

function bindOverviewActions() {
  document.getElementById("refreshOverviewBtn").addEventListener("click", loadOverview);
  document.getElementById("refreshJobsBtn").addEventListener("click", pollJobs);
  document.getElementById("reloadProvidersBtn").addEventListener("click", async () => {
    await runAction("模型配置已重新加载。", "/admin/providers/reload", { method: "POST" });
    await loadOverview();
  });
}

function bindMemoryActions() {
  document.getElementById("refreshCardsBtn").addEventListener("click", async () => {
    await loadCardStatus();
    await loadCardList();
  });

  document.getElementById("loadCardListBtn").addEventListener("click", async () => {
    state.cardsOffset = 0;
    await loadCardList();
  });

  document.getElementById("cardsPrevBtn").addEventListener("click", async () => {
    if (state.cardsOffset === 0) {
      flash("已经是第一页。", "info");
      return;
    }
    state.cardsOffset = Math.max(0, state.cardsOffset - PAGE_SIZE);
    await loadCardList();
  });

  document.getElementById("cardsNextBtn").addEventListener("click", async () => {
    if (state.cardsOffset + PAGE_SIZE >= state.cardsTotal) {
      flash("已经是最后一页。", "info");
      return;
    }
    state.cardsOffset += PAGE_SIZE;
    await loadCardList();
  });

  document.getElementById("searchCardsBtn").addEventListener("click", searchCards);

  document.getElementById("generateSingleBtn").addEventListener("click", async () => {
    const date = document.getElementById("generateDateInput").value;
    if (!date) {
      flash("请先选一天。", "error");
      return;
    }
    await runAction(
      "已开始生成这一天的记忆卡。",
      "/admin/cards/generate",
      {
        method: "POST",
        body: {
          date,
          force: document.getElementById("forceGenerateInput").checked,
        },
      },
    );
    await pollJobs();
  });

  document.getElementById("generateRangeBtn").addEventListener("click", async () => {
    const startDate = document.getElementById("rangeStartInput").value;
    const endDate = document.getElementById("rangeEndInput").value;
    if (!startDate || !endDate) {
      flash("请先选开始日期和结束日期。", "error");
      return;
    }
    await runAction(
      "已开始生成这一段时间的记忆卡。",
      "/admin/cards/generate",
      {
        method: "POST",
        body: {
          start_date: startDate,
          end_date: endDate,
          force: document.getElementById("forceGenerateInput").checked,
        },
      },
    );
    await pollJobs();
  });

  document.getElementById("embedCardsBtn").addEventListener("click", async () => {
    await runAction("已开始补做记忆卡向量。", "/admin/cards/embed", { method: "POST" });
    await pollJobs();
  });

  document.getElementById("rebuildCardsBtn").addEventListener("click", async () => {
    const startDate = document.getElementById("rangeStartInput").value;
    const endDate = document.getElementById("rangeEndInput").value;
    const ok = window.confirm("全量重建会重做大量历史总结，可能持续很久。确定继续吗？");
    if (!ok) {
      return;
    }
    await runAction(
      "已开始全量重建记忆卡。",
      "/admin/cards/rebuild_full",
      {
        method: "POST",
        body: {
          start_date: startDate || null,
          end_date: endDate || null,
        },
      },
    );
    await pollJobs();
  });
}

function bindVectorActions() {
  document.getElementById("refreshVectorsBtn").addEventListener("click", loadVectorSection);

  document.getElementById("nightlyVectorBtn").addEventListener("click", async () => {
    await runAction("已开始日常补量。", "/admin/vectors/nightly", { method: "POST" });
    await pollJobs();
  });

  document.getElementById("fullVectorRebuildBtn").addEventListener("click", async () => {
    const ok = window.confirm("全量重建向量会重新处理全部消息。确定继续吗？");
    if (!ok) {
      return;
    }
    await runAction("已开始全量重建向量。", "/admin/vectors/rebuild", { method: "POST" });
    await pollJobs();
  });

  document.getElementById("refreshNotionBtn").addEventListener("click", async () => {
    const data = await runAction("Notion 缓存已刷新。", "/admin/notion/refresh", { method: "POST" });
    renderDiagnosticOutput(data);
  });

  document.getElementById("testNotionBtn").addEventListener("click", async () => {
    const data = await api("/admin/notion/test");
    renderDiagnosticSummary(data);
    renderDiagnosticOutput(data);
    flash("Notion 诊断已完成。", "success");
  });

  document.getElementById("backupBtn").addEventListener("click", async () => {
    const ok = window.confirm("现在要创建一次数据库备份吗？");
    if (!ok) {
      return;
    }
    const data = await runAction("数据库备份已开始。", "/admin/backup", { method: "POST" });
    renderDiagnosticOutput(data);
  });
}

async function refreshAll() {
  await Promise.all([loadOverview(), loadMemorySection(), loadVectorSection(), pollJobs()]);
}

async function loadOverview() {
  try {
    const health = await api("/health", { auth: false });
    document.getElementById("gatewayHealthChip").textContent = `网关状态：${health.status === "ok" ? "正常" : "异常"}`;

    if (!state.token) {
      document.getElementById("overviewCards").innerHTML = [
        makeMetricCard("网关状态", health.status === "ok" ? "正常" : "异常", `时间：${safeText(health.timestamp)}`),
        makeMetricCard("下一步", "先填口令", "填好管理员口令后，才能读取详细后台数据"),
      ].join("");
      document.getElementById("providerSummary").innerHTML = renderNeedToken("填好管理员口令后，这里会显示提供商和模型来源。");
      document.getElementById("jobSummary").innerHTML = renderNeedToken("填好管理员口令后，这里会显示长任务进度。");
      return;
    }

    const [stats, models, cardStatus, vectorStatus] = await Promise.all([
      api("/admin/stats"),
      api("/v1/models"),
      api("/admin/cards/status"),
      api("/admin/vectors/status"),
    ]);

    const metrics = [
      makeMetricCard("网关状态", health.status === "ok" ? "正常" : "异常", `时间：${safeText(health.timestamp)}`),
      makeMetricCard("总消息数", numberText(stats.total_messages), `今日新增：${numberText(stats.today_messages)}`),
      makeMetricCard("总会话数", numberText(stats.total_conversations), "表示一共有多少段聊天"),
      makeMetricCard("记忆卡数量", numberText(cardStatus.total_cards), `卡片向量：${numberText(cardStatus.card_vector_store_size)}`),
      makeMetricCard("消息向量", numberText(vectorStatus.vector_store_size), `待处理分块：${numberText(vectorStatus.pending_chunks)}`),
      makeMetricCard("摘要架构", "已统一", "每日和每周摘要都走新架构"),
      makeMetricCard("可用提供商数", numberText((models.data || []).length), "这里显示接口读到的模型来源"),
      makeMetricCard("今天消息数", numberText(stats.today_messages), "可用来看今天是否正常写入"),
    ];

    document.getElementById("overviewCards").innerHTML = metrics.join("");
    document.getElementById("providerSummary").innerHTML = renderProviderSummary(models.data || []);
  } catch (error) {
    flash(error.message, "error");
  }
}

async function loadMemorySection() {
  await Promise.all([loadCardStatus(), loadCardList()]);
}

async function loadCardStatus() {
  try {
    if (!state.token) {
      document.getElementById("cardStatusSummary").innerHTML = renderNeedToken("先填管理员口令，再读取记忆卡状态。");
      return;
    }
    const data = await api("/admin/cards/status");
    const summary = [
      summaryLine("记忆卡总数", numberText(data.total_cards)),
      summaryLine("已做向量", numberText(data.embedded_cards)),
      summaryLine("未做向量", numberText(data.pending_embedding)),
      summaryLine("摘要架构", "每日总结和每周摘要都来自记忆卡"),
    ];
    document.getElementById("cardStatusSummary").innerHTML = summary.join("");
  } catch (error) {
    flash(error.message, "error");
  }
}

async function loadCardList() {
  try {
    if (!state.token) {
      document.getElementById("cardList").innerHTML = "先填管理员口令，再读取记忆卡列表。";
      document.getElementById("cardsPageText").textContent = "请先填口令";
      return;
    }
    const filterDate = document.getElementById("filterDateInput").value;
    const query = new URLSearchParams({
      full: "1",
      limit: String(PAGE_SIZE),
      offset: String(state.cardsOffset),
    });
    if (filterDate) {
      query.set("date", filterDate);
    }
    const data = await api(`/admin/cards/list?${query.toString()}`);
    state.cardsTotal = data.total || data.count || 0;
    document.getElementById("cardsPageText").textContent = `第 ${Math.floor(state.cardsOffset / PAGE_SIZE) + 1} 页 / 共 ${Math.max(1, Math.ceil(state.cardsTotal / PAGE_SIZE))} 页`;
    document.getElementById("cardList").innerHTML = renderCardList(data.cards || []);
  } catch (error) {
    flash(error.message, "error");
  }
}

async function searchCards() {
  const query = document.getElementById("cardSearchInput").value.trim();
  if (!state.token) {
    flash("请先填管理员口令。", "error");
    return;
  }
  if (!query) {
    flash("请先输入关键词。", "error");
    return;
  }
  try {
    const data = await api("/admin/cards/search", {
      method: "POST",
      body: { query },
    });
    document.getElementById("cardSearchResults").innerHTML = renderSearchResults(data.results || []);
  } catch (error) {
    flash(error.message, "error");
  }
}

async function loadVectorSection() {
  try {
    if (!state.token) {
      document.getElementById("vectorStatusSummary").innerHTML = renderNeedToken("先填管理员口令，再读取向量状态。");
      document.getElementById("diagnosticSummary").innerHTML = renderNeedToken("先填管理员口令，再做工具诊断。");
      return;
    }
    const [vectorStatus, notionStatus] = await Promise.all([
      api("/admin/vectors/status"),
      api("/admin/notion/test"),
    ]);
    const vectorLines = [
      summaryLine("消息向量数量", numberText(vectorStatus.vector_store_size)),
      summaryLine("总分块数", numberText(vectorStatus.total_chunks)),
      summaryLine("待做向量分块", numberText(vectorStatus.pending_chunks)),
      summaryLine("记忆卡向量数量", numberText(vectorStatus.card_vector_store_size)),
    ];
    document.getElementById("vectorStatusSummary").innerHTML = vectorLines.join("");
    renderDiagnosticSummary(notionStatus);
  } catch (error) {
    flash(error.message, "error");
  }
}

async function pollJobs() {
  try {
    if (!state.token) {
      document.getElementById("jobSummary").innerHTML = renderNeedToken("先填管理员口令，再查看任务进度。");
      document.getElementById("jobStatusChip").textContent = "任务状态：请先填口令";
      return;
    }
    const [cardJob, vectorJob] = await Promise.all([
      api("/admin/cards/rebuild_full/status"),
      api("/admin/vectors/rebuild/status"),
    ]);
    const lines = [
      renderJobLine("记忆卡任务", cardJob),
      renderJobLine("向量任务", vectorJob),
    ];
    document.getElementById("jobSummary").innerHTML = lines.join("");
    document.getElementById("jobStatusChip").textContent = `任务状态：${jobChipText(cardJob, vectorJob)}`;
  } catch (error) {
    document.getElementById("jobSummary").innerHTML = `<div class="summary-line">${safeText(error.message)}</div>`;
  }
}

async function runAction(successMessage, path, options) {
  const data = await api(path, options);
  flash(successMessage, "success");
  return data;
}

async function api(path, options = {}) {
  const method = options.method || "GET";
  const auth = options.auth !== false;
  const headers = {};

  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  if (auth && state.token) {
    headers.Authorization = `Bearer ${state.token}`;
  }

  const response = await fetch(path, {
    method,
    headers,
    body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
  });

  let payload = null;
  const text = await response.text();
  payload = text ? safeJsonParse(text) : {};

  if (!response.ok) {
    const message =
      payload?.error ||
      payload?.message ||
      (response.status === 401 ? "口令不对，或者还没有填管理员口令。" : `请求失败：${response.status}`);
    throw new Error(message);
  }
  return payload;
}

function renderProviderSummary(models) {
  if (!models.length) {
    return `<div class="list-item"><h4>还没有模型数据</h4><p>可能是口令没填，或接口暂时无法读取。</p></div>`;
  }
  return models
    .slice(0, 12)
    .map((model) => `
      <div class="list-item">
        <h4>${safeText(model.provider || "未命名提供商")}</h4>
        <p>前缀：${safeText((model.prefixes || []).join(", ") || "无")}</p>
      </div>
    `)
    .join("");
}

function renderCardList(cards) {
  if (!cards.length) {
    return "没有找到记忆卡。";
  }
  return cards
    .map((card) => `
      <article class="card-item">
        <h4>${safeText(card.date || "未知日期")}</h4>
        <p>标签：${safeText(card.tags || "无")}</p>
        <p>${safeText(shortText(card.summary || "", 420))}</p>
      </article>
    `)
    .join("");
}

function renderSearchResults(results) {
  if (!results.length) {
    return "没有搜到相关记忆卡。";
  }
  return results
    .map((item) => `
      <article class="card-item">
        <h4>${safeText(item.date || "未知日期")} · 分数 ${safeText(String(item.score ?? ""))}</h4>
        <p>标签：${safeText(item.tags || "无")}</p>
        <p>${safeText(item.summary || "")}</p>
      </article>
    `)
    .join("");
}

function renderDiagnosticSummary(data) {
  const lines = [
    summaryLine("Notion 口令", data.notion_token_set ? "已设置" : "未设置"),
    summaryLine("工具开关", data.tools_enabled ? "已开启" : "未开启"),
    summaryLine("可用工具数", numberText(data.tools_count)),
    summaryLine("API 状态", safeText(String(data.api_status || data.api_error || "未检查"))),
  ];
  document.getElementById("diagnosticSummary").innerHTML = lines.join("");
}

function renderDiagnosticOutput(data) {
  document.getElementById("diagnosticOutput").textContent = JSON.stringify(data, null, 2);
}

function renderJobLine(title, job) {
  const phase = safeText(job.phase || "idle");
  const message = safeText(job.message || "暂无信息");
  const finished = job.completed_at ? `，完成时间：${safeText(job.completed_at)}` : "";
  return summaryLine(title, `${phase}，${message}${finished}`);
}

function jobChipText(cardJob, vectorJob) {
  if (cardJob.running) {
    return `记忆卡处理中：${cardJob.phase || "进行中"}`;
  }
  if (vectorJob.running) {
    return `向量处理中：${vectorJob.phase || "进行中"}`;
  }
  return "当前没有长任务在跑";
}

function makeMetricCard(title, value, help) {
  return `
    <article class="metric-card">
      <span>${safeText(title)}</span>
      <strong>${safeText(value)}</strong>
      <p>${safeText(help)}</p>
    </article>
  `;
}

function renderNeedToken(message) {
  return `<div class="summary-line">${safeText(message)}</div>`;
}

function summaryLine(label, value) {
  return `<div class="summary-line"><strong>${safeText(label)}：</strong>${safeText(value)}</div>`;
}

function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch (_error) {
    return { message: text };
  }
}

function numberText(value) {
  return typeof value === "number" ? String(value) : safeText(String(value ?? 0));
}

function shortText(text, maxLength) {
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength)}...`;
}

function safeText(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function flash(message, type = "info") {
  const node = document.getElementById("flashMessage");
  node.textContent = message;
  node.className = `flash ${type}`;
  if (state.flashTimer) {
    window.clearTimeout(state.flashTimer);
  }
  state.flashTimer = window.setTimeout(() => {
    node.className = "flash hidden";
  }, 2800);
}
