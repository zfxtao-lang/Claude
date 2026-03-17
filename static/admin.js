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
  document.getElementById("workerDateInput").value = new Date().toISOString().slice(0, 10);
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
  document.getElementById("refreshSystemStatusBtn").addEventListener("click", loadSystemStatus);
  document.getElementById("refreshWorkerRunsBtn").addEventListener("click", loadWorkerRuns);
  document.getElementById("runMemoryWorkerBtn").addEventListener("click", runMemoryWorker);
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
  document.getElementById("debugSearchBtn").addEventListener("click", debugSearch);
  document.getElementById("refreshReviewsBtn").addEventListener("click", loadReviews);
  document.getElementById("refreshDiariesBtn").addEventListener("click", loadDiaries);
  document.getElementById("refreshMemoryDataBtn").addEventListener("click", async () => {
    await Promise.all([loadSlices(), loadLongMemories()]);
  });

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
  await Promise.all([loadOverview(), loadMemorySection(), loadVectorSection(), pollJobs(), loadSystemStatus(), loadWorkerRuns()]);
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
      document.getElementById("systemStatusSummary").innerHTML = renderNeedToken("填好管理员口令后，这里会显示 worker 和记忆系统状态。");
      document.getElementById("workerRunsList").innerHTML = renderNeedToken("填好管理员口令后，这里会显示最近 worker 运行记录。");
      return;
    }

    const [stats, models, cardStatus, vectorStatus, systemStatus] = await Promise.all([
      api("/admin/stats"),
      api("/v1/models"),
      api("/admin/cards/status"),
      api("/admin/vectors/status"),
      api("/admin/system/status"),
    ]);

    const metrics = [
      makeMetricCard("网关状态", health.status === "ok" ? "正常" : "异常", `时间：${safeText(health.timestamp)}`),
      makeMetricCard("总消息数", numberText(stats.total_messages), `今日新增：${numberText(stats.today_messages)}`),
      makeMetricCard("总会话数", numberText(stats.total_conversations), "表示一共有多少段聊天"),
      makeMetricCard("记忆卡数量", numberText(cardStatus.total_cards), `卡片向量：${numberText(cardStatus.card_vector_store_size)}`),
      makeMetricCard("消息向量", numberText(vectorStatus.vector_store_size), `待处理分块：${numberText(vectorStatus.pending_chunks)}`),
      makeMetricCard("待审核画像", numberText(stats.reviews_pending), "需要你人工确认后才会生效"),
      makeMetricCard("长期记忆", numberText(stats.long_term_total), `已做向量：${numberText(systemStatus.vectors.long_term_vectors)}`),
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
  await Promise.all([loadCardStatus(), loadCardList(), loadReviews(), loadDiaries(), loadSlices(), loadLongMemories()]);
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

async function loadSystemStatus() {
  try {
    if (!state.token) {
      document.getElementById("systemStatusSummary").innerHTML = renderNeedToken("先填管理员口令，再读取系统状态。");
      return;
    }
    const data = await api("/admin/system/status");
    const lines = [
      summaryLine("worker 开关", data.worker_config?.enabled ? "已开启" : "未开启"),
      summaryLine("worker 模型", `${safeText(data.worker_config?.provider || "")} / ${safeText(data.worker_config?.model || "")}`),
      summaryLine("切片数量", numberText(data.memory?.slice_total)),
      summaryLine("活跃切片", numberText(data.memory?.slice_active)),
      summaryLine("长期记忆", numberText(data.memory?.long_term_total)),
      summaryLine("待审核条目", numberText(data.memory?.reviews_pending)),
      summaryLine("当前画像更新时间", safeText(data.active_profile?.updated_at || "还没有审核通过的画像")),
    ];
    document.getElementById("systemStatusSummary").innerHTML = lines.join("");
  } catch (error) {
    flash(error.message, "error");
  }
}

async function loadWorkerRuns() {
  try {
    if (!state.token) {
      document.getElementById("workerRunsList").innerHTML = renderNeedToken("先填管理员口令，再读取 worker 运行记录。");
      return;
    }
    const data = await api("/admin/system/worker_runs?limit=10");
    const items = data.worker_runs || [];
    const rows = data.items || items;
    if (!rows.length) {
      document.getElementById("workerRunsList").innerHTML = "还没有 worker 运行记录。";
      return;
    }
    document.getElementById("workerRunsList").innerHTML = rows.map((item) => `
      <article class="list-item">
        <h4>${safeText(item.worker_name || "memory_worker")} · ${safeText(item.status || "unknown")}</h4>
        <p>模式：${safeText(item.run_mode || "manual")} · 阶段：${safeText(item.phase || "queued")}</p>
        <p>输入 tokens：${numberText(item.token_input)} · 输出 tokens：${numberText(item.token_output)} · 总计：${numberText(item.token_total)}</p>
        <p>${safeText(item.message || item.error || "暂无备注")}</p>
      </article>
    `).join("");
  } catch (error) {
    flash(error.message, "error");
  }
}

async function runMemoryWorker() {
  if (!state.token) {
    flash("请先填管理员口令。", "error");
    return;
  }
  const date = document.getElementById("workerDateInput").value || new Date().toISOString().slice(0, 10);
  try {
    await api("/admin/memory/worker/run", {
      method: "POST",
      body: { date, mode: "manual" },
    });
    flash("memory worker 已开始执行。", "success");
    await Promise.all([loadSystemStatus(), loadWorkerRuns()]);
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
    const [cardJob, vectorJob, workerRuns] = await Promise.all([
      api("/admin/cards/rebuild_full/status"),
      api("/admin/vectors/rebuild/status"),
      api("/admin/system/worker_runs?limit=1"),
    ]);
    const latestWorker = (workerRuns.items || [])[0] || {};
    const lines = [
      renderJobLine("记忆卡任务", cardJob),
      renderJobLine("向量任务", vectorJob),
      renderJobLine("总结 worker", latestWorker),
    ];
    document.getElementById("jobSummary").innerHTML = lines.join("");
    document.getElementById("jobStatusChip").textContent = `任务状态：${jobChipText(cardJob, vectorJob, latestWorker)}`;
  } catch (error) {
    document.getElementById("jobSummary").innerHTML = `<div class="summary-line">${safeText(error.message)}</div>`;
  }
}

async function debugSearch() {
  const query = document.getElementById("debugSearchInput").value.trim();
  if (!state.token) {
    flash("请先填管理员口令。", "error");
    return;
  }
  if (!query) {
    flash("请先输入调试问题。", "error");
    return;
  }
  try {
    const data = await api("/admin/memory/debug_search", {
      method: "POST",
      body: { query },
    });
    const rows = data.long_term_results || [];
    const longTermHtml = rows.length ? rows.map((item) => {
      const score = item.score_breakdown || {};
      return `
        <article class="card-item">
          <h4>${safeText(item.title || "长期记忆")} · 最终分 ${safeText(String(score.Score_final ?? item.score ?? ""))}</h4>
          <p>${safeText(shortText(item.content || "", 260))}</p>
          <div class="mini-summary">
            ${summaryLine("S_vec", score.S_vec ?? "")}
            ${summaryLine("S_key", score.S_key ?? "")}
            ${summaryLine("W_fact", score.W_fact ?? "")}
            ${summaryLine("distance_days", score.distance_days ?? "")}
            ${summaryLine("half_life_days", score.half_life_days ?? "")}
            ${summaryLine("D_time", score.D_time ?? "")}
            ${summaryLine("hits", score.hits ?? "")}
            ${summaryLine("B_hits", score.B_hits ?? "")}
            ${summaryLine("Score_final", score.Score_final ?? "")}
          </div>
        </article>
      `;
    }).join("") : `<article class="card-item"><h4>长期记忆</h4><p>没有查到长期记忆命中。</p></article>`;
    const legacyHtml = (data.legacy_card_results || []).length
      ? (data.legacy_card_results || []).map((item) => `
        <article class="card-item">
          <h4>旧记忆卡 · ${safeText(item.date || "未知日期")} · 分数 ${safeText(String(item.score ?? ""))}</h4>
          <p>${safeText(shortText(item.summary || "", 220))}</p>
        </article>
      `).join("")
      : `<article class="card-item"><h4>旧记忆卡</h4><p>没有兼容层命中。</p></article>`;
    const keywordHtml = (data.keyword_results || []).length
      ? (data.keyword_results || []).map((item) => `
        <article class="card-item">
          <h4>关键词回退 · ${safeText(item.created_at || "").slice(0, 10)}</h4>
          <p>${safeText(shortText(item.content || "", 220))}</p>
        </article>
      `).join("")
      : `<article class="card-item"><h4>关键词回退</h4><p>没有关键词回退命中。</p></article>`;
    document.getElementById("debugSearchResults").innerHTML = `${longTermHtml}${legacyHtml}${keywordHtml}`;
  } catch (error) {
    flash(error.message, "error");
  }
}

async function loadReviews() {
  try {
    if (!state.token) {
      document.getElementById("reviewList").innerHTML = renderNeedToken("先填管理员口令，再读取待审核画像。");
      return;
    }
    const data = await api("/admin/reviews?status=pending&limit=20");
    const items = data.items || [];
    if (!items.length) {
      document.getElementById("reviewList").innerHTML = "当前没有待审核画像。";
      return;
    }
    document.getElementById("reviewList").innerHTML = items.map((item) => `
      <article class="card-item">
        <h4>${safeText(item.review_type || "persona")} · #${safeText(item.id)}</h4>
        <p>${safeText(item.diff_summary || "暂无变化摘要")}</p>
        ${renderReviewDiff(item)}
        <div class="action-row">
          <button class="primary-btn" data-review-action="approve" data-review-id="${safeText(item.id)}">同意生效</button>
          <button class="ghost-btn" data-review-action="edit" data-review-id="${safeText(item.id)}">修改后通过</button>
          <button class="danger-btn" data-review-action="reject" data-review-id="${safeText(item.id)}">拒绝</button>
        </div>
      </article>
    `).join("");
    document.querySelectorAll("[data-review-action]").forEach((button) => {
      button.addEventListener("click", () => submitReviewAction(button.dataset.reviewId, button.dataset.reviewAction));
    });
  } catch (error) {
    flash(error.message, "error");
  }
}

async function submitReviewAction(reviewId, action) {
  try {
    const note = window.prompt("可选备注：", "") || "";
    const options = {
      method: "POST",
      body: { note },
    };
    if (action === "edit") {
      const detail = await api(`/admin/reviews/${reviewId}`);
      const currentPayload = detail.edited_payload_json || detail.proposed_payload_json || {};
      const editedText = window.prompt(
        "请直接修改 JSON 后提交：",
        JSON.stringify(currentPayload, null, 2),
      );
      if (!editedText) {
        return;
      }
      try {
        options.body.payload = JSON.parse(editedText);
      } catch (_error) {
        throw new Error("你输入的内容不是有效的 JSON，请检查括号和引号。");
      }
    }
    await api(`/admin/reviews/${reviewId}/${action}`, options);
    flash("审核操作已提交。", "success");
    await Promise.all([loadReviews(), loadSystemStatus(), loadWorkerRuns()]);
  } catch (error) {
    flash(error.message, "error");
  }
}

async function loadDiaries() {
  try {
    if (!state.token) {
      document.getElementById("diaryList").innerHTML = renderNeedToken("先填管理员口令，再读取日记。");
      document.getElementById("moodTrend").innerHTML = renderNeedToken("先填管理员口令，再显示情绪趋势。");
      return;
    }
    const data = await api("/admin/memory/diaries?limit=20");
    const items = data.items || [];
    document.getElementById("moodTrend").innerHTML = renderMoodTrend(items);
    document.getElementById("diaryList").innerHTML = items.length ? items.map((item) => `
      <article class="card-item">
        <h4>${safeText(item.entry_date || "未知日期")} · 心情 ${numberText(item.mood_score)}/10 ${safeText(item.mood_label || "")}</h4>
        <p>${safeText(shortText(item.content || "", 240))}</p>
      </article>
    `).join("") : "还没有日记。";
  } catch (error) {
    flash(error.message, "error");
  }
}

async function loadSlices() {
  try {
    if (!state.token) {
      document.getElementById("sliceList").innerHTML = renderNeedToken("先填管理员口令，再读取记忆切片。");
      return;
    }
    const data = await api("/admin/memory/slices?limit=12");
    const items = data.items || [];
    document.getElementById("sliceList").innerHTML = items.length ? items.map((item) => `
      <article class="card-item">
        <h4>切片 #${safeText(item.id)} · ${safeText(item.status || "active")}</h4>
        <p>消息范围：${safeText(item.msg_id_start)} - ${safeText(item.msg_id_end)} · 共 ${safeText(item.message_count)} 条</p>
        <p>${safeText(shortText(item.summary || "", 180))}</p>
      </article>
    `).join("") : "还没有记忆切片。";
  } catch (error) {
    flash(error.message, "error");
  }
}

async function loadLongMemories() {
  try {
    if (!state.token) {
      document.getElementById("longMemoryList").innerHTML = renderNeedToken("先填管理员口令，再读取长期记忆。");
      return;
    }
    const data = await api("/admin/memory/long_term?limit=12");
    const items = data.items || [];
    document.getElementById("longMemoryList").innerHTML = items.length ? items.map((item) => `
      <article class="card-item">
        <h4>${safeText(item.title || "长期记忆")} · ${safeText(item.memory_type || "summary")}</h4>
        <p>权重 ${safeText(item.fact_weight)} · 半衰期 ${safeText(item.half_life_days)} 天 · 命中 ${safeText(item.hits)}</p>
        <p>${safeText(shortText(item.content || "", 180))}</p>
      </article>
    `).join("") : "还没有长期记忆。";
  } catch (error) {
    flash(error.message, "error");
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

function renderReviewDiff(item) {
  const payload = item.proposed_payload_json || {};
  const persona = payload.persona || {};
  const relationship = payload.relationship || {};
  return `
    <div class="mini-summary">
      ${summaryLine("稳定特征", (persona.stable_traits || []).join("、") || "无")}
      ${summaryLine("当前需要", (persona.current_needs || []).join("、") || "无")}
      ${summaryLine("偏好", (persona.preferences || []).join("、") || "无")}
      ${summaryLine("关系温度", relationship.temperature || "未给出")}
      ${summaryLine("变化", (relationship.changes || []).join("、") || "无")}
    </div>
  `;
}

function renderMoodTrend(items) {
  if (!items.length) {
    return "还没有足够的日记数据。";
  }
  return items
    .slice(0, 7)
    .reverse()
    .map((item) => summaryLine(item.entry_date || "未知日期", `${numberText(item.mood_score)}/10 ${safeText(item.mood_label || "")}`))
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

function jobChipText(cardJob, vectorJob, workerJob) {
  if (cardJob.running) {
    return `记忆卡处理中：${cardJob.phase || "进行中"}`;
  }
  if (vectorJob.running) {
    return `向量处理中：${vectorJob.phase || "进行中"}`;
  }
  if (workerJob.status === "running") {
    return `总结处理中：${workerJob.phase || "进行中"}`;
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
