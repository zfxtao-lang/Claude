const PAGE_SIZE = 20;

const state = {
  token: window.localStorage.getItem("gateway_admin_token") || "",
  cardsOffset: 0,
  cardsTotal: 0,
  slicesOffset: 0,
  slicesTotal: 0,
  slicesLimit: 100,
  slicesStatus: "",
  flashTimer: null,
  pollTimer: null,
  sliceItems: [],
  longMemItems: [],
  coreFactItems: [],
};

document.addEventListener("DOMContentLoaded", () => {
  modal.init();
  bindTabs();
  bindDrawer();
  bindSubTabs();
  bindAuth();
  bindOverviewActions();
  bindMemoryActions();
  bindVectorActions();
  bindExpandToggles();

  document.getElementById("tokenInput").value = state.token;
  _syncTokenUI();
  document.getElementById("workerDateInput").value = new Date().toISOString().slice(0, 10);
  refreshAll();
  state.pollTimer = window.setInterval(pollJobs, 5000);
});

function bindTabs() {
  // Handles ALL .tab-btn elements, including the ones inside the fox drawer.
  document.querySelectorAll(".tab-btn").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((item) => item.classList.remove("tab-active"));
      document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("panel-active"));
      button.classList.add("tab-active");
      // Sync active state to sibling drawer buttons with same data-target
      const target = button.dataset.target;
      document.querySelectorAll(`.tab-btn[data-target="${target}"]`).forEach((b) =>
        b.classList.add("tab-active")
      );
      document.getElementById(target).classList.add("panel-active");
    });
  });
}

function bindDrawer() {
  const foxBtn = document.getElementById("foxBtn");
  const drawer = document.getElementById("foxDrawer");

  foxBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    drawer.classList.toggle("hidden");
  });

  // Close when clicking anywhere outside the drawer or fox button
  document.addEventListener("click", (e) => {
    if (!drawer.classList.contains("hidden") &&
        !drawer.contains(e.target) &&
        e.target !== foxBtn) {
      drawer.classList.add("hidden");
    }
  });

  // Close drawer after a nav tab is selected
  drawer.querySelectorAll(".drawer-nav-btn").forEach((navBtn) => {
    navBtn.addEventListener("click", () => drawer.classList.add("hidden"));
  });
}

function bindSubTabs() {
  document.querySelectorAll(".sub-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".sub-tab-btn").forEach((b) => b.classList.remove("sub-tab-active"));
      document.querySelectorAll(".sub-panel").forEach((p) => p.classList.remove("sub-panel-active"));
      btn.classList.add("sub-tab-active");
      document.getElementById(btn.dataset.subtarget).classList.add("sub-panel-active");
    });
  });
}

function _syncTokenUI() {
  const savedEl = document.getElementById("tokenSaved");
  const formEl = document.getElementById("tokenForm");
  if (state.token) {
    savedEl.classList.remove("hidden");
    formEl.classList.add("hidden");
  } else {
    savedEl.classList.add("hidden");
    formEl.classList.remove("hidden");
  }
}

function bindAuth() {
  document.getElementById("saveTokenBtn").addEventListener("click", () => {
    state.token = document.getElementById("tokenInput").value.trim();
    window.localStorage.setItem("gateway_admin_token", state.token);
    flash("口令已保存。", "success");
    _syncTokenUI();
    refreshAll();
  });

  document.getElementById("clearTokenBtn").addEventListener("click", () => {
    state.token = "";
    document.getElementById("tokenInput").value = "";
    window.localStorage.removeItem("gateway_admin_token");
    flash("口令已清空。", "info");
    _syncTokenUI();
    refreshAll();
  });

  document.getElementById("editTokenBtn").addEventListener("click", () => {
    document.getElementById("tokenSaved").classList.add("hidden");
    document.getElementById("tokenForm").classList.remove("hidden");
    document.getElementById("tokenInput").value = state.token;
    document.getElementById("tokenInput").focus();
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
  document.getElementById("refreshSliceBtn").addEventListener("click", () => { state.slicesOffset = 0; loadSlices(); });
  const slicesStatusEl = document.getElementById("slicesStatusFilter");
  if (slicesStatusEl) {
    slicesStatusEl.addEventListener("change", () => {
      state.slicesStatus = slicesStatusEl.value || "";
      state.slicesOffset = 0;
      loadSlices();
    });
  }
  document.getElementById("refreshLongMemBtn").addEventListener("click", loadLongMemories);
  document.getElementById("refreshCoreFactsBtn").addEventListener("click", loadCoreFacts);
  document.getElementById("addCoreFactBtn").addEventListener("click", openAddCoreFactModal);
  document.getElementById("slicesPrevBtn").addEventListener("click", async () => {
    if (state.slicesOffset <= 0) { flash("已经是第一页。", "info"); return; }
    state.slicesOffset = Math.max(0, state.slicesOffset - (state.slicesLimit || 100));
    await loadSlices();
  });
  document.getElementById("slicesNextBtn").addEventListener("click", async () => {
    if (state.slicesOffset + (state.slicesLimit || 100) >= state.slicesTotal) { flash("已经是最后一页。", "info"); return; }
    state.slicesOffset += state.slicesLimit || 100;
    await loadSlices();
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

  // 记忆切片 事件委托
  document.getElementById("sliceList").addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-action]");
    if (!btn) return;
    const id = parseInt(btn.dataset.id, 10);
    const item = state.sliceItems.find((s) => s.id === id) || {};

    if (btn.dataset.action === "edit-slice") {
      const bodyEl = await modal.open(
        `编辑切片 #${id}`,
        `<div class="mf">
           <label class="ml">内容</label>
           <textarea id="mSliceSummary" class="mta" rows="5"></textarea>
         </div>
         <div class="mf">
           <label class="ml">标签（逗号分隔）</label>
           <input id="mSliceTags" class="mi" type="text">
         </div>`,
        (el) => {
          const val = item.content !== null && item.content !== undefined && String(item.content).trim().length > 0
            ? item.content
            : (item.summary || "");
          el.querySelector("#mSliceSummary").value = val;
          el.querySelector("#mSliceTags").value = item.tags || "";
        },
      );
      if (!bodyEl) return;
      try {
        await api(`/admin/memory/slices/${id}`, {
          method: "PATCH",
          body: {
            content: bodyEl.querySelector("#mSliceSummary").value,
            tags: bodyEl.querySelector("#mSliceTags").value,
          },
        });
        flash("切片已更新。", "success");
        await loadSlices();
      } catch (err) { flash(err.message, "error"); }

    } else if (btn.dataset.action === "delete-slice") {
      const bodyEl = await modal.open(
        "确认删除",
        `<p class="mct">确定要删除切片 <strong>#${id}</strong> 吗？此操作不可撤销。</p>`,
        null,
        "确认删除",
      );
      if (!bodyEl) return;
      try {
        await api(`/admin/memory/slices/${id}`, { method: "DELETE" });
        flash("切片已删除。", "success");
        await loadSlices();
      } catch (err) { flash(err.message, "error"); }
    }
  });

  // 长期记忆 事件委托
  document.getElementById("longMemoryList").addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-action]");
    if (!btn) return;
    const id = parseInt(btn.dataset.id, 10);
    const item = state.longMemItems.find((m) => m.id === id) || {};

    if (btn.dataset.action === "edit-longmem") {
      const bodyEl = await modal.open(
        `编辑长期记忆 #${id}`,
        `<div class="mf">
           <label class="ml">标题</label>
           <input id="mMemTitle" class="mi" type="text">
         </div>
         <div class="mf">
           <label class="ml">内容</label>
           <textarea id="mMemContent" class="mta" rows="6"></textarea>
         </div>
         <div class="mf">
           <label class="ml">权重（0 ~ 1）</label>
           <input id="mMemWeight" class="mi" type="number" min="0" max="1" step="0.05">
         </div>`,
        (el) => {
          el.querySelector("#mMemTitle").value = item.title || "";
          el.querySelector("#mMemContent").value = item.content || "";
          el.querySelector("#mMemWeight").value = item.fact_weight ?? 1;
        },
      );
      if (!bodyEl) return;
      try {
        await api(`/admin/memory/long_term/${id}`, {
          method: "PATCH",
          body: {
            title: bodyEl.querySelector("#mMemTitle").value,
            content: bodyEl.querySelector("#mMemContent").value,
            fact_weight: parseFloat(bodyEl.querySelector("#mMemWeight").value) || 1,
          },
        });
        flash("长期记忆已更新。", "success");
        await loadLongMemories();
      } catch (err) { flash(err.message, "error"); }

    } else if (btn.dataset.action === "delete-longmem") {
      const bodyEl = await modal.open(
        "确认删除",
        `<p class="mct">确定要删除长期记忆 <strong>#${id}</strong> 吗？此操作不可撤销。</p>`,
        null,
        "确认删除",
      );
      if (!bodyEl) return;
      try {
        await api(`/admin/memory/long_term/${id}`, { method: "DELETE" });
        flash("长期记忆已删除。", "success");
        await loadLongMemories();
      } catch (err) { flash(err.message, "error"); }
    }
  });

  // 核心档案 事件委托
  document.getElementById("coreFactsList").addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-action]");
    if (!btn) return;
    const id = parseInt(btn.dataset.id, 10);
    const item = state.coreFactItems.find((f) => f.id === id) || {};

    if (btn.dataset.action === "edit-corefact") {
      const bodyEl = await modal.open(
        `编辑核心档案 #${id}`,
        `<div class="mf">
           <label class="ml">标题</label>
           <input id="mFactTitle" class="mi" type="text">
         </div>
         <div class="mf">
           <label class="ml">内容</label>
           <textarea id="mFactContent" class="mta" rows="5"></textarea>
         </div>`,
        (el) => {
          el.querySelector("#mFactTitle").value = item.title || "";
          el.querySelector("#mFactContent").value = item.content || "";
        },
      );
      if (!bodyEl) return;
      try {
        await api(`/admin/memory/core_facts/${id}`, {
          method: "PATCH",
          body: {
            title: bodyEl.querySelector("#mFactTitle").value.trim(),
            content: bodyEl.querySelector("#mFactContent").value.trim(),
          },
        });
        flash("核心档案已更新。", "success");
        await loadCoreFacts();
      } catch (err) { flash(err.message, "error"); }

    } else if (btn.dataset.action === "delete-corefact") {
      const bodyEl = await modal.open(
        "确认删除",
        `<p class="mct">确定要删除核心档案 <strong>#${id}</strong> 吗？此操作不可撤销。</p>`,
        null,
        "确认删除",
      );
      if (!bodyEl) return;
      try {
        await api(`/admin/memory/core_facts/${id}`, { method: "DELETE" });
        flash("核心档案已删除。", "success");
        await loadCoreFacts();
      } catch (err) { flash(err.message, "error"); }
    }
  });
}

async function openAddCoreFactModal() {
  const bodyEl = await modal.open(
    "新增核心档案",
    `<div class="mf">
       <label class="ml">标题</label>
       <input id="mFactTitle" class="mi" type="text" placeholder="例如：表白日期">
     </div>
     <div class="mf">
       <label class="ml">内容</label>
       <textarea id="mFactContent" class="mta" rows="5" placeholder="例如：2025年12月25日"></textarea>
     </div>`,
  );
  if (!bodyEl) return;
  const title = bodyEl.querySelector("#mFactTitle").value.trim();
  const content = bodyEl.querySelector("#mFactContent").value.trim();
  if (!title && !content) {
    flash("请至少填写标题或内容。", "error");
    return;
  }
  try {
    await api("/admin/memory/core_facts", {
      method: "POST",
      body: { title: title || "未命名", content: content || "" },
    });
    flash("核心档案已添加。", "success");
    await loadCoreFacts();
  } catch (err) { flash(err.message, "error"); }
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
    document.getElementById("gatewayHealthChip").textContent = health.status === "ok" ? "🟢 正常" : "🔴 异常";

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

    _metricIdx = 0;
    const metrics = [
      makeMetricCard("网关状态", health.status === "ok" ? "正常" : "异常", `时间：${safeText(health.timestamp)}`),
      makeMetricCard("总消息数", numberText(stats.total_messages), `今日新增：${numberText(stats.today_messages)}`),
      makeMetricCard("总会话数", numberText(stats.total_conversations), "一共有多少段聊天"),
      makeMetricCard("日记条目", numberText(cardStatus.diary_entries_total ?? 0), `legacy 记忆卡：${numberText(cardStatus.total_cards)} · 向量：${numberText(cardStatus.card_vector_store_size)}`),
      makeMetricCard("消息向量", numberText(vectorStatus.vector_store_size), `待处理分块：${numberText(vectorStatus.pending_chunks)}`),
      makeMetricCard("待审核画像", numberText(stats.reviews_pending), "需要人工确认后才会生效"),
      makeMetricCard("长期记忆", numberText(stats.long_term_total), `已做向量：${numberText(systemStatus.vectors.long_term_vectors)}`),
      makeMetricCard("Token 输入", numberText(stats.total_token_input), "worker 累计消耗输入 token"),
      makeMetricCard("Token 输出", numberText(stats.total_token_output), "worker 累计消耗输出 token"),
    ];

    document.getElementById("overviewCards").innerHTML = metrics.join("");
    document.getElementById("providerSummary").innerHTML = renderProviderSummary(models.data || []);
  } catch (error) {
    flash(error.message, "error");
  }
}

async function loadMemorySection() {
  await Promise.all([loadCardStatus(), loadCardList(), loadReviews(), loadDiaries(), loadSlices(), loadLongMemories(), loadCoreFacts()]);
}

async function loadCardStatus() {
  try {
    if (!state.token) {
      document.getElementById("cardStatusSummary").innerHTML = renderNeedToken("先填管理员口令，再读取记忆卡状态。");
      return;
    }
    const data = await api("/admin/cards/status");
    const summary = [
      summaryLine("日记条目 (diary_entries)", numberText(data.diary_entries_total ?? 0)),
      summaryLine("legacy 记忆卡 (向量管线)", numberText(data.total_cards)),
      summaryLine("已做向量", numberText(data.embedded_cards)),
      summaryLine("未做向量", numberText(data.pending_embedding)),
      summaryLine("摘要架构", "每日后台展示来自 diary_entries"),
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
      <article class="list-item-base">
        <p class="text-xs font-semibold text-gray-800">${safeText(item.worker_name || "memory_worker")} · <span class="text-indigo-500">${safeText(item.status || "unknown")}</span></p>
        <p class="text-[11px] text-gray-400 mt-0.5">模式：${safeText(item.run_mode || "manual")} · 阶段：${safeText(item.phase || "queued")}</p>
        <p class="text-[11px] text-gray-400">输入 ${numberText(item.token_input)} · 输出 ${numberText(item.token_output)} · 合计 ${numberText(item.token_total)} tokens</p>
        <p class="text-[11px] text-gray-500 mt-0.5">${safeText(item.message || item.error || "暂无备注")}</p>
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
      document.getElementById("jobStatusChip").textContent = "";
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
    const chipEl = document.getElementById("jobStatusChip");
    const chipText = jobChipText(cardJob, vectorJob, latestWorker);
    if (chipText !== "当前没有长任务在跑") {
      chipEl.textContent = `⚙️ ${chipText}`;
      chipEl.classList.remove("hidden");
    } else {
      chipEl.classList.add("hidden");
    }
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
        <article class="card-item-base">
          <p class="text-xs font-semibold text-gray-800">${safeText(item.title || "长期记忆")} <span class="font-normal text-gray-400">· 最终分 ${safeText(String(score.Score_final ?? item.score ?? ""))}</span></p>
          <div class="card-text-wrap mt-1 mb-2">
            <p class="card-text">${safeText(item.content || "")}</p>
            <button class="expand-btn" type="button">展开阅读</button>
          </div>
          <div class="grid gap-1">
            ${summaryLine("S_vec", score.S_vec ?? "")}
            ${summaryLine("S_key", score.S_key ?? "")}
            ${summaryLine("W_fact", score.W_fact ?? "")}
            ${summaryLine("D_time", score.D_time ?? "")}
            ${summaryLine("hits / B_hits", `${score.hits ?? ""} / ${score.B_hits ?? ""}`)}
            ${summaryLine("Score_final", score.Score_final ?? "")}
          </div>
        </article>
      `;
    }).join("") : `<article class="card-item-base"><p class="text-xs font-semibold text-gray-800">长期记忆</p><p class="text-xs text-gray-400 mt-0.5">没有查到长期记忆命中。</p></article>`;
    const legacyHtml = (data.legacy_card_results || []).length
      ? (data.legacy_card_results || []).map((item) => `
        <article class="card-item-base">
          <p class="text-xs font-semibold text-gray-800">旧记忆卡 · ${safeText(item.date || "未知日期")} <span class="font-normal text-gray-400">· 分数 ${safeText(String(item.score ?? ""))}</span></p>
          <div class="card-text-wrap mt-1"><p class="card-text">${safeText(item.summary || "")}</p><button class="expand-btn" type="button">展开阅读</button></div>
        </article>
      `).join("")
      : `<article class="card-item-base"><p class="text-xs font-semibold text-gray-800">旧记忆卡</p><p class="text-xs text-gray-400 mt-0.5">没有兼容层命中。</p></article>`;
    const keywordHtml = (data.keyword_results || []).length
      ? (data.keyword_results || []).map((item) => `
        <article class="card-item-base">
          <p class="text-xs font-semibold text-gray-800">关键词回退 · ${safeText(item.created_at || "").slice(0, 10)}</p>
          <div class="card-text-wrap mt-1"><p class="card-text">${safeText(item.content || "")}</p><button class="expand-btn" type="button">展开阅读</button></div>
        </article>
      `).join("")
      : `<article class="card-item-base"><p class="text-xs font-semibold text-gray-800">关键词回退</p><p class="text-xs text-gray-400 mt-0.5">没有关键词回退命中。</p></article>`;
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
      <article class="card-item-base">
        <p class="text-xs font-semibold text-gray-800">${safeText(item.review_type || "persona")} <span class="font-normal text-gray-400">#${safeText(item.id)}</span></p>
        <p class="text-[11px] text-gray-500 mt-0.5 mb-2">${safeText(item.diff_summary || "暂无变化摘要")}</p>
        ${renderReviewDiff(item)}
        <div class="flex flex-wrap gap-2 mt-2">
          <button class="btn-primary btn-sm" data-review-action="approve" data-review-id="${safeText(item.id)}">同意生效</button>
          <button class="btn-ghost  btn-sm" data-review-action="edit"    data-review-id="${safeText(item.id)}">修改后通过</button>
          <button class="btn-danger btn-sm" data-review-action="reject"  data-review-id="${safeText(item.id)}">拒绝</button>
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
    let note = "";
    let payload = null;

    if (action === "edit") {
      const detail = await api(`/admin/reviews/${reviewId}`);
      const cur = detail.edited_payload_json || detail.proposed_payload_json || {};
      const fallback = detail.active_profile_fallback || {};
      const hasPersona = cur.persona && Object.keys(cur.persona).length > 0;
      const hasRel = cur.relationship && Object.keys(cur.relationship).length > 0;
      const persona = hasPersona ? cur.persona : (fallback.persona || {});
      const rel = hasRel ? cur.relationship : (fallback.relationship || {});

      const bodyEl = await modal.open(
        "修改后通过",
        `<div class="mf">
           <label class="ml">稳定特征（每行一条）</label>
           <textarea id="mStableTraits" class="mta" rows="3"></textarea>
         </div>
         <div class="mf">
           <label class="ml">当前需要（每行一条）</label>
           <textarea id="mCurrentNeeds" class="mta" rows="3"></textarea>
         </div>
         <div class="mf">
           <label class="ml">偏好（每行一条）</label>
           <textarea id="mPreferences" class="mta" rows="3"></textarea>
         </div>
         <div class="mf">
           <label class="ml">关系温度</label>
           <input id="mTemperature" class="mi" type="text">
         </div>
         <div class="mf">
           <label class="ml">关系变化（每行一条）</label>
           <textarea id="mChanges" class="mta" rows="3"></textarea>
         </div>
         <div class="mf">
           <label class="ml">审核备注（可选）</label>
           <input id="mReviewNote" class="mi" type="text" placeholder="可选">
         </div>`,
        (el) => {
          el.querySelector("#mStableTraits").value = (persona.stable_traits || []).join("\n");
          el.querySelector("#mCurrentNeeds").value = (persona.current_needs || []).join("\n");
          el.querySelector("#mPreferences").value = (persona.preferences || []).join("\n");
          el.querySelector("#mTemperature").value = rel.temperature || "";
          el.querySelector("#mChanges").value = (rel.changes || []).join("\n");
        },
        "修改并通过",
      );
      if (!bodyEl) return;

      const lines = (text) => text.split("\n").map((s) => s.trim()).filter(Boolean);
      note = bodyEl.querySelector("#mReviewNote").value;
      payload = {
        persona: { ...persona, stable_traits: lines(bodyEl.querySelector("#mStableTraits").value), current_needs: lines(bodyEl.querySelector("#mCurrentNeeds").value), preferences: lines(bodyEl.querySelector("#mPreferences").value) },
        relationship: { ...rel, temperature: bodyEl.querySelector("#mTemperature").value, changes: lines(bodyEl.querySelector("#mChanges").value) },
      };

    } else {
      const label = action === "approve" ? "同意生效" : "确认拒绝";
      const bodyEl = await modal.open(
        label,
        `<div class="mf">
           <label class="ml">审核备注（可选）</label>
           <input id="mReviewNote" class="mi" type="text" placeholder="可选">
         </div>`,
        null,
        label,
      );
      if (!bodyEl) return;
      note = bodyEl.querySelector("#mReviewNote").value;
    }

    const options = { method: "POST", body: { note } };
    if (payload) options.body.payload = payload;
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
      <article class="card-item-base">
        <p class="text-xs font-semibold text-gray-800">${safeText(item.entry_date || "未知日期")} <span class="font-normal text-gray-400">· 心情 ${numberText(item.mood_score)}/10 ${safeText(item.mood_label || "")}</span></p>
        <div class="card-text-wrap mt-1">
          <p class="card-text">${safeText(item.content || "")}</p>
          <button class="expand-btn" type="button">展开阅读</button>
        </div>
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
    const limit = state.slicesLimit || 100;
    const offset = state.slicesOffset || 0;
    const status = state.slicesStatus || "";
    const qs = new URLSearchParams({ limit, offset });
    if (status) qs.set("status", status);
    const data = await api(`/admin/memory/slices?${qs.toString()}`);
    const items = data.items || [];
    state.slicesTotal = data.total ?? items.length;
    state.sliceItems = items;
    const total = state.slicesTotal;
    const page = Math.floor(offset / limit) + 1;
    const totalPages = Math.max(1, Math.ceil(total / limit));
    const slicesPageEl = document.getElementById("slicesPageText");
    if (slicesPageEl) slicesPageEl.textContent = `第 ${page} 页 / 共 ${totalPages} 页（共 ${total} 条）`;
    document.getElementById("sliceList").innerHTML = items.length ? items.map((item) => `
      <article class="card-item-base">
        <p class="text-xs font-semibold text-gray-800">切片 #${safeText(item.id)} ${renderSliceStatusBadge(item.status || "in_pool")}</p>
        <p class="text-[11px] text-gray-400 mt-0.5 mb-1">${_fmtSourceDateRange(item.source_date_start, item.source_date_end)}消息 ${safeText(item.msg_id_start)}–${safeText(item.msg_id_end)} · 共 ${safeText(item.message_count)} 条</p>
        <div class="flex flex-wrap gap-2 mt-2">
          ${item.type ? `<span style="font-size:10px;padding:2px 8px;border-radius:999px;background:rgba(196,120,78,0.10);color:#C4784E;font-weight:800;display:inline-block;line-height:1.2;">Type:${safeText(item.type)}</span>` : ""}
          ${item.context_anchor ? `<span style="font-size:10px;padding:2px 8px;border-radius:999px;background:rgba(40,38,31,0.05);color:#6B6760;font-weight:800;display:inline-block;line-height:1.2;">Ctx:${safeText(item.context_anchor)}</span>` : ""}
          ${(item.score !== undefined || item.hits !== undefined) ? `<span style="font-size:10px;padding:2px 8px;border-radius:999px;background:rgba(40,38,31,0.05);color:#6B6760;font-weight:800;display:inline-block;line-height:1.2;">Score:${safeText(fmtScore(item.score))} | Hits:${safeText(item.hits ?? 0)}</span>` : ""}
          ${item.first_impact ? renderFirstImpact(item.first_impact) : ""}
        </div>
        <div class="card-text-wrap">
          <p class="card-text">${safeText(item.content || item.summary || "（暂无内容）")}</p>
          <button class="expand-btn" type="button">展开阅读</button>
        </div>
        <div class="flex flex-wrap gap-2 mt-2">
          <button class="btn-ghost btn-sm" data-action="edit-slice"   data-id="${safeText(item.id)}">编辑</button>
          <button class="btn-danger btn-sm" data-action="delete-slice" data-id="${safeText(item.id)}">删除</button>
        </div>
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
    state.longMemItems = items;
    document.getElementById("longMemoryList").innerHTML = items.length ? items.map((item) => `
      <article class="card-item-base">
        <p class="text-xs font-semibold text-gray-800">${safeText(item.title || "长期记忆")} <span class="font-normal text-gray-400">${safeText(item.memory_type || "summary")}</span></p>
        <p class="text-[11px] text-gray-400 mt-0.5 mb-1">${_fmtSourceDateRange(item.source_date_start, item.source_date_end)}权重 ${safeText(item.fact_weight)} · 半衰期 ${safeText(item.half_life_days)} 天 · 命中 ${safeText(item.hits)} 次</p>
        <div class="card-text-wrap">
          <p class="card-text">${safeText(item.content || "（暂无内容）")}</p>
          <button class="expand-btn" type="button">展开阅读</button>
        </div>
        <div class="flex flex-wrap gap-2 mt-2">
          <button class="btn-ghost  btn-sm" data-action="edit-longmem"   data-id="${safeText(item.id)}">编辑</button>
          <button class="btn-danger btn-sm" data-action="delete-longmem" data-id="${safeText(item.id)}">删除</button>
        </div>
      </article>
    `).join("") : "还没有长期记忆。";
  } catch (error) {
    flash(error.message, "error");
  }
}

async function loadCoreFacts() {
  const el = document.getElementById("coreFactsList");
  if (!el) return;
  try {
    if (!state.token) {
      el.innerHTML = renderNeedToken("先填管理员口令，再读取核心档案。");
      return;
    }
    const data = await api("/admin/memory/core_facts");
    const items = data.items || [];
    state.coreFactItems = items;
    el.innerHTML = items.length ? items.map((item) => `
      <article class="card-item-base">
        <p class="text-xs font-semibold text-gray-800">${safeText(item.title || "未命名")}</p>
        <div class="card-text-wrap">
          <p class="card-text">${safeText(item.content || "（暂无内容）")}</p>
          <button class="expand-btn" type="button">展开阅读</button>
        </div>
        <div class="flex flex-wrap gap-2 mt-2">
          <button class="btn-ghost btn-sm" data-action="edit-corefact"   data-id="${safeText(item.id)}">编辑</button>
          <button class="btn-danger btn-sm" data-action="delete-corefact" data-id="${safeText(item.id)}">删除</button>
        </div>
      </article>
    `).join("") : "还没有核心档案，点击「新增」添加。";
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
    return `<div class="list-item-base"><p class="text-xs font-semibold text-gray-800">还没有模型数据</p><p class="text-xs text-gray-400 mt-0.5">可能是口令没填，或接口暂时无法读取。</p></div>`;
  }
  return models
    .slice(0, 12)
    .map((model) => `
      <div class="list-item-base">
        <p class="text-xs font-semibold text-gray-800">${safeText(model.provider || "未命名提供商")}</p>
        <p class="text-xs text-gray-400 mt-0.5">前缀：${safeText((model.prefixes || []).join(", ") || "无")}</p>
      </div>
    `)
    .join("");
}

function renderCardList(cards) {
  if (!cards.length) {
    return "没有找到日记条目。";
  }
  return cards
    .map((card) => {
      const range = _fmtSourceDateRange(card.source_date_start, card.source_date_end);
      const dateStr = range ? range.replace(/ · $/, "") : (card.date || card.entry_date || "未知日期");
      const title = card.title || "";
      const mood = [card.mood_label, card.mood_score != null ? `${card.mood_score}/10` : ""].filter(Boolean).join(" · ");
      return `
      <article class="card-item-base">
        <p class="text-sm font-semibold text-gray-800">${safeText(dateStr)}</p>
        ${title ? `<p class="text-xs font-medium text-amber-800/90 mt-0.5">${safeText(title)}</p>` : ""}
        <p class="text-[11px] text-gray-400 mt-0.5 mb-1">${safeText(mood || "心情 —")}</p>
        <div class="card-text-wrap">
          <p class="card-text">${safeText(card.content || "")}</p>
          <button class="expand-btn" type="button">展开阅读</button>
        </div>
      </article>
    `;
    })
    .join("");
}

function renderSearchResults(results) {
  if (!results.length) {
    return "没有搜到相关记忆卡。";
  }
  return results
    .map((item) => {
      const range = _fmtSourceDateRange(item.source_date_start, item.source_date_end);
      const dateStr = range ? range.replace(/ · $/, "") : (item.date || "未知日期");
      return `
      <article class="card-item-base">
        <p class="text-sm font-semibold text-gray-800">${safeText(dateStr)} <span class="font-normal text-gray-400 text-xs">· 分数 ${safeText(String(item.score ?? ""))}</span></p>
        <p class="text-[11px] text-gray-400 mt-0.5 mb-1">标签：${safeText(item.tags || "无")}</p>
        <div class="card-text-wrap">
          <p class="card-text">${safeText(item.summary || "")}</p>
          <button class="expand-btn" type="button">展开阅读</button>
        </div>
      </article>
    `;
    })
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

const METRIC_COLORS = [
  "bg-purple-100/80",
  "bg-pink-100/80",
  "bg-emerald-100/80",
  "bg-sky-100/80",
];
let _metricIdx = 0;

function makeMetricCard(title, value, help) {
  const color = METRIC_COLORS[_metricIdx++ % METRIC_COLORS.length];
  return `
    <article class="${color} rounded-2xl border border-white/50 p-4">
      <span class="text-[11px] font-bold uppercase tracking-wide text-gray-500">${safeText(title)}</span>
      <strong class="block text-2xl font-bold tracking-tight text-gray-900 my-1.5">${safeText(value)}</strong>
      <p class="text-[11px] text-gray-400">${safeText(help)}</p>
    </article>
  `;
}

function renderNeedToken(message) {
  return `<div class="px-3 py-2 rounded-xl bg-white/55 border border-white/70 text-xs text-gray-400">${safeText(message)}</div>`;
}

function summaryLine(label, value) {
  return `<div class="px-3 py-2 rounded-xl bg-white/55 border border-white/70 text-xs text-gray-600"><strong class="text-gray-700">${safeText(label)}：</strong>${safeText(value)}</div>`;
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

function _fmtSourceDateRange(start, end) {
  if (!start && !end) return "";
  const s = safeText(start || "");
  const e = safeText(end || "");
  if (!s) return e ? `${e} · ` : "";
  if (!e || s === e) return `${s} · `;
  return `${s}–${e} · `;
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

function fmtScore(score) {
  const n = typeof score === "number"
    ? score
    : (score !== undefined && score !== null && score !== "" ? parseFloat(score) : 0);
  if (!isFinite(n)) return "0.0";
  return n.toFixed(2);
}

function sliceStatusBadgeStyle(status) {
  const s = String(status || "").toLowerCase();
  if (s === "in_pool") return { bg: "#DCEBFF", color: "#2F6FED" };
  if (s === "promoted") return { bg: "#EDF6F1", color: "#5A9E7A" };
  if (s === "discarded") return { bg: "#EFEFEF", color: "#9B9690" };
  if (s === "active" || s === "compacted") return { bg: "transparent", color: "#6B6760" };
  return { bg: "transparent", color: "#9B9690" };
}

function renderSliceStatusBadge(status) {
  const label = String(status || "active").trim();
  if (!label) return "";
  const st = sliceStatusBadgeStyle(label);
  return `<span style="font-size:10px;padding:2px 8px;border-radius:999px;background:${st.bg};color:${st.color};font-weight:700;display:inline-block;line-height:1.2;">${safeText(label)}</span>`;
}

function renderFirstImpact(firstImpact) {
  if (!firstImpact) return "";
  return `<span style="font-size:10px;padding:2px 8px;border-radius:999px;background:#FDECEC;color:#C85A5A;font-weight:900;display:inline-block;line-height:1.2;">FIRST</span>`;
}

const FLASH_BASE = "min-w-[240px] max-w-sm px-4 py-3 rounded-2xl text-sm font-medium backdrop-blur-xl border shadow-xl";
const FLASH_TYPES = {
  info:    "bg-indigo-50/95  text-indigo-800  border-indigo-200",
  success: "bg-emerald-50/95 text-emerald-800 border-emerald-200",
  error:   "bg-rose-50/95    text-rose-800    border-rose-200",
};

function flash(message, type = "info") {
  const node = document.getElementById("flashMessage");
  node.textContent = message;
  node.className = `${FLASH_BASE} ${FLASH_TYPES[type] || FLASH_TYPES.info}`;
  if (state.flashTimer) {
    window.clearTimeout(state.flashTimer);
  }
  state.flashTimer = window.setTimeout(() => {
    node.className = "hidden";
  }, 2800);
}

// ── Expand/collapse toggle (global delegation) ──
function bindExpandToggles() {
  document.addEventListener("click", (e) => {
    if (!e.target.matches(".expand-btn")) return;
    const btn = e.target;
    const wrap = btn.closest(".card-text-wrap");
    if (!wrap) return;
    const textEl = wrap.querySelector(".card-text");
    if (!textEl) return;
    const expanded = textEl.classList.toggle("expanded");
    btn.textContent = expanded ? "收起" : "展开";
  });
}

// ── Modal ──
const modal = {
  el: null,
  titleEl: null,
  bodyEl: null,
  confirmBtn: null,
  cancelBtn: null,
  closeBtn: null,
  _resolve: null,

  init() {
    this.el = document.getElementById("editModal");
    this.titleEl = document.getElementById("modalTitle");
    this.bodyEl = document.getElementById("modalBody");
    this.confirmBtn = document.getElementById("modalConfirmBtn");
    this.cancelBtn = document.getElementById("modalCancelBtn");
    this.closeBtn = document.getElementById("modalCloseBtn");

    const close = () => this._close(false);
    this.cancelBtn.addEventListener("click", close);
    this.closeBtn.addEventListener("click", close);
    this.el.addEventListener("click", (e) => { if (e.target === this.el) close(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !this.el.classList.contains("hidden")) close(); });
  },

  // title: string
  // bodyHtml: HTML string
  // afterInsert: (bodyEl) => void  — called right after innerHTML set, use to set .value on inputs
  // confirmLabel: button text
  // Returns Promise<HTMLElement|null>  — null means cancelled
  open(title, bodyHtml, afterInsert = null, confirmLabel = "确认") {
    return new Promise((resolve) => {
      this._resolve = resolve;
      this.titleEl.textContent = title;
      this.bodyEl.innerHTML = bodyHtml;
      if (afterInsert) afterInsert(this.bodyEl);
      this.confirmBtn.textContent = confirmLabel;
      this.el.classList.remove("hidden");
      const first = this.bodyEl.querySelector("textarea, input");
      if (first) setTimeout(() => first.focus(), 40);
      this.confirmBtn.onclick = () => this._close(true);
    });
  },

  _close(confirmed) {
    this.el.classList.add("hidden");
    if (this._resolve) this._resolve(confirmed ? this.bodyEl : null);
    this._resolve = null;
    this.confirmBtn.onclick = null;
  },
};
