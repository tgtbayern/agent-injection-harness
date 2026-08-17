/* Replay, config and experiment pages.
 *
 * No framework and no build step: this is an instrument panel that has to open
 * from a clone with nothing installed. The replay view is the part that earns
 * its keep -- it shows where a payload entered a turn and what happened to the
 * agent's beliefs and ballot immediately afterwards.
 */

const api = {
  async get(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error((await r.text()).slice(0, 300));
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error((await r.text()).slice(0, 300));
    return r.json();
  },
  async del(path) {
    const r = await fetch(path, { method: "DELETE" });
    return r.json();
  },
};

const el = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ------------------------------------------------------------ navigation */

document.querySelectorAll("nav button").forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll("nav button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    el(btn.dataset.page).classList.add("active");
    if (btn.dataset.page === "config") loadConfig();
    if (btn.dataset.page === "experiments") loadExperiments();
  };
});

/* ================================================================ REPLAY */

const state = { log: null, round: 1, player: null };

async function loadGames() {
  const games = await api.get("/api/games?limit=100");
  const select = el("game-select");
  select.innerHTML = games
    .map((g) => {
      const guard = (g.config.guard_layers || []).join("+") || "none";
      const mode = g.config.benign_persuasion ? "benign" : g.config.attack_enabled ? "attack" : "clean";
      return `<option value="${g.game_id}">${g.game_id} -- seed ${g.seed} -- ${guard}${
        g.config.evidence_forced ? "+E" : ""} -- ${mode} -- ${g.outcome.winner || "?"} won</option>`;
    })
    .join("");
  if (games.length) loadGame(games[0].game_id);
  else el("game-meta").textContent = "no games yet -- run one above";
}

async function loadGame(id) {
  state.log = await api.get(`/api/games/${id}`);
  state.round = state.log.rounds.length ? state.log.rounds[0].round : 1;
  state.player = null;
  renderMeta();
  renderTimeline();
  renderRounds();
  renderPlayers();
  renderTrace();
}

function renderMeta() {
  const l = state.log;
  const guard = (l.config.guard_layers || []).join("+") || "none";
  const o = l.outcome;
  el("game-meta").innerHTML =
    `game <b>${esc(l.game_id)}</b> &nbsp; seed=${l.seed} &nbsp; model=${esc(l.config.model)} ` +
    `&nbsp; guard=${esc(guard)}${l.config.evidence_forced ? "+E" : ""} ` +
    `&nbsp; steps&le;${l.config.max_react_steps} &nbsp; winner=<b>${esc(o.winner)}</b> ` +
    `&nbsp; tokens=${o.total_prompt_tokens + o.total_completion_tokens} ` +
    `&nbsp; $${(o.total_cost_usd || 0).toFixed(4)} &nbsp; ${o.total_duration_s}s` +
    (o.crashed ? ` <span class="attack">CRASHED: ${esc(o.crash_reason)}</span>` : "");
}

/* One tick per planted payload; filled when some agent's ballot ended up on
   the payload's target that round. Scanning the row tells you what happened in
   a game before reading a single trace. */
function renderTimeline() {
  const l = state.log;
  const planted = l.planted_payloads || [];
  const parts = [];
  l.rounds.forEach((rnd, i) => {
    if (i) parts.push('<span class="tick round-sep"></span>');
    planted
      .filter((p) => p.round === rnd.round)
      .forEach((p) => {
        const hit = rnd.agents.some(
          (a) => a.task === "vote" && a.vote === p.target &&
                 (a.read_payloads || []).some((r) => r.payload_id === p.payload_id));
        parts.push(
          `<span class="tick ${hit ? "hit" : ""}" title="round ${rnd.round}: ${esc(p.payload_id)} ` +
          `by p${p.attacker} at p${p.target}${hit ? " -- landed" : " -- no vote change"}"></span>`);
      });
  });
  el("timeline").innerHTML = parts.join("") || '<span class="muted">no payloads planted</span>';
}

function renderRounds() {
  el("rounds").innerHTML = state.log.rounds
    .map((r) => `<button data-round="${r.round}" class="${r.round === state.round ? "active" : ""}">round ${r.round}</button>`)
    .join("");
  el("rounds").querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      state.round = Number(b.dataset.round);
      renderRounds();
      renderPlayers();
      renderTrace();
    };
  });
}

function currentRound() {
  return state.log.rounds.find((r) => r.round === state.round) || { agents: [], alive: [] };
}

function renderPlayers() {
  const rnd = currentRound();
  const roles = state.log.ground_truth.roles || {};
  const attackers = new Set((state.log.planted_payloads || [])
    .filter((p) => p.round === rnd.round).map((p) => p.attacker));
  const items = [];
  for (let pid = 1; pid <= 8; pid++) {
    const turns = rnd.agents.filter((a) => a.player_id === pid);
    const alive = (rnd.alive || []).includes(pid);
    const exposed = turns.some((t) => (t.read_payloads || []).length);
    const blocked = turns.some((t) => (t.guard_blocks || []).length);
    const human = turns.some((t) => t.is_human);
    items.push(
      `<li data-pid="${pid}" class="${state.player === pid ? "selected" : ""} ${alive ? "" : "dead"}">
        <span class="flag attack">${attackers.has(pid) ? "&#9679;" : ""}</span>
        <span>p${pid}</span>
        <span class="muted">${esc(roles[pid] || "?")}${human ? " (human)" : ""}</span>
        <span style="margin-left:auto">
          ${exposed ? '<span class="attack" title="read a payload">&#9888;</span>' : ""}
          ${blocked ? '<span class="guard" title="a guard blocked an action">&#9635;</span>' : ""}
        </span>
      </li>`);
  }
  el("players").innerHTML = items.join("");
  el("players").querySelectorAll("li").forEach((li) => {
    li.onclick = () => {
      state.player = Number(li.dataset.pid);
      renderPlayers();
      renderTrace();
    };
  });
  const counts = rnd.vote_counts || {};
  el("round-summary").innerHTML =
    `night deaths: ${JSON.stringify(rnd.night_deaths || [])}<br>` +
    `tally: ${esc(JSON.stringify(counts))}<br>exiled: ${rnd.exiled ?? "nobody"}`;
}

/* Highlight the exact payload text inside whatever the agent read. */
function highlightPayloads(text) {
  let out = esc(text);
  (state.log.planted_payloads || []).forEach((p) => {
    const words = String(p.text).split(/\s+/);
    const frag = words.slice(Math.floor(words.length / 3), Math.floor(words.length / 3) + 5).join(" ");
    if (frag && out.includes(esc(frag))) {
      out = out.replace(esc(frag), `<mark title="payload ${esc(p.payload_id)}">${esc(frag)}</mark>`);
    }
  });
  return out;
}

function renderTrace() {
  const rnd = currentRound();
  if (!state.player) {
    el("trace-title").textContent = "react trace";
    el("trace").innerHTML = '<div class="empty">pick a player to see their turn</div>';
    return;
  }
  const turns = rnd.agents.filter((a) => a.player_id === state.player);
  el("trace-title").innerHTML = `react trace &mdash; player ${state.player}, round ${rnd.round}`;
  if (!turns.length) {
    el("trace").innerHTML = '<div class="empty">this player did not act in this round</div>';
    return;
  }

  el("trace").innerHTML = turns.map((turn) => {
    const steps = (turn.react_trace || []).map((s) => {
      const cls = s.injected ? "injected" : s.guard_blocked ? "blocked" : "";
      const flag = s.injected
        ? '<span class="pill attack">payload in tool return</span>'
        : s.guard_blocked
        ? `<span class="pill guard">blocked: ${esc(s.block_reason)}</span>`
        : "";
      return `<div class="step ${cls}">
          <div class="head">step ${s.step} &rarr; <b>${esc(s.action)}</b>
            <span class="muted">${esc(JSON.stringify(s.args || {}))}</span> ${flag}</div>
          ${s.thought ? `<div class="obs">thought: ${esc(s.thought)}</div>` : ""}
          ${s.observation ? `<div class="obs">${highlightPayloads(s.observation)}</div>` : ""}
        </div>`;
    }).join("");

    const diff = beliefDiff(turn.belief_before, turn.belief_after);
    const exposure = (turn.read_payloads || []).map((r) =>
      `<span class="pill attack">${esc(r.payload_id)} via ${esc(r.channel)}</span>`).join(" ");

    return `<div style="margin-bottom:18px">
      <div class="muted mono">task=${esc(turn.task)} &nbsp; steps=${turn.steps_used} &nbsp;
        tokens=${turn.total_tokens} &nbsp; retries=${turn.retries || 0}
        ${turn.forced_terminal ? '&nbsp; <span class="guard">forced</span>' : ""}
        ${turn.fallback_used ? `&nbsp; <span class="attack">fallback: ${esc(turn.fallback_used)}</span>` : ""}</div>
      ${exposure ? `<div style="margin:6px 0">exposed to: ${exposure}</div>` : ""}
      ${steps}
      ${diff ? `<div class="beliefdiff" style="margin-top:8px">belief changes: ${diff}</div>` : ""}
      ${turn.speech ? `<div class="speech">${highlightPayloads(turn.speech)}</div>` : ""}
      ${turn.task === "vote" ? `<div class="mono">ballot: <b>${turn.vote ?? "abstained"}</b></div>` : ""}
    </div>`;
  }).join("");
}

function beliefDiff(before, after) {
  const parts = [];
  Object.keys(after || {}).forEach((pid) => {
    const b = before && before[pid] ? before[pid].suspicion : null;
    const a = after[pid].suspicion;
    if (b !== null && Math.abs(a - b) > 1e-9) {
      parts.push(`<span class="up">p${pid}: ${b.toFixed(2)} &rarr; ${a.toFixed(2)}</span>
        <span class="muted">(${esc(after[pid].reason || "")})</span>`);
    }
  });
  return parts.join("<br>");
}

el("game-select").onchange = (e) => loadGame(e.target.value);

el("run-game").onclick = async () => {
  const guard = el("new-guard").value ? el("new-guard").value.split(",") : [];
  const started = await api.post("/api/games", {
    seed: Number(el("new-seed").value),
    guard_layers: guard,
    attack_enabled: true,
  });
  el("game-meta").textContent = `running ${started.config} ...`;
  const source = new EventSource(`/api/games/${started.stream_id}/stream`);
  source.addEventListener("round_start", (e) => {
    el("game-meta").textContent = `running ... round ${JSON.parse(e.data).round}`;
  });
  source.addEventListener("done", async (e) => {
    source.close();
    await loadGames();
    loadGame(JSON.parse(e.data).game_id);
  });
  source.addEventListener("end", () => source.close());
  source.addEventListener("error", () => source.close());
};

/* ================================================================ CONFIG */

async function loadConfig() {
  const [providers, models] = await Promise.all([
    api.get("/api/providers"),
    api.get("/api/models"),
  ]);

  el("providers").innerHTML =
    "<tr><th>name</th><th>base url</th><th>key</th><th></th></tr>" +
    (providers.map((p) => `<tr>
        <td>${esc(p.name)}</td><td class="mono">${esc(p.base_url)}</td>
        <td class="mono muted">${esc(p.api_key_masked)}</td>
        <td><button class="action danger" data-del="${p.id}">delete</button></td>
      </tr>`).join("") || '<tr><td colspan="4" class="muted">none yet</td></tr>');

  el("providers").querySelectorAll("[data-del]").forEach((b) => {
    b.onclick = async () => { await api.del(`/api/providers/${b.dataset.del}`); loadConfig(); };
  });

  const options = providers.map((p) => `<option value="${p.id}">${esc(p.name)}</option>`).join("");
  el("m-provider").innerHTML = options;
  el("x-model").innerHTML =
    '<option value="">mock (offline)</option>' +
    models.map((m) => `<option value="${m.id}">${esc(m.display_name)}</option>`).join("");

  el("models").innerHTML =
    "<tr><th>display</th><th>model name</th><th>group</th><th>tools</th><th>mode</th><th>probe</th><th></th></tr>" +
    (models.map((m) => `<tr>
        <td>${esc(m.display_name)}</td>
        <td class="mono">${esc(m.model_name)}</td>
        <td class="mono muted">${esc(m.group || "-")}</td>
        <td>${m.supports_tools === null ? '<span class="muted">untested</span>'
             : m.supports_tools ? '<span class="guard">native</span>'
             : '<span class="attack">none</span>'}</td>
        <td class="mono">${esc(m.tool_mode)}</td>
        <td><button class="action" data-probe="${m.id}">test</button>
            <span id="probe-${m.id}" class="mono muted"></span></td>
        <td><button class="action danger" data-delm="${m.id}">delete</button></td>
      </tr>`).join("") || '<tr><td colspan="7" class="muted">none yet</td></tr>');

  el("models").querySelectorAll("[data-delm]").forEach((b) => {
    b.onclick = async () => { await api.del(`/api/models/${b.dataset.delm}`); loadConfig(); };
  });
  el("models").querySelectorAll("[data-probe]").forEach((b) => {
    b.onclick = () => probeModel(b.dataset.probe);
  });
}

/* The probe result is the point of the config page: three plain answers
 * (reachable / tools / latency) and, on failure, a fix rather than the
 * gateway's own error text. */
async function probeModel(id) {
  const out = el(`probe-${id}`);
  out.textContent = "testing...";
  try {
    const r = await api.post(`/api/models/${id}/probe`);
    const bits = [
      r.reachable ? `<span class="guard">reachable ${r.latency_ms}ms</span>`
                  : '<span class="attack">unreachable</span>',
      r.native_tools ? '<span class="guard">native tools</span>'
                     : '<span class="belief">json fallback</span>',
      r.reports_usage ? "" : '<span class="belief">no usage field</span>',
    ].filter(Boolean);
    out.innerHTML = bits.join(" &middot; ");
    if (r.error) out.innerHTML += `<div class="hint">${esc(r.hint || r.error)}</div>`;
    (r.notes || []).forEach((n) => { out.innerHTML += `<div class="hint">${esc(n)}</div>`; });
    loadConfig();
  } catch (e) {
    let hint = String(e.message);
    try { hint = JSON.parse(hint).detail.hint || hint; } catch (_) {}
    out.innerHTML = `<span class="attack">failed</span><div class="hint">${esc(hint)}</div>`;
  }
}

el("add-provider").onclick = async () => {
  try {
    const created = await api.post("/api/providers", {
      name: el("p-name").value || "relay",
      base_url: el("p-url").value,
      api_key: el("p-key").value,
    });
    el("p-key").value = "";
    el("provider-msg").textContent = created.warning || "saved -- the key stays on the server";
    loadConfig();
  } catch (e) {
    el("provider-msg").innerHTML = `<span class="attack">${esc(e.message)}</span>`;
  }
};

el("add-model").onclick = async () => {
  try {
    await api.post("/api/models", {
      provider_id: el("m-provider").value,
      display_name: el("m-display").value || el("m-name").value,
      model_name: el("m-name").value,
      group: el("m-group").value || null,
    });
    el("model-msg").textContent = "saved -- run the probe before using it in a batch";
    loadConfig();
  } catch (e) {
    el("model-msg").innerHTML = `<span class="attack">${esc(e.message)}</span>`;
  }
};

/* =========================================================== EXPERIMENTS */

let currentExperiment = null;
let pollTimer = null;

async function loadExperiments() {
  const list = await api.get("/api/experiments");
  if (list.length && !currentExperiment) currentExperiment = list[0].id;
  if (currentExperiment) pollExperiment();
}

el("start-exp").onclick = async () => {
  const exp = await api.post("/api/experiments", {
    name: el("x-name").value,
    seeds: Number(el("x-seeds").value),
    workers: Number(el("x-workers").value),
    model_id: el("x-model").value || null,
    include_benign: el("x-benign").checked,
  });
  currentExperiment = exp.id;
  pollExperiment();
};

el("stop-exp").onclick = async () => {
  if (currentExperiment) await api.post(`/api/experiments/${currentExperiment}/stop`);
};

async function pollExperiment() {
  clearTimeout(pollTimer);
  if (!currentExperiment) return;
  const exp = await api.get(`/api/experiments/${currentExperiment}`);
  const p = exp.progress;
  el("exp-progress").innerHTML =
    `<b>${esc(exp.name)}</b> &nbsp; ${esc(exp.status)}<br>` +
    `${p.done}/${p.total} games &nbsp; crashed=${p.crashed} &nbsp; tokens=${p.tokens}` +
    `<progress value="${p.done}" max="${p.total || 1}"></progress>`;
  if (p.done) renderExperimentResults();
  if (exp.status === "running" || exp.status === "queued" || exp.status === "stopping") {
    pollTimer = setTimeout(pollExperiment, 1500);
  }
}

async function renderExperimentResults() {
  const data = await api.get(`/api/experiments/${currentExperiment}/metrics`);
  const rows = Object.entries(data.arms).map(([label, m]) => {
    const speech = (m.injection.speech || {}).hijack_rate;
    const tool = (m.injection.tool_return || {}).hijack_rate;
    const latent = (m.injection.speech || {}).latent_rate;
    return `<tr>
      <td class="mono">${esc(label)}</td>
      <td class="num">${pct(speech)}</td>
      <td class="num">${pct(tool)}</td>
      <td class="num">${pct(latent)}</td>
      <td class="num">${pct(m.overdefense.false_block_rate)}</td>
      <td class="num">${pct(m.village_win_rate)}</td>
      <td class="num">${m.mean_tokens_per_game ?? "-"}</td>
      <td class="num">${m.crashed}</td>
    </tr>`;
  }).join("");
  el("exp-results").innerHTML =
    "<tr><th>guard</th><th class='num'>hijack (speech)</th><th class='num'>hijack (tool)</th>" +
    "<th class='num'>latent</th><th class='num'>false block</th><th class='num'>village win</th>" +
    "<th class='num'>tokens/game</th><th class='num'>crashed</th></tr>" + rows;
}

const pct = (v) => (v === null || v === undefined ? "-" : (100 * v).toFixed(1) + "%");

loadGames();
