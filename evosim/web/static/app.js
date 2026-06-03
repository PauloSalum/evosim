"use strict";
// EvoSim — frontend. Visualizador 3D próprio (Canvas 2D), zero dependências.

const PALETA = ["#58a6ff", "#f0883e", "#2ea043", "#db61a2", "#e3b341", "#a371f7"];
const $ = (id) => document.getElementById(id);
const getJSON = (u) => fetch(u).then((r) => r.json());
const postJSON = (u, b) =>
  fetch(u, { method: "POST", headers: { "Content-Type": "application/json" },
             body: JSON.stringify(b || {}) }).then((r) => r.json());

// ---------------------------------------------------------------- estado
const S = {
  frames: [], dt: 0.05, idx: 0, acc: 0,
  geracaoMostrada: -1, aoVivo: true, rodando: false,
  cam: { yaw: 0.6, pitch: 0.35, dist: 6, target: [0, 0.8, 0] },
};

// --------------------------------------------------------------- helpers
function preencherSelect(sel, itens, valor) {
  sel.innerHTML = "";
  itens.forEach((it) => {
    const o = document.createElement("option");
    o.value = it; o.textContent = it; sel.appendChild(o);
  });
  if (valor) sel.value = valor;
}

function params() {
  return {
    modo: $("modo").value,
    preset: $("preset").value,
    save_path: $("save").value,
    fitness: $("fitness").value,
    algoritmo: $("algoritmo").value,
    controlador: $("controlador").value,
    geracoes: +$("geracoes").value,
    pop: +$("pop").value,
    duracao: +$("duracao").value,
    seed: +$("seed").value,
    workers: +$("workers").value,
    gravidade: [0, +$("gravY").value, 0],
    friccao_solo: +$("friccao").value,
    arrasto_fluido: +$("arrasto").value,
  };
}

function ambienteAtual() {
  return {
    gravidade: [0, +$("gravY").value, 0],
    friccao_solo: +$("friccao").value,
    arrasto_fluido: +$("arrasto").value,
  };
}

// ------------------------------------------------------------ inicializa
async function init() {
  try {
    const op = await getJSON("/api/opcoes");
    preencherSelect($("preset"), op.presets || []);
    preencherSelect($("fitness"), op.fitness || []);
    preencherSelect($("algoritmo"), op.algoritmos || []);
    preencherSelect($("controlador"), op.controladores || []);
    const saves = op.saves || [];
    preencherSelect($("save"), saves.length ? saves : ["(nenhum save)"]);
    montarCompeticao(saves);
    $("cpus-info").textContent = `(0 = todos; você tem ${op.cpus})`;
  } catch (e) {
    $("msg").textContent = "Falha ao carregar opções do servidor: " + e;
  }

  $("modo").addEventListener("change", () => {
    const cont = $("modo").value === "continuar";
    $("campo-preset").style.display = cont ? "none" : "block";
    $("campo-save").style.display = cont ? "block" : "none";
  });

  $("btn-iniciar").addEventListener("click", iniciar);
  $("btn-parar").addEventListener("click", () => postJSON("/api/parar"));
  $("btn-salvar").addEventListener("click", salvar);
  $("ao-vivo").addEventListener("change", (e) => { S.aoVivo = e.target.checked; });

  // competição
  $("comp-modo").addEventListener("change", () => {
    const corr = $("comp-modo").value === "corrida";
    $("bloco-corrida").style.display = corr ? "block" : "none";
    $("bloco-caca").style.display = corr ? "none" : "block";
  });
  $("btn-correr").addEventListener("click", correr);
  $("btn-cacar").addEventListener("click", cacar);

  // sliders de ambiente: rótulo ao vivo + re-simula no playback (debounce).
  const liga = (id, span, fmt) => {
    $(id).addEventListener("input", () => {
      $(span).textContent = fmt(+$(id).value);
      agendarPlayback();
    });
  };
  liga("gravY", "gy-val", (v) => v.toFixed(2));
  liga("friccao", "fr-val", (v) => v.toFixed(2));
  liga("arrasto", "ar-val", (v) => v.toFixed(2));

  configurarMouse();
  setInterval(loopStatus, 1000);
  setInterval(loopFrames, 600);
  requestAnimationFrame(loopRender);
}

function montarCompeticao(saves) {
  const reais = saves.filter((s) => !s.startsWith("("));
  const cont = $("corrida-saves");
  cont.innerHTML = "";
  reais.forEach((p, i) => {
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.value = p; if (i < 2) cb.checked = true;
    lab.appendChild(cb); lab.appendChild(document.createTextNode(p));
    cont.appendChild(lab);
  });
  const lista = reais.length ? reais : ["(nenhum save)"];
  preencherSelect($("caca-cacador"), lista);
  preencherSelect($("caca-presa"), lista);
  if (reais.length > 1) $("caca-presa").value = reais[1];
}

function legenda(grupos) {
  return grupos.map((g, i) =>
    `<span class="leg" style="background:${PALETA[i % PALETA.length]}"></span>${g}`
  ).join(" &nbsp; ");
}

async function correr() {
  const saves = [...document.querySelectorAll("#corrida-saves input:checked")]
    .map((c) => c.value);
  if (saves.length < 2) { $("comp-msg").textContent = "Marque ao menos 2 competidores."; return; }
  $("comp-msg").textContent = "correndo…";
  $("ao-vivo").checked = false; S.aoVivo = false;
  const r = await postJSON("/api/corrida",
    { saves, ambiente: ambienteAtual(), segundos: +$("duracao").value });
  if (r.erro) { $("comp-msg").textContent = r.erro; return; }
  carregarFrames(r);
  const linhas = r.ranking.map((x, i) => `${i + 1}º ${x.nome} — ${x.dist} m`).join("<br>");
  $("comp-msg").innerHTML = legenda(r.grupos) + "<br>" + linhas;
}

async function cacar() {
  const cacador = $("caca-cacador").value, presa = $("caca-presa").value;
  if (cacador.startsWith("(")) { $("comp-msg").textContent = "Sem saves disponíveis."; return; }
  $("comp-msg").textContent = "caçando…";
  $("ao-vivo").checked = false; S.aoVivo = false;
  const r = await postJSON("/api/caca",
    { cacador, presa, ambiente: ambienteAtual(), segundos: +$("duracao").value });
  if (r.erro) { $("comp-msg").textContent = r.erro; return; }
  carregarFrames(r);
  const re = r.resultado;
  const txt = re.capturou
    ? `🎯 CAPTUROU em ${re.tempo}s`
    : `🏃 a presa sobreviveu (distância final ${re.distancia} m)`;
  $("comp-msg").innerHTML = legenda(r.grupos) + "<br>" + txt;
}

async function iniciar() {
  $("msg").textContent = "";
  const r = await postJSON("/api/iniciar", params());
  if (!r.ok) $("msg").textContent = r.erro || "erro ao iniciar";
}

async function salvar() {
  const nome = prompt("Salvar como:", `runs/web_${$("preset").value}.json`);
  if (!nome) return;
  const r = await postJSON("/api/salvar", { caminho: nome });
  $("msg").textContent = r.ok ? `Salvo em ${r.caminho}` : (r.erro || "erro");
  if (r.ok) (await getJSON("/api/opcoes")).saves &&
    preencherSelect($("save"), (await getJSON("/api/opcoes")).saves);
}

// ------------------------------------------------------- polling status
let playbackTimer = null;
function agendarPlayback() {
  if (S.rodando && S.aoVivo) return; // durante treino ao vivo, não interrompe
  $("ao-vivo").checked = false; S.aoVivo = false;
  clearTimeout(playbackTimer);
  playbackTimer = setTimeout(async () => {
    const r = await postJSON("/api/playback",
      { ambiente: ambienteAtual(), segundos: +$("duracao").value });
    if (r.frames && r.frames.length) { carregarFrames(r); }
    else if (r.erro) $("msg").textContent = r.erro;
  }, 350);
}

async function loopStatus() {
  const st = await getJSON("/api/status");
  S.rodando = st.rodando;
  $("btn-iniciar").disabled = st.rodando;
  $("btn-parar").disabled = !st.rodando;
  $("btn-salvar").disabled = !st.tem_melhor;
  let txt = st.rodando ? `treinando — geração ${st.geracao}` :
            (st.concluido ? "concluído" : "parado");
  const ult = st.historico[st.historico.length - 1];
  if (ult) txt += ` | melhor fitness: ${ult.melhor.toFixed(2)}`;
  if (st.erro) { txt = "erro: " + st.erro; $("msg").textContent = st.erro; }
  $("status").textContent = txt;
  desenharGrafico(st.historico);
}

async function loopFrames() {
  if (!S.aoVivo || !S.rodando) return;
  const f = await getJSON("/api/frames");
  if (f.frames && f.frames.length && f.geracao !== S.geracaoMostrada) {
    S.geracaoMostrada = f.geracao;
    carregarFrames(f);
  }
}

function carregarFrames(f) {
  S.frames = f.frames; S.dt = f.dt || 0.05; S.idx = 0; S.acc = 0;
}

// ------------------------------------------------------ render 3D (orbit)
const cv = $("viewer"), cx = cv.getContext("2d");

function projetar(p) {
  const c = S.cam, t = c.target;
  let dx = p[0] - t[0], dy = p[1] - t[1], dz = p[2] - t[2];
  const cy = Math.cos(c.yaw), sy = Math.sin(c.yaw);
  let x1 = dx * cy - dz * sy, z1 = dx * sy + dz * cy, y1 = dy;
  const cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
  let y2 = y1 * cp - z1 * sp, z2 = y1 * sp + z1 * cp, x2 = x1;
  const zc = z2 + c.dist;
  if (zc < 0.15) return null;
  const f = 430;
  return [cv.width / 2 + f * x2 / zc, cv.height / 2 - f * y2 / zc, zc];
}

function linha(a, b, cor, w) {
  const pa = projetar(a), pb = projetar(b);
  if (!pa || !pb) return;
  cx.strokeStyle = cor; cx.lineWidth = w || 1;
  cx.beginPath(); cx.moveTo(pa[0], pa[1]); cx.lineTo(pb[0], pb[1]); cx.stroke();
}

function desenharGrade(cxc, czc) {
  const r = 4;
  for (let i = -r; i <= r; i++) {
    linha([cxc - r, 0, i + czc], [cxc + r, 0, i + czc], "#1d2630", 1);
    linha([cxc + i, 0, czc - r], [cxc + i, 0, czc + r], "#1d2630", 1);
  }
}

function loopRender(now) {
  if (!loopRender.last) loopRender.last = now;
  const dt = (now - loopRender.last) / 1000; loopRender.last = now;
  // avança frames pelo tempo real
  if (S.frames.length) {
    S.acc += dt;
    while (S.acc >= S.dt) { S.idx = (S.idx + 1) % S.frames.length; S.acc -= S.dt; }
  }
  cx.clearRect(0, 0, cv.width, cv.height);
  const frame = S.frames[S.idx];
  if (frame && frame.length) {
    // câmera segue o centro da criatura no eixo de avanço.
    let mx = 0, mz = 0, n = 0;
    frame.forEach((s) => { mx += s[0] + s[3]; mz += s[2] + s[5]; n += 2; });
    if (n) { S.cam.target[0] += ((mx / n) - S.cam.target[0]) * 0.1;
             S.cam.target[2] += ((mz / n) - S.cam.target[2]) * 0.1; }
    desenharGrade(S.cam.target[0], S.cam.target[2]);
    frame.forEach((s) => {
      const a = [s[0], s[1], s[2]], b = [s[3], s[4], s[5]];
      let cor;
      if (s.length > 6) cor = PALETA[s[6] % PALETA.length];       // grupo (corrida/caça)
      else cor = (Math.min(s[1], s[4]) < 0.12) ? "#f0883e" : "#58a6ff";
      linha(a, b, cor, 3);
    });
  } else {
    cx.fillStyle = "#8b949e"; cx.font = "14px system-ui";
    cx.fillText("Inicie um treino ou ajuste a gravidade para ver o 3D.", 24, 30);
  }
  requestAnimationFrame(loopRender);
}

// ---------------------------------------------------------- mouse câmera
function configurarMouse() {
  let arrastando = false, px = 0, py = 0;
  cv.addEventListener("mousedown", (e) => { arrastando = true; px = e.clientX; py = e.clientY; });
  window.addEventListener("mouseup", () => { arrastando = false; });
  window.addEventListener("mousemove", (e) => {
    if (!arrastando) return;
    S.cam.yaw += (e.clientX - px) * 0.01;
    S.cam.pitch = Math.max(-1.4, Math.min(1.4, S.cam.pitch + (e.clientY - py) * 0.01));
    px = e.clientX; py = e.clientY;
  });
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    S.cam.dist = Math.max(2, Math.min(20, S.cam.dist + e.deltaY * 0.01));
  }, { passive: false });
}

// ------------------------------------------------------------ gráfico
const gc = $("grafico"), gx = gc.getContext("2d");
function desenharGrafico(hist) {
  gx.clearRect(0, 0, gc.width, gc.height);
  if (!hist || !hist.length) return;
  const melh = hist.map((h) => h.melhor), med = hist.map((h) => h.media);
  const todos = melh.concat(med);
  let lo = Math.min(...todos), hi = Math.max(...todos);
  if (hi - lo < 1e-6) hi = lo + 1;
  const X = (i) => 30 + i / Math.max(1, hist.length - 1) * (gc.width - 40);
  const Y = (v) => gc.height - 16 - (v - lo) / (hi - lo) * (gc.height - 28);
  const serie = (vals, cor) => {
    gx.strokeStyle = cor; gx.lineWidth = 2; gx.beginPath();
    vals.forEach((v, i) => i ? gx.lineTo(X(i), Y(v)) : gx.moveTo(X(i), Y(v)));
    gx.stroke();
  };
  serie(med, "#8b949e"); serie(melh, "#2ea043");
  gx.fillStyle = "#8b949e"; gx.font = "11px system-ui";
  gx.fillText("fitness (verde=melhor, cinza=média) × geração", 30, 12);
}

init();
