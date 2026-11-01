// AppendixAi front end — talks to the Flask server through /api/... (no internet needed)
"use strict";

const UNKNOWN = "Unknown";
const COLORS = { "No appendicitis": "var(--c0)", "Uncomplicated": "var(--c1)", "Complicated": "var(--c2)" };
const $ = (id) => document.getElementById(id);

let CONFIG = null;          // model info + form layout from /api/config
let loadedSample = null;    // { id, true_diagnosis } when a sample patient is loaded
let lastResult = null;      // last prediction (for the report download)

const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const showSpinner = (on) => $("spinner").classList.toggle("hidden", !on);

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Server error (${res.status})`);
  return data;
}

// ============================================================ build the page
function buildSidebar() {
  const m = CONFIG.metrics;
  $("modelName").textContent = CONFIG.model_name;
  $("nFeatures").textContent = CONFIG.n_features;
  $("mAcc").textContent = pct(m["Accuracy"], 1);
  $("mAuc").textContent = m["Macro ROC-AUC (OvR)"].toFixed(2);
  $("mSensApp").textContent = pct(m["Appendicitis sensitivity"]);
  $("mSensComp").textContent = pct(m["Complicated sensitivity"]);
  for (const s of CONFIG.samples) {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = `${s.id} · true: ${s.true_diagnosis}`;
    $("sampleSelect").appendChild(opt);
  }
}

function buildForm() {
  const container = $("formSections");
  for (const section of CONFIG.sections) {
    const details = document.createElement("details");
    details.className = "section";
    details.open = true;
    details.innerHTML = `<summary>${esc(section.title)}</summary><div class="grid"></div>`;
    const grid = details.querySelector(".grid");
    for (const name of section.fields) {
      const f = CONFIG.fields[name];
      const wrap = document.createElement("div");
      wrap.className = "field";
      if (f.type === "number") {
        const tip = `Typical range in training data: ${f.min}–${f.max}. Median: ${f.median}. Leave empty if unknown.`;
        wrap.innerHTML = `<label for="f_${name}"><span>${esc(f.label)}</span><span class="help" title="${esc(tip)}">?</span></label>
          <input type="number" step="any" id="f_${name}" name="${name}" placeholder="—">`;
        const input = wrap.querySelector("input");
        input.addEventListener("input", () => {
          const v = parseFloat(input.value);
          input.classList.toggle("out-of-range", !isNaN(v) && (v < f.min || v > f.max));
        });
      } else {
        const opts = [UNKNOWN, ...f.options].map((o) => `<option value="${esc(o)}">${esc(o)}</option>`).join("");
        wrap.innerHTML = `<label for="f_${name}"><span>${esc(f.label)}</span></label><select id="f_${name}" name="${name}">${opts}</select>`;
      }
      grid.appendChild(wrap);
    }
    container.appendChild(details);
  }
}

function buildPerformance() {
  const ci = CONFIG.accuracy_ci;
  $("perfCaption").textContent = "Locked test set of 156 patients that were never used for training or model selection. " +
    `95% confidence interval for accuracy: ${pct(ci[0], 1)}–${pct(ci[1], 1)}.`;
  $("metricsBody").innerHTML = Object.entries(CONFIG.metrics)
    .map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${v.toFixed(3)}</td></tr>`).join("");
  $("gallery").innerHTML = CONFIG.figures
    .map((f) => `<figure data-src="/figures/${f.file}" data-caption="${esc(f.caption)}">
                   <img src="/figures/${f.file}" alt="${esc(f.caption)}" loading="lazy"><figcaption>${esc(f.caption)}</figcaption></figure>`)
    .join("");
  $("gallery").addEventListener("click", (e) => {
    const fig = e.target.closest("figure");
    if (!fig) return;
    $("lightboxImg").src = fig.dataset.src;
    $("lightboxCaption").textContent = fig.dataset.caption;
    $("lightbox").classList.remove("hidden");
  });
  $("lightbox").addEventListener("click", () => $("lightbox").classList.add("hidden"));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("lightbox").classList.add("hidden"); });
}

function buildAbout() {
  const m = CONFIG.metrics;
  $("aboutText").innerHTML = `
    <p><strong>Goal.</strong> Help clinicians (in research settings) estimate whether a child with abdominal pain has
    appendicitis and, if so, whether it is <strong>complicated</strong> (the group that includes perforation).</p>
    <p><strong>Data.</strong> Regensburg Pediatric Appendicitis Dataset — 780 children with a confirmed diagnosis.
    The model uses ${CONFIG.n_features} routinely collected variables: demographics, symptoms, physical examination,
    blood and urine tests, and the structured ultrasound report.</p>
    <p><strong>Model.</strong> ${esc(CONFIG.model_name)}, tuned with 5-fold cross-validation on 624 training patients and
    tested once on 156 locked patients. Raw ultrasound images were also studied (ResNet18 features), but they did not
    improve accuracy, so the app uses the structured ultrasound report instead.</p>
    <p><strong>Important limitations</strong></p>
    <ul>
      <li>Single hospital, retrospective data; no external validation yet.</li>
      <li>"Complicated" is the dataset's severity label. It covers advanced disease such as perforation but is
          <strong>not a direct perforation test</strong>.</li>
      <li>Complicated cases are detected less reliably (${pct(m["Complicated sensitivity"])} sensitivity) than
          appendicitis itself (${pct(m["Appendicitis sensitivity"])}).</li>
    </ul>
    <p><strong>${esc(CONFIG.disclaimer)}</strong></p>`;
}

// ============================================================ form helpers
function readForm() {
  const patient = {};
  for (const name of Object.keys(CONFIG.fields)) {
    const el = $(`f_${name}`);
    if (CONFIG.fields[name].type === "number") {
      patient[name] = el.value === "" ? null : parseFloat(el.value);
    } else {
      patient[name] = el.value === UNKNOWN ? null : el.value;
    }
  }
  return patient;
}

function fillForm(patient) {
  for (const name of Object.keys(CONFIG.fields)) {
    const el = $(`f_${name}`);
    const v = patient[name];
    if (CONFIG.fields[name].type === "number") {
      el.value = v === null || v === undefined ? "" : Number(v.toFixed(2));
      el.dispatchEvent(new Event("input"));
    } else {
      el.value = v === null || v === undefined ? UNKNOWN : v;
    }
  }
}

function clearForm() {
  fillForm({});
  loadedSample = null;
  lastResult = null;
  $("sampleSelect").value = "";
  $("sampleBanner").classList.add("hidden");
  $("results").classList.add("hidden");
}

async function loadSample(id) {
  if (!id) return;
  showSpinner(true);
  try {
    const data = await api(`/api/sample?id=${encodeURIComponent(id)}`);
    fillForm(data.patient);
    loadedSample = { id: data.id, true_diagnosis: data.true_diagnosis };
    $("sampleBanner").innerHTML = `Loaded <strong>${esc(data.id)}</strong> (true diagnosis: <strong>${esc(data.true_diagnosis)}</strong>). ` +
      "Edit any value, then press <strong>Predict</strong>.";
    $("sampleBanner").classList.remove("hidden");
    $("results").classList.add("hidden");
    switchTab("assess");
  } catch (err) {
    alert(err.message);
  } finally {
    showSpinner(false);
  }
}

// ============================================================ prediction + results
async function predict(event) {
  event.preventDefault();
  showSpinner(true);
  try {
    lastResult = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient: readForm() }),
    });
    showResult(lastResult);
  } catch (err) {
    alert(err.message);
  } finally {
    showSpinner(false);
  }
}

function showResult(res) {
  const p = res.prediction;
  const cls = p.predicted_class;

  $("warnings").innerHTML = res.warnings.map((w) => `<div class="alert alert-warning">${esc(w)}</div>`).join("");
  $("rDiag").textContent = p.diagnosis;
  $("rDiagSub").textContent = `P(appendicitis) = ${pct(p.appendicitis_probability)}`;
  $("rClass").textContent = cls;
  $("rClassSub").textContent = `${pct(p.probabilities[cls])} confidence`;
  $("rRisk").textContent = p.complicated_risk;
  $("rRiskSub").textContent = `P(complicated) = ${pct(p.complicated_probability)}`;

  const messages = {
    "No appendicitis": ["alert-success", "The model finds <strong>no appendicitis</strong> most likely."],
    "Uncomplicated": ["alert-warning", "The model finds <strong>uncomplicated appendicitis</strong> most likely (no signs of complicated disease)."],
    "Complicated": ["alert-error", "The model finds <strong>complicated appendicitis</strong> most likely — this is the group that includes perforated appendicitis."],
  };
  $("rMessage").className = `alert ${messages[cls][0]}`;
  $("rMessage").innerHTML = messages[cls][1];

  const check = $("rCheck");
  if (loadedSample) {
    const ok = loadedSample.true_diagnosis === cls;
    check.className = `alert ${ok ? "alert-success" : "alert-error"}`;
    check.innerHTML = `${ok ? "✅ Correct" : "❌ Incorrect"} — true diagnosis for this sample patient: <strong>${esc(loadedSample.true_diagnosis)}</strong>`;
  } else {
    check.className = "alert hidden";
  }

  $("probBars").innerHTML = Object.entries(p.probabilities).map(([name, value]) => `
    <div class="bar-row"><span>${esc(name)}</span>
      <div class="bar-track"><div class="bar-fill" style="background:${COLORS[name]}" data-w="${(value * 100).toFixed(1)}"></div></div>
      <span class="bar-pct">${pct(value, 1)}</span></div>`).join("");
  requestAnimationFrame(() => document.querySelectorAll(".bar-fill").forEach((b) => (b.style.width = `${b.dataset.w}%`)));

  $("factorsTitle").textContent = `Why? Top factors for “${cls}”`;
  $("factorsBody").innerHTML = res.explanation.map((r) => `
    <tr><td>${esc(r.feature)}</td><td>${esc(r.value)}</td>
      <td class="${r.effect > 0 ? "up" : "down"}">${r.effect > 0 ? "▲ supports" : "▼ against"}</td>
      <td class="num">${r.effect.toFixed(3)}</td></tr>`).join("");

  $("results").classList.remove("hidden");
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function downloadReport() {
  if (!lastResult) return;
  const p = lastResult.prediction;
  const pad = (n) => String(n).padStart(2, "0");
  const d = new Date();
  const lines = [
    "AppendixAi — prediction report",
    `Generated: ${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`,
    "RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE",
    "",
    `Diagnosis: ${p.diagnosis} (P = ${pct(p.appendicitis_probability, 1)})`,
    `Most likely class: ${p.predicted_class}`,
    `Complicated/perforation risk: ${p.complicated_risk} (P = ${pct(p.complicated_probability, 1)})`,
    "",
    "Class probabilities:",
    ...Object.entries(p.probabilities).map(([c, v]) => `  ${c}: ${pct(v, 1)}`),
    "",
    "Inputs:",
    ...Object.entries(lastResult.patient).map(([k, v]) =>
      `  ${CONFIG.fields[k].label}: ${v === null ? "—" : typeof v === "number" ? Number(v.toFixed(2)) : v}`),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "appendixai_report.txt";
  a.click();
  URL.revokeObjectURL(a.href);
}

// ============================================================ tabs + start-up
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
}

async function init() {
  showSpinner(true);
  try {
    CONFIG = await api("/api/config");
    buildSidebar();
    buildForm();
    buildPerformance();
    buildAbout();
    document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
    $("sampleSelect").addEventListener("change", (e) => loadSample(e.target.value));
    $("clearBtn").addEventListener("click", clearForm);
    $("patientForm").addEventListener("submit", predict);
    $("downloadBtn").addEventListener("click", downloadReport);
  } catch (err) {
    document.querySelector(".main").innerHTML =
      `<div class="alert alert-error">Could not reach the AppendixAi server: ${esc(err.message)}<br>
       Start it with <code>python web_app.py</code> and reload this page.</div>`;
  } finally {
    showSpinner(false);
  }
}

init();
