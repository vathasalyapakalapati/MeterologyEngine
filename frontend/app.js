import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const $ = id => document.getElementById(id);
const status = t => $('status').textContent = t;
const NO_MODEL = '__none__';
const state = { part: null, base: null, geo: null, fields: null, result: null, bounds: null, markers: [], maxDev: 1 };

// ---------- scene ----------
const canvas = $('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x10141b);
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 5000); camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, canvas); controls.enableDamping = true;
scene.add(new THREE.AxesHelper(40));
const sliceU = { plane: { value: new THREE.Vector4(0, 0, 1, 0) }, on: { value: true } };

const VS = `attribute vec3 disp; uniform float uEx; uniform float uMax; varying float vDev; varying vec3 vV; varying vec3 vN0; varying vec3 vW; varying vec3 vP;
void main(){ vec3 p = position + disp * uEx; vDev = clamp(length(disp) / uMax, 0.0, 1.0); vP = position; vW = p;
  vec4 mv = modelViewMatrix * vec4(p, 1.0); vV = mv.xyz; gl_Position = projectionMatrix * mv; }`;
const FS = `uniform vec4 uPlane; uniform bool uSliceOn; uniform bool uHeat; uniform bool uTex; uniform sampler2D uMap; uniform vec3 uMin; uniform vec3 uSize;
uniform float uAlpha; uniform vec3 uColor; uniform bool uGhost;
varying float vDev; varying vec3 vV; varying vec3 vW; varying vec3 vP;
vec3 ramp(float t){ vec3 a=vec3(.17,.36,1.),b=vec3(.15,.83,.63),c=vec3(1.,.88,.30),d=vec3(1.,.23,.19);
  return t<.33?mix(a,b,t/.33):t<.66?mix(b,c,(t-.33)/.33):mix(c,d,(t-.66)/.34); }
void main(){
  if(uSliceOn && dot(uPlane.xyz, vW) - uPlane.w > 0.0) discard;
  vec3 n = normalize(cross(dFdx(vV), dFdy(vV))); float l = .35 + .65 * abs(n.z);
  if(uGhost){ gl_FragColor = vec4(uColor * (.6 + .4 * l), uAlpha); return; }
  vec3 base = vec3(.74,.78,.84);
  if(uTex){ vec2 uv = (vP.xy - uMin.xy) / uSize.xy; base = texture2D(uMap, uv).rgb; }
  else if(uHeat) base = ramp(vDev);
  if(!gl_FrontFacing) base *= .45;
  gl_FragColor = vec4(base * l, 1.0);
}`;
const matMain = new THREE.ShaderMaterial({ vertexShader: VS, fragmentShader: FS, side: THREE.DoubleSide, uniforms: {
  uEx: { value: 10 }, uMax: { value: 1 }, uPlane: sliceU.plane, uSliceOn: sliceU.on, uHeat: { value: true }, uTex: { value: false },
  uMap: { value: null }, uMin: { value: new THREE.Vector3() }, uSize: { value: new THREE.Vector3(1, 1, 1) },
  uAlpha: { value: 1 }, uColor: { value: new THREE.Color() }, uGhost: { value: false } } });
const matGhost = matMain.clone();
matGhost.uniforms = THREE.UniformsUtils.clone(matMain.uniforms);
Object.assign(matGhost.uniforms, { uPlane: sliceU.plane, uSliceOn: sliceU.on });
matGhost.uniforms.uEx.value = 0; matGhost.uniforms.uGhost.value = true; matGhost.uniforms.uAlpha.value = 0.22;
matGhost.uniforms.uColor.value.set(0x7aa2ff); matGhost.transparent = true; matGhost.depthWrite = false;
let meshMain, meshGhost; const markerGroup = new THREE.Group(); scene.add(markerGroup);

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false); camera.aspect = r.width / r.height; camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
(function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();

// ---------- part loading ----------
async function loadPart(id) {
  status('loading ' + id + '…');
  const gltf = await new GLTFLoader().loadAsync(`/api/parts/${id}/model.glb`);
  let geo; gltf.scene.traverse(o => { if (o.isMesh && !geo) geo = o.geometry; });
  geo = geo.clone(); geo.applyMatrix4(gltf.scene.children[0]?.matrixWorld ?? new THREE.Matrix4());
  const n = geo.attributes.position.count;
  geo.setAttribute('disp', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  geo.computeBoundingBox();
  for (const m of [meshMain, meshGhost]) if (m) scene.remove(m);
  meshMain = new THREE.Mesh(geo, matMain); meshGhost = new THREE.Mesh(geo, matGhost); meshGhost.renderOrder = 2;
  meshMain.frustumCulled = meshGhost.frustumCulled = false;
  scene.add(meshMain, meshGhost);
  state.geo = geo; state.base = geo.attributes.position.array; state.bounds = geo.boundingBox.clone();
  state.fields = null; state.result = null; clearMarkers();
  const sz = state.bounds.getSize(new THREE.Vector3()), c = state.bounds.getCenter(new THREE.Vector3());
  matMain.uniforms.uMin.value.copy(state.bounds.min); matMain.uniforms.uSize.value.copy(sz);
  const d = sz.length();
  camera.position.copy(c).add(new THREE.Vector3(d * .8, -d * 1.1, d * .7)); controls.target.copy(c); camera.near = d / 100; camera.far = d * 20; camera.updateProjectionMatrix();
  $('partinfo').textContent = `${n.toLocaleString()} vertices · ${(geo.index ? geo.index.count / 3 : n / 3).toLocaleString()} triangles · ${sz.x.toFixed(0)}×${sz.y.toFixed(0)}×${sz.z.toFixed(0)} mm`;
  $('tbl').innerHTML = ''; $('sum').textContent = ''; $('prov').textContent = ''; $('alignbox').hidden = true; $('legend').hidden = true;
  setSliceRange(); drawSlice(); status('ready');
}

// ---------- report-only mode (no CAD model) ----------
function unloadPart() {
  for (const m of [meshMain, meshGhost]) if (m) scene.remove(m);
  meshMain = meshGhost = null; state.geo = state.base = state.fields = state.result = null; clearMarkers();
  state.bounds = new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
  $('partinfo').textContent = 'No model: results table, alignment and report only.';
  $('tbl').innerHTML = ''; $('sum').textContent = ''; $('prov').textContent = ''; $('alignbox').hidden = true; $('legend').hidden = true;
  drawSlice(); status('ready');
}
function frameFromPoints(res) {
  const b = new THREE.Box3(); for (const p of res.payload.inspection_points) b.expandByPoint(new THREE.Vector3(...p.nominal));
  state.bounds = b; const c = b.getCenter(new THREE.Vector3()), d = Math.max(b.getSize(new THREE.Vector3()).length(), 1);
  camera.position.copy(c).add(new THREE.Vector3(d * .8, -d * 1.1, d * .7)); controls.target.copy(c);
  camera.near = d / 100; camera.far = d * 20; camera.updateProjectionMatrix();
}

// ---------- analysis ----------
const f32 = b64 => { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Float32Array(u.buffer); };

async function analyze(blob, name) {
  status('analyzing…');
  const fd = new FormData();
  fd.append('part_id', $('part').value === NO_MODEL ? '' : $('part').value); fd.append('file', blob, name);
  fd.append('machine', $('machine').value); fd.append('calibration_cert', $('cert').value); fd.append('datum_ids', $('datums').value);
  const r = await fetch('/api/analyze', { method: 'POST', body: fd });
  if (!r.ok) { const e = await r.json().catch(() => ({})); status('error'); alert('Analysis failed: ' + (e.detail ?? r.status)); return; }
  const res = await r.json();
  const hasModel = !!res.fields;
  if (hasModel && res.n_vertices !== state.geo.attributes.position.count) { alert('Mesh/field vertex count mismatch'); return; }
  state.result = res; state.fields = hasModel ? { raw: f32(res.fields.raw), aligned: f32(res.fields.aligned) } : null;
  if (!hasModel) frameFromPoints(res);
  applyMode(); renderResults(res); buildMarkers(res); status('analysis ' + res.analysis_id);
}

function applyMode() {
  if (!state.fields) { $('legend').hidden = true; if (state.result) buildMarkers(state.result); return; }
  const mode = document.querySelector('input[name=mode]:checked').value, f = state.fields[mode];
  state.geo.attributes.disp.array.set(f); state.geo.attributes.disp.needsUpdate = true;
  let mx = 1e-9; for (let i = 0; i < f.length; i += 3) mx = Math.max(mx, Math.hypot(f[i], f[i + 1], f[i + 2]));
  state.maxDev = mx; matMain.uniforms.uMax.value = mx;
  $('legend').hidden = false; $('legend').innerHTML = `deviation<i></i><span>0</span> — <span>${mx.toFixed(3)} mm</span>`;
  if (state.result) buildMarkers(state.result);
  drawSlice();
}

function renderResults(r) {
  const s = r.summary, a = r.alignment;
  $('alignbox').hidden = false;
  $('align').innerHTML = `Rot (°) ${a.euler_deg.map(v => v.toFixed(3)).join(', ')}<br>Trans (mm) ${a.T.map(v => v.toFixed(3)).join(', ')}<br>Best-fit RMS ${a.rms} mm · max residual ${s.max_residual} mm`;
  $('sum').innerHTML = `<b style="color:var(--ok)">${s.holes_pass}</b>/${s.holes} holes in tolerance · <span class="bonus">${s.saved_by_bonus} saved by MMC bonus</span>`;
  const rows = r.gdt.filter(g => g.check.startsWith('TRUE')).map(g =>
    `<tr class="${g.status}" data-id="${g.feature_id}"><td>${g.feature_id}</td><td>${g.deviation.toFixed(3)}</td><td>${g.tol_base}</td><td class="bonus">+${g.bonus.toFixed(3)}</td><td>${g.tol_total.toFixed(3)}</td><td class="st">${g.status}</td></tr>`).join('');
  $('tbl').innerHTML = `<table><tr><th>Feature</th><th>Pos Ø</th><th>Base</th><th>Bonus</th><th>Total</th><th></th></tr>${rows}</table>`;
  $('tbl').querySelectorAll('tr[data-id]').forEach(tr => tr.onclick = () => highlight(tr.dataset.id, tr));
  $('prov').innerHTML = `SHA-256 <code>${r.sha256.slice(0, 32)}…</code><br>Audit hash <code>${r.audit_hash.slice(0, 32)}…</code><br>Machine ${r.machine} · Cert ${r.calibration_cert}<br><a href="/api/report/${r.analysis_id}" target="_blank">Open printable report</a>`;
}

// ---------- markers ----------
function clearMarkers() { markerGroup.clear(); state.markers = []; }
function buildMarkers(res) {
  clearMarkers();
  const R = res.alignment.R, T = res.alignment.T, mode = document.querySelector('input[name=mode]:checked').value;
  const status = Object.fromEntries(res.gdt.map(g => [g.feature_id, g.status]));
  const sz = state.bounds.getSize(new THREE.Vector3()).length() * 0.007 || 0.5, ex = +$('ex').value;
  const geo = new THREE.SphereGeometry(sz, 12, 8);
  for (const p of res.payload.inspection_points) {
    const nom = new THREE.Vector3(...p.nominal), act = new THREE.Vector3(...p.actual);
    let a = act;
    if (mode === 'aligned') a = new THREE.Vector3(R[0][0] * act.x + R[0][1] * act.y + R[0][2] * act.z + T[0], R[1][0] * act.x + R[1][1] * act.y + R[1][2] * act.z + T[1], R[2][0] * act.x + R[2][1] * act.y + R[2][2] * act.z + T[2]);
    const pos = nom.clone().add(a.sub(nom).multiplyScalar(ex));
    const st = status[p.feature_id], hole = st === 'PASS' || st === 'FAIL';
    const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: !hole ? 0xaab4c3 : st === 'PASS' ? 0x3ecf8e : 0xff6a5e }));
    m.position.copy(pos); m.userData = { id: p.feature_id, nom, act: a.clone(), hole }; m.scale.setScalar(hole ? 1 : 0.55);
    markerGroup.add(m); state.markers.push(m);
  }
  markerGroup.visible = $('pins').checked;
}
function moveMarkers() { const ex = +$('ex').value; for (const m of state.markers) { const d = m.userData; m.position.copy(d.nom).add(d.act.clone().sub(d.nom).multiplyScalar(ex)); } }
// NB: act stored in the display mode frame; rebuilt on mode change.
function highlight(id, tr) {
  document.querySelectorAll('tr.hl').forEach(e => e.classList.remove('hl')); tr?.classList.add('hl');
  for (const m of state.markers) m.scale.setScalar(m.userData.id === id ? 3 : m.userData.hole ? 1 : 0.55);
  const m = state.markers.find(m => m.userData.id === id); if (m) controls.target.lerp(m.position, 0.6);
}

// ---------- slicing ----------
function sliceNormal() {
  const a = $('axis').value; $('custom').hidden = a !== 'Custom';
  if (a === 'X') return new THREE.Vector3(1, 0, 0); if (a === 'Y') return new THREE.Vector3(0, 1, 0); if (a === 'Z') return new THREE.Vector3(0, 0, 1);
  const v = new THREE.Vector3(+$('cx').value, +$('cy').value, +$('cz').value); return v.lengthSq() ? v.normalize() : new THREE.Vector3(0, 0, 1);
}
let sRange = [0, 1];
function setSliceRange() {
  if (!state.bounds) return; const n = sliceNormal(), b = state.bounds; let lo = 1e9, hi = -1e9;
  for (const x of [b.min.x, b.max.x]) for (const y of [b.min.y, b.max.y]) for (const z of [b.min.z, b.max.z]) { const d = n.x * x + n.y * y + n.z * z; lo = Math.min(lo, d); hi = Math.max(hi, d); }
  sRange = [lo, hi];
}
function sliceOffset() { return sRange[0] + (sRange[1] - sRange[0]) * +$('slpos').value; }
function updateSlice() {
  const n = sliceNormal(), d = sliceOffset();
  sliceU.plane.value.set(n.x, n.y, n.z, d); sliceU.on.value = $('slon').checked; $('slv').textContent = ` offset ${d.toFixed(1)} mm`;
  drawSlice();
}

const sc = $('sc'), ctx = sc.getContext('2d');
function drawSlice() {
  ctx.clearRect(0, 0, sc.width, sc.height); ctx.fillStyle = '#0b0f15'; ctx.fillRect(0, 0, sc.width, sc.height);
  $('slice2d').style.display = $('slon').checked && state.geo ? '' : 'none';
  if (!state.geo || !$('slon').checked) return;
  const n = sliceNormal(), d = sliceOffset(), ex = +$('ex').value;
  const up = Math.abs(n.z) > 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
  const u = new THREE.Vector3().crossVectors(up, n).normalize(), v = new THREE.Vector3().crossVectors(n, u);
  const P = state.base, D = state.geo.attributes.disp.array, idx = state.geo.index?.array;
  const nt = idx ? idx.length / 3 : P.length / 9;
  const nomS = [], actS = [], vecs = [];
  const vert = (i, k, e) => P[3 * i + k] + (e ? D[3 * i + k] * ex : 0);
  const sd = i => n.x * P[3 * i] + n.y * P[3 * i + 1] + n.z * P[3 * i + 2] - d;
  for (let t = 0; t < nt; t++) {
    const tri = idx ? [idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]] : [3 * t, 3 * t + 1, 3 * t + 2];
    const dist = tri.map(sd); const pn = [], pa = [];
    for (let e = 0; e < 3; e++) {
      const i = tri[e], j = tri[(e + 1) % 3], di = dist[e], dj = dist[(e + 1) % 3];
      if ((di > 0) === (dj > 0) || di === dj) continue;
      const s = di / (di - dj), q = [], r = [];
      for (let k = 0; k < 3; k++) { const a = P[3 * i + k] + s * (P[3 * j + k] - P[3 * i + k]); q.push(a); r.push(a + (D[3 * i + k] + s * (D[3 * j + k] - D[3 * i + k])) * ex); }
      pn.push(q); pa.push(r);
    }
    if (pn.length === 2) { nomS.push(pn); actS.push(pa); }
  }
  if (!nomS.length) { ctx.fillStyle = '#8b97a9'; ctx.fillText('no intersection at this offset', 10, 20); return; }
  const to2 = p => [p[0] * u.x + p[1] * u.y + p[2] * u.z, p[0] * v.x + p[1] * v.y + p[2] * v.z];
  let mnx = 1e9, mxx = -1e9, mny = 1e9, mxy = -1e9;
  for (const s of [...nomS, ...actS]) for (const p of s) { const [a, b] = to2(p); mnx = Math.min(mnx, a); mxx = Math.max(mxx, a); mny = Math.min(mny, b); mxy = Math.max(mxy, b); }
  const pad = 14, sc_ = Math.min((sc.width - 2 * pad) / (mxx - mnx || 1), (sc.height - 2 * pad) / (mxy - mny || 1));
  const X = a => pad + (a - mnx) * sc_, Y = b => sc.height - pad - (b - mny) * sc_;
  const draw = (segs, col) => { ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.beginPath(); for (const [p, q] of segs) { const a = to2(p), b = to2(q); ctx.moveTo(X(a[0]), Y(a[1])); ctx.lineTo(X(b[0]), Y(b[1])); } ctx.stroke(); };
  draw(nomS, '#7aa2ff'); draw(actS, '#ff6a5e');
  ctx.strokeStyle = '#ffe14daa'; ctx.beginPath(); const step = Math.max(1, Math.floor(nomS.length / 70));
  for (let i = 0; i < nomS.length; i += step) { const a = to2(nomS[i][0]), b = to2(actS[i][0]); ctx.moveTo(X(a[0]), Y(a[1])); ctx.lineTo(X(b[0]), Y(b[1])); }
  ctx.stroke();
}

// ---------- UI wiring ----------
$('ex').oninput = () => { const e = +$('ex').value; $('exv').textContent = e + '×'; matMain.uniforms.uEx.value = e; moveMarkers(); drawSlice(); };
document.querySelectorAll('input[name=mode]').forEach(r => r.onchange = applyMode);
$('nomv').onchange = () => meshGhost && (meshGhost.visible = $('nomv').checked);
$('heat').onchange = () => { matMain.uniforms.uHeat.value = $('heat').checked; };
$('pins').onchange = () => markerGroup.visible = $('pins').checked;
$('heatimg').onchange = e => {
  const f = e.target.files[0]; if (!f) { matMain.uniforms.uTex.value = false; return; }
  new THREE.TextureLoader().load(URL.createObjectURL(f), t => { t.colorSpace = THREE.SRGBColorSpace; matMain.uniforms.uMap.value = t; matMain.uniforms.uTex.value = true; $('heat').checked = false; });
};
['axis', 'cx', 'cy', 'cz'].forEach(i => $(i).onchange = () => { setSliceRange(); updateSlice(); });
$('slpos').oninput = updateSlice; $('slon').onchange = updateSlice;
$('part').onchange = () => $('part').value === NO_MODEL ? unloadPart() : loadPart($('part').value).then(updateSlice);
$('gen').onclick = async () => {
  if ($('part').value === NO_MODEL) { alert('Pick a sample part to generate a synthetic report, or upload your own report.'); return; }
  $('gen').disabled = true;
  try {
    const r = await fetch(`/api/parts/${$('part').value}/sample?fmt=${$('fmt').value}&severity=${$('sev').value}&seed=${Math.floor(Math.random() * 1e4)}`);
    await analyze(await r.blob(), `${$('part').value}_synthetic.${$('fmt').value === 'csv' ? 'csv' : 'txt'}`);
  } finally { $('gen').disabled = false; }
};
$('file').onchange = e => { const f = e.target.files[0]; if (f) analyze(f, f.name); e.target.value = ''; };

(async () => {
  resize();
  const parts = await (await fetch('/api/parts')).json();
  $('part').innerHTML = `<option value="${NO_MODEL}">No model (report only)</option>` + parts.map(p => `<option value="${p.id}">${p.id.replace('nist_', '').toUpperCase()} · ${p.holes} holes</option>`).join('');
  $('part').value = NO_MODEL; unloadPart(); updateSlice(); resize();
})();
