import base64, hashlib, json, time, uuid
from pathlib import Path
import numpy as np, trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import math_engine as me, parsers, synth

ROOT = Path(__file__).resolve().parent.parent
MODELS, FRONT = ROOT / "data/models", ROOT / "frontend"
AUDIT = ROOT / "data/audit.jsonl"
app = FastAPI(title="3D Spatial Intelligence Metrology Engine")
PARTS, MESHES, RESULTS = {}, {}, {}

for j in sorted(MODELS.glob("*.json")):
    meta = json.loads(j.read_text()); PARTS[meta["id"]] = meta


def mesh_of(pid):
    if pid not in PARTS: raise HTTPException(404, "unknown part")
    if pid not in MESHES:
        MESHES[pid] = trimesh.load(MODELS / f"{pid}.glb", force="mesh", process=False)
    return MESHES[pid]


def b64(a): return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


@app.get("/api/parts")
def parts():
    return [{k: p[k] for k in ("id", "vertices", "triangles", "bounds", "source")} | {"holes": len(p["features"])} for p in PARTS.values()]


@app.get("/api/parts/{pid}/model.glb")
def model(pid: str):
    mesh_of(pid); return FileResponse(MODELS / f"{pid}.glb", media_type="model/gltf-binary")


@app.get("/api/parts/{pid}/sample")
def sample(pid: str, fmt: str = "csv", severity: float = 1.0, seed: int = 7):
    if fmt not in ("csv", "pcdmis"): raise HTTPException(400, "fmt must be csv|pcdmis")
    return PlainTextResponse(synth.generate(PARTS[pid] if pid in PARTS else mesh_of(pid), mesh_of(pid), severity, seed, fmt),
                             headers={"Content-Disposition": f'attachment; filename="{pid}_synthetic_cmm.{"csv" if fmt=="csv" else "txt"}"'})


def audit(entry):
    prev = "0" * 64
    if AUDIT.exists():
        lines = AUDIT.read_text().strip().splitlines()
        if lines: prev = json.loads(lines[-1])["entry_hash"]
    entry = {**entry, "prev_hash": prev}
    entry["entry_hash"] = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
    with AUDIT.open("a") as f: f.write(json.dumps(entry) + "\n")
    return entry


@app.post("/api/analyze")
async def analyze(part_id: str = Form(""), file: UploadFile = File(...), vendor: str = Form("auto"),
                  machine: str = Form("UNSPECIFIED"), calibration_cert: str = Form("UNSPECIFIED"),
                  datum_ids: str = Form("")):
    mesh = mesh_of(part_id) if part_id else None  # model is optional: report-only analysis
    raw = await file.read()
    if len(raw) > 5_000_000: raise HTTPException(413, "report too large")
    try:
        payload = parsers.parse_report(raw.decode("utf-8", "replace"), vendor)
    except parsers.ParseError as e:
        raise HTTPException(422, str(e))
    pts = payload["inspection_points"]
    nom = np.array([p["nominal"] for p in pts]); act = np.array([p["actual"] for p in pts])
    ids = {s.strip() for s in datum_ids.split(",") if s.strip()}
    w = np.array([1.0 if (not ids or p["feature_id"] in ids) else 1e-6 for p in pts])
    R, T, rms = me.best_fit(nom, act, w)
    aligned = (R @ act.T).T + T
    gdt = me.evaluate_gdt(pts, aligned)
    if mesh is not None:
        V = np.asarray(mesh.vertices)
        raw_field = me.rbf_field(nom, act - nom, V)
        local_field = me.rbf_field(nom, aligned - nom, V)
    else:
        V = raw_field = local_field = None
    sha = hashlib.sha256(raw).hexdigest()
    rid = uuid.uuid4().hex[:12]
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    resid = np.linalg.norm(aligned - nom, axis=1)
    holes = [g for g in gdt if g["check"].startswith("TRUE")]
    summary = {"points": len(pts), "rms_residual": round(rms, 4), "max_residual": round(float(resid.max()), 4),
               "holes": len(holes), "holes_pass": sum(g["status"] == "PASS" for g in holes),
               "saved_by_bonus": sum(g["status"] == "PASS" and not g["pass_without_bonus"] for g in holes),
               "max_local_vertex_shift": round(float(np.linalg.norm(local_field, axis=1).max()), 4) if mesh is not None else None,
               "max_raw_vertex_shift": round(float(np.linalg.norm(raw_field, axis=1).max()), 4) if mesh is not None else None}
    meta = {"analysis_id": rid, "timestamp": ts, "part_id": part_id or "(no model)", "file_name": file.filename, "sha256": sha,
            "machine": machine, "calibration_cert": calibration_cert, "summary": summary}
    entry = audit({k: meta[k] for k in ("analysis_id", "timestamp", "part_id", "file_name", "sha256", "machine", "calibration_cert")})
    payload["part_metadata"].setdefault("part_id", part_id or "(no model)")
    alignment = {"R": R.round(8).tolist(), "T": T.round(5).tolist(), "euler_deg": [round(v, 4) for v in me.euler_deg(R)], "rms": round(rms, 5),
                 "datum_features": sorted(ids) or "ALL (best-fit)"}
    out = {**meta, "payload": payload, "alignment": alignment, "gdt": gdt, "audit_hash": entry["entry_hash"],
           "fields": {"raw": b64(raw_field), "aligned": b64(local_field)} if mesh is not None else None,
           "n_vertices": len(V) if mesh is not None else 0}
    RESULTS[rid] = out
    return out


@app.get("/api/audit")
def audit_log():
    return [json.loads(l) for l in AUDIT.read_text().splitlines()][-50:] if AUDIT.exists() else []


@app.get("/api/report/{rid}", response_class=HTMLResponse)
def report(rid: str):
    r = RESULTS.get(rid)
    if not r: raise HTTPException(404, "analysis not found (results are kept in memory for this session)")
    rows = "".join(f"<tr class='{g['status']}'><td>{g['feature_id']}</td><td>{g['check']}</td><td>{g['deviation']}</td>"
                   f"<td>{g['tol_base'] if g['tol_base'] is not None else '–'}</td><td>{g['bonus']}</td>"
                   f"<td>{g['tol_total'] if g['tol_total'] is not None else '–'}</td><td>{g['status']}</td></tr>" for g in r["gdt"])
    a, s = r["alignment"], r["summary"]
    return f"""<!doctype html><meta charset=utf-8><title>Inspection report {rid}</title><style>
body{{font:14px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #bbb;padding:4px 8px;text-align:left}}.FAIL{{background:#fde2e2}}.PASS{{background:#e3f6e8}}code{{word-break:break-all}}</style>
<h1>Inspection Report — {r['part_id']}</h1>
<p><b>Analysis</b> {rid} · <b>Generated</b> {r['timestamp']} · <b>Machine</b> {r['machine']} · <b>Calibration cert</b> {r['calibration_cert']}</p>
<p><b>Source file</b> {r['file_name']}<br><b>SHA-256</b> <code>{r['sha256']}</code><br><b>Audit-chain hash</b> <code>{r['audit_hash']}</code></p>
<h2>Alignment (6-DoF best-fit)</h2><p>Rotation (deg, XYZ) {a['euler_deg']} · Translation (mm) {a['T']} · RMS {a['rms']} mm · Datums: {a['datum_features']}</p>
<h2>Summary</h2><p>{s['holes_pass']}/{s['holes']} holes in tolerance; {s['saved_by_bonus']} accepted only thanks to MMC bonus; max residual {s['max_residual']} mm.</p>
<h2>GD&amp;T results</h2><table><tr><th>Feature</th><th>Check</th><th>Deviation</th><th>Base tol</th><th>Bonus</th><th>Total tol</th><th>Status</th></tr>{rows}</table>
<p><small>Proof-of-concept output. Not a certified deliverable until the IQ/OQ/PQ and NIST/PTB algorithm validation steps are complete.</small></p>"""


app.mount("/", StaticFiles(directory=FRONT, html=True), name="static")
