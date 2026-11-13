"""Synthetic CMM export generator (NIST STEP gives geometry/features, not measurements)."""
import numpy as np


def generate(part, mesh, severity=1.0, seed=7, fmt="csv", n_surface=14):
    rng = np.random.default_rng(seed)
    feats = part["features"]
    diag = float(np.linalg.norm(np.ptp(np.array(part["bounds"]), axis=0)))
    pts = []
    for f in feats:
        pts.append({"id": f["feature_id"], "type": "INTERNAL_HOLE", "nom": np.array(f["center"]), "axis": np.array(f["axis"]),
                    "size": f["diameter"]})
    samp, fidx = mesh.sample(n_surface, return_index=True)
    for i, (p, fi) in enumerate(zip(samp, fidx)):
        pts.append({"id": f"Surf_{i+1:02d}", "type": "SURFACE_POINT", "nom": p, "axis": mesh.face_normals[fi], "size": None})
    nom = np.array([p["nom"] for p in pts])

    # global rigid mis-fixturing + local springback bump
    ang = np.radians(rng.normal(0, 0.12 * severity, 3))
    cx, cy, cz = np.cos(ang); sx, sy, sz = np.sin(ang)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = rng.normal(0, 0.15 * severity, 3)
    centre = nom.mean(0)
    act = (R @ (nom - centre).T).T + centre + T
    bump_c = nom[rng.integers(len(nom))]
    bump_dir = rng.normal(size=3); bump_dir /= np.linalg.norm(bump_dir)
    amp = 0.35 * severity
    sigma = diag * 0.18
    w = np.exp(-np.sum((nom - bump_c) ** 2, axis=1) / (2 * sigma ** 2))
    act = act + amp * w[:, None] * bump_dir + rng.normal(0, 0.012, act.shape)

    rows = []
    for p, a in zip(pts, act):
        if p["size"] is not None:
            mmc = round(p["size"] - 0.1, 3)
            meas = round(p["size"] + rng.uniform(-0.04, 0.12), 3)
            tol = 0.2
        else:
            mmc = meas = tol = None
        rows.append((p, a, meas, mmc, tol))

    if fmt == "pcdmis":
        L = [f"PART: {part['id'].upper()}", "REV: B", "DATUM: A-B-C", "# synthetic export - generated from NIST CAD feature data", ""]
        for p, a, meas, mmc, tol in rows:
            L.append(f"FEATURE: {p['id']} TYPE: {p['type']}")
            L.append(f"AXIS: {p['axis'][0]:.5f} {p['axis'][1]:.5f} {p['axis'][2]:.5f}")
            for k, i in zip("XYZ", range(3)):
                L.append(f"  {k}  {p['nom'][i]:.4f}  {a[i]:.4f}  {a[i]-p['nom'][i]:.4f}")
            if meas is not None:
                L.append(f"  D  {p['size']:.4f}  {meas:.4f}  {meas-p['size']:.4f}")
                L.append(f"MMC: {mmc}"); L.append(f"TOL: {tol}")
            L.append("")
        return "\n".join(L)
    L = ["feature_id,type,nom_x,nom_y,nom_z,act_x,act_y,act_z,measured_size,mmc_size,tol,axis_i,axis_j,axis_k"]
    for p, a, meas, mmc, tol in rows:
        L.append(",".join([p["id"], p["type"], *(f"{v:.4f}" for v in p["nom"]), *(f"{v:.4f}" for v in a),
                           "" if meas is None else str(meas), "" if mmc is None else str(mmc), "" if tol is None else str(tol),
                           *(f"{v:.5f}" for v in p["axis"])]))
    return "\n".join(L) + "\n"
