"""6-DoF best-fit alignment, GD&T evaluation (position @ MMC, profile), RBF deformation."""
import numpy as np
from scipy.interpolate import RBFInterpolator


def best_fit(nominal, actual, weights=None):
    """Least-squares rigid transform mapping ACTUAL onto NOMINAL (Kabsch). Returns R, T, rms."""
    P, Q = np.asarray(actual, float), np.asarray(nominal, float)
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    pc, qc = (w[:, None] * P).sum(0), (w[:, None] * Q).sum(0)
    H = ((P - pc) * w[:, None]).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    T = qc - R @ pc
    res = (R @ P.T).T + T - Q
    return R, T, float(np.sqrt((np.linalg.norm(res, axis=1) ** 2).mean()))


def euler_deg(R):
    """Small-angle-safe XYZ (roll, pitch, yaw) in degrees."""
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-9:
        return [float(np.degrees(np.arctan2(R[2, 1], R[2, 2]))), float(np.degrees(np.arctan2(-R[2, 0], sy))),
                float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))]
    return [float(np.degrees(np.arctan2(-R[1, 2], R[1, 1]))), float(np.degrees(np.arctan2(-R[2, 0], sy))), 0.0]


def evaluate_gdt(points, aligned_actual, default_tol=0.2, default_mmc_offset=0.1):
    """True position (diametral, perpendicular to axis when known) with MMC bonus; surface profile."""
    rows = []
    for p, a in zip(points, aligned_actual):
        nom = np.array(p["nominal"]); dev = a - nom
        typ = p["type"]
        is_hole = "HOLE" in typ or "PIN" in typ or "CIRCLE" in typ or "BOSS" in typ or p.get("measured_size") is not None
        if not is_hole:
            ax = p.get("axis")
            n = np.array(ax) / np.linalg.norm(ax) if ax else None
            d_n = float(dev @ n) if n is not None else float(np.linalg.norm(dev))
            rows.append({"feature_id": p["feature_id"], "type": typ, "check": "SURFACE_PROFILE",
                         "deviation": round(d_n, 4), "zone": None, "tol_base": None, "bonus": 0.0, "tol_total": None, "status": "INFO"})
            continue
        ax = p.get("axis")
        if ax:
            n = np.array(ax, float); n /= np.linalg.norm(n)
            perp = dev - (dev @ n) * n
        else:
            perp = dev.copy(); perp[2] = 0.0  # assume Z-axis hole when axis unknown
        pos = 2 * float(np.linalg.norm(perp))
        base = p.get("base_position_tolerance") or default_tol
        size, mmc = p.get("measured_size"), p.get("mmc_size")
        internal = "EXTERNAL" not in typ and "PIN" not in typ and "BOSS" not in typ
        bonus = 0.0
        size_status = None
        if size is not None and mmc is not None:
            bonus = max(0.0, (size - mmc) if internal else (mmc - size))
        total = base + bonus
        rows.append({"feature_id": p["feature_id"], "type": typ, "check": "TRUE_POSITION@MMC",
                     "deviation": round(pos, 4), "zone": round(pos, 4), "tol_base": round(base, 4),
                     "bonus": round(bonus, 4), "tol_total": round(total, 4),
                     "size": size, "mmc": mmc,
                     "status": "PASS" if pos <= total + 1e-9 else "FAIL",
                     "pass_without_bonus": pos <= base + 1e-9})
    return rows


def rbf_field(sites, deltas, verts, smoothing=1e-3):
    """Interpolate point deltas over mesh vertices. Falls back gracefully on degenerate layouts."""
    sites, deltas = np.asarray(sites, float), np.asarray(deltas, float)
    attempts = [dict(kernel="thin_plate_spline", degree=1), dict(kernel="thin_plate_spline", degree=0),
                dict(kernel="linear", degree=0)]
    for kw in attempts:
        try:
            f = RBFInterpolator(sites, deltas, smoothing=smoothing, **kw)
            out = np.vstack([f(verts[i:i + 20000]) for i in range(0, len(verts), 20000)])
            return out.astype(np.float32)
        except (np.linalg.LinAlgError, ValueError):
            continue
    raise RuntimeError("RBF interpolation failed")
