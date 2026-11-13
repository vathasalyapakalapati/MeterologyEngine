"""Vendor report -> universal JSON contract.

Supported: generic CSV (header aliases), PC-DMIS style text blocks, PolyWorks-style
whitespace tables. Regexes live in PATTERNS so new vendor variants are config, not code.
"""
import csv, io, re

ALIASES = {
    "feature_id": ["feature_id", "feature", "name", "id", "label"],
    "type": ["type", "feature_type", "kind"],
    "nx": ["nom_x", "nominal_x", "nx", "x_nom", "xnom"], "ny": ["nom_y", "nominal_y", "ny", "y_nom", "ynom"],
    "nz": ["nom_z", "nominal_z", "nz", "z_nom", "znom"],
    "ax": ["act_x", "actual_x", "ax", "x_act", "xact", "meas_x"], "ay": ["act_y", "actual_y", "ay", "y_act", "yact", "meas_y"],
    "az": ["act_z", "actual_z", "az", "z_act", "zact", "meas_z"],
    "measured_size": ["measured_size", "meas_dia", "diameter_act", "dia_act", "size_act"],
    "mmc_size": ["mmc_size", "mmc", "mmc_dia"],
    "base_position_tolerance": ["base_position_tolerance", "tol", "tolerance", "pos_tol", "position_tol"],
    "ai": ["axis_i", "ai"], "aj": ["axis_j", "aj"], "ak": ["axis_k", "ak"],
}

# PC-DMIS style:  FEATURE: Hole_A1 TYPE: INTERNAL_HOLE  then axis rows "X  nom act dev"
PATTERNS = {
    "pcdmis_head": re.compile(r"^\s*FEATURE\s*[:=]\s*(?P<id>\S+)\s+TYPE\s*[:=]\s*(?P<type>\S+)", re.I),
    "pcdmis_axis": re.compile(r"^\s*(?P<ax>[XYZD])\s+(?P<nom>-?\d+\.?\d*)\s+(?P<act>-?\d+\.?\d*)(?:\s+(?P<dev>-?\d+\.?\d*))?\s*$", re.I),
    "pcdmis_meta": re.compile(r"^\s*(?P<k>PART|REV|DATUM|MMC|TOL|AXIS)\s*[:=]\s*(?P<v>.+?)\s*$", re.I),
}


class ParseError(ValueError):
    pass


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _point(fid, typ, nom, act, size=None, mmc=None, tol=None, axis=None):
    p = {"feature_id": fid, "type": typ, "nominal": [round(v, 4) for v in nom], "actual": [round(v, 4) for v in act],
         "delta": [round(a - n, 4) for a, n in zip(act, nom)]}
    if size is not None: p["measured_size"] = size
    if mmc is not None: p["mmc_size"] = mmc
    if tol is not None: p["base_position_tolerance"] = tol
    if axis: p["axis"] = axis
    return p


def detect_vendor(text: str) -> str:
    if any(PATTERNS["pcdmis_head"].match(l) for l in text.splitlines()[:60]):
        return "pcdmis"
    return "csv"


def parse_csv(text):
    lines = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    if not lines:
        raise ParseError("empty report")
    delim = max([",", ";", "\t"], key=lambda d: lines[0].count(d))
    rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delim))
    head = [h.strip().lower().replace(" ", "_") for h in rows[0]]
    col = {}
    for key, names in ALIASES.items():
        for n in names:
            if n in head:
                col[key] = head.index(n); break
    need = ["feature_id", "nx", "ny", "nz", "ax", "ay", "az"]
    miss = [k for k in need if k not in col]
    if miss:
        raise ParseError(f"CSV header missing columns: {miss}. Found: {head}")
    pts = []
    for ln, r in enumerate(rows[1:], 2):
        g = lambda k: r[col[k]].strip() if k in col and col[k] < len(r) else None
        nom = [_f(g("nx")), _f(g("ny")), _f(g("nz"))]
        act = [_f(g("ax")), _f(g("ay")), _f(g("az"))]
        if None in nom or None in act:
            raise ParseError(f"line {ln}: non-numeric coordinate")
        axis = [_f(g("ai")), _f(g("aj")), _f(g("ak"))]
        pts.append(_point(g("feature_id"), (g("type") or "POINT").upper(), nom, act, _f(g("measured_size")),
                          _f(g("mmc_size")), _f(g("base_position_tolerance")), axis if None not in axis else None))
    return {"part_metadata": {}, "inspection_points": pts}


def parse_pcdmis(text):
    meta, pts, cur = {}, [], None
    def flush():
        if not cur: return
        if not all(k in cur["ax"] for k in "XYZ"):
            raise ParseError(f"feature {cur['id']}: missing X/Y/Z rows")
        nom = [cur["ax"][k][0] for k in "XYZ"]; act = [cur["ax"][k][1] for k in "XYZ"]
        size = cur["ax"].get("D", (None, None))[1]
        mmc = cur["ax"].get("D", (None, None))[0] if size is not None else None
        pts.append(_point(cur["id"], cur["type"].upper(), nom, act, size, cur.get("mmc", mmc), cur.get("tol"), cur.get("axis")))
    for line in text.splitlines():
        m = PATTERNS["pcdmis_head"].match(line)
        if m:
            flush(); cur = {"id": m["id"], "type": m["type"], "ax": {}}; continue
        m = PATTERNS["pcdmis_axis"].match(line)
        if m and cur is not None:
            cur["ax"][m["ax"].upper()] = (float(m["nom"]), float(m["act"])); continue
        m = PATTERNS["pcdmis_meta"].match(line)
        if m:
            k, v = m["k"].lower(), m["v"]
            if cur is None: meta[{"part": "part_id", "rev": "revision", "datum": "datum_frame"}.get(k, k)] = v
            elif k == "tol": cur["tol"] = _f(v)
            elif k == "mmc": cur["mmc"] = _f(v)
            elif k == "axis": cur["axis"] = [float(x) for x in v.split()[:3]]
    flush()
    if not pts:
        raise ParseError("no FEATURE blocks recognised")
    return {"part_metadata": meta, "inspection_points": pts}


def parse_report(text: str, vendor: str = "auto"):
    text = text.lstrip("﻿")
    v = detect_vendor(text) if vendor == "auto" else vendor
    out = parse_pcdmis(text) if v == "pcdmis" else parse_csv(text)
    out["part_metadata"].setdefault("datum_frame", "A-B-C")
    out["part_metadata"]["source_format"] = v
    if len(out["inspection_points"]) < 3:
        raise ParseError("need at least 3 inspection points for alignment")
    return out
