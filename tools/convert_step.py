"""Convert NIST STEP parts to a single-primitive GLB + hole-feature JSON.

usage: convert_step.py <out_dir> <step files...>
"""
import json, sys, re
from pathlib import Path
import numpy as np
import trimesh
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder


def convert(step_path: Path, out_dir: Path):
    rd = STEPControl_Reader()
    if rd.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError("read failed")
    rd.TransferRoots()
    shape = rd.OneShape()
    box = Bnd_Box(); BRepBndLib.Add_s(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
    BRepMesh_IncrementalMesh(shape, diag * 0.002, False, 0.35, True)

    verts, faces = [], []
    holes = {}
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            tr = loc.Transformation()
            base = len(verts)
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(tr)
                verts.append((p.X(), p.Y(), p.Z()))
            rev = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                t = (base + a - 1, base + b - 1, base + c - 1)
                faces.append((t[0], t[2], t[1]) if rev else t)
        s = BRepAdaptor_Surface(face)
        if s.GetType() == GeomAbs_Cylinder:
            cyl = s.Cylinder()
            ax = cyl.Axis(); o = ax.Location(); d = ax.Direction()
            r = cyl.Radius()
            internal = face.Orientation() == TopAbs_REVERSED
            u0, u1, v0, v1 = s.FirstUParameter(), s.LastUParameter(), s.FirstVParameter(), s.LastVParameter()
            if internal and 1.0 < r < diag * 0.2 and (v1 - v0) > 0.3:
                dd = np.array([d.X(), d.Y(), d.Z()])
                oo = np.array([o.X(), o.Y(), o.Z()])
                oo = oo - dd * np.dot(oo, dd)  # canonical point on axis
                key = tuple(np.round(oo, 2)) + tuple(np.round(np.abs(dd), 3)) + (round(r, 2),)
                g = holes.setdefault(key, {"center": oo, "axis": dd, "diameter": 2 * r,
                                           "span": 0.0, "v0": v0, "v1": v1, "o": np.array([o.X(), o.Y(), o.Z()])})
                g["span"] += u1 - u0
                g["v0"], g["v1"] = min(g["v0"], v0), max(g["v1"], v1)
        exp.Next()

    m = trimesh.Trimesh(np.array(verts), np.array(faces), process=False)
    # centre the part; shift feature coordinates identically
    c = m.bounds.mean(axis=0)
    m.vertices -= c
    feats = []
    full = [h for h in holes.values() if h["span"] > 5.5]
    for i, h in enumerate(sorted(full, key=lambda h: (-h["diameter"], tuple(h["o"])))[:24]):
        ctr = h["o"] + h["axis"] * (h["v0"] + h["v1"]) / 2 - c
        feats.append({"feature_id": f"Hole_{i+1:02d}", "center": ctr.round(4).tolist(),
                      "axis": h["axis"].round(5).tolist(), "diameter": round(h["diameter"], 4),
                      "depth": round(float(h["v1"] - h["v0"]), 3)})
    name = re.sub(r"_(asme1|ap\d+).*", "", step_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    m.export(out_dir / f"{name}.glb")
    meta = {"id": name, "vertices": len(m.vertices), "triangles": len(m.faces),
            "bounds": m.bounds.round(3).tolist(), "source": step_path.name, "features": feats}
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print(name, len(m.vertices), "verts", len(feats), "holes", round((out_dir / f"{name}.glb").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for f in sys.argv[2:]:
        try:
            convert(Path(f), out)
        except Exception as e:
            print("FAIL", f, e)
