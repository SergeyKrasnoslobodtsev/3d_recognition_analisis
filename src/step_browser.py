import os
from pathlib import Path
from collections import Counter, defaultdict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QListWidget, QListWidgetItem,
    QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QCheckBox, QLabel, QComboBox,
    QPlainTextEdit, QSpinBox, QMessageBox
)

# --------- pythonocc / OCC ----------
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
    GeomAbs_BSplineSurface, GeomAbs_BezierSurface, GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface, GeomAbs_OtherSurface
)
from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
from OCC.Core.BRep import BRep_Tool
from OCC.Extend import TopologyUtils

from OCC.Core.AIS import AIS_Shape, AIS_Point, AIS_DisplayMode, AIS_ColoredShape
from OCC.Core.Quantity import Quantity_NOC_WHITE, Quantity_NOC_BLACK, Quantity_Color, Quantity_TOC_RGB
from OCC.Core.Prs3d import Prs3d_Drawer
# OCC 7.6+/pythonocc: enum для shading живёт в Graphic3d
try:
    from OCC.Core.Graphic3d import Graphic3d_TypeOfShadingModel  # e.g. Graphic3d_TOSM_PHONG
except Exception:
    Graphic3d_TypeOfShadingModel = None
from OCC.Core.gp import gp_Pnt
# ---- Select and load Qt backend for pythonocc before importing qtViewer3d ----
from OCC.Display.backend import load_backend

load_backend("pyside6")

from OCC.Display.qtDisplay import qtViewer3d


# ----------------------- Geometry utils -----------------------

def read_step_shape(path: Path):
    reader = STEPControl_Reader()
    stat = reader.ReadFile(str(path))
    if stat != IFSelect_RetDone:
        return None, f"ReadFile failed (status={stat})"
    reader.TransferRoots()
    shape = reader.OneShape()
    try:
        if shape is None or shape.IsNull():
            return None, "Empty shape"
    except Exception:
        return None, "Empty shape"
    return shape, None


def split_into_solids(shape):
    """Return list of TopoDS_Solid (can be empty)."""
    top = TopologyUtils.TopologyExplorer(shape, ignore_orientation=True)
    solids = list(top.solids())
    return solids


def is_valid_closed(solid):
    ana = BRepCheck_Analyzer(solid)
    valid = bool(ana.IsValid())
    vol = volume(solid)
    closed = valid and (vol is not None and vol > 0.0)
    return valid, closed


def volume(shape):
    try:
        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        return float(props.Mass())
    except Exception:
        return None


def surface_area(shape):
    try:
        props = GProp_GProps()
        brepgprop.SurfaceProperties(shape, props)
        return float(props.Mass())
    except Exception:
        return None


def bbox(shape):
    try:
        box = Bnd_Box()
        brepbndlib.Add(shape, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        return (xmin, ymin, zmin, xmax, ymax, zmax)
    except Exception:
        return None


def count_topology(shape):
    top = TopologyUtils.TopologyExplorer(shape, ignore_orientation=True)
    faces = sum(1 for _ in top.faces())
    edges = sum(1 for _ in top.edges())
    vertices = sum(1 for _ in top.vertices())
    shells = sum(1 for _ in top.shells())
    return faces, edges, vertices, shells


def face_type_name(t):
    return {
        GeomAbs_Plane: "plane",
        GeomAbs_Cylinder: "cylinder",
        GeomAbs_Cone: "cone",
        GeomAbs_Sphere: "sphere",
        GeomAbs_Torus: "torus",
        GeomAbs_BSplineSurface: "bspline",
        GeomAbs_BezierSurface: "bezier",
        GeomAbs_SurfaceOfRevolution: "revolution",
        GeomAbs_SurfaceOfExtrusion: "extrusion",
        GeomAbs_OffsetSurface: "offset",
        GeomAbs_OtherSurface: "other",
    }.get(t, f"type_{int(t)}")


def analyze_face_types(shape):
    """Return (Counter of face types, extra_stats dict like cylinder_radii)."""
    top = TopologyUtils.TopologyExplorer(shape, ignore_orientation=True)
    counts = Counter()
    extras = defaultdict(list)

    for f in top.faces():
        try:
            surf = BRep_Tool.Surface(f)
            ga = GeomAdaptor_Surface(surf)
            t = ga.GetType()
            counts[face_type_name(t)] += 1

            if t == GeomAbs_Cylinder:
                r = ga.Cylinder().Radius()
                extras["cylinder_radius"].append(float(r))
            elif t == GeomAbs_Sphere:
                r = ga.Sphere().Radius()
                extras["sphere_radius"].append(float(r))
            elif t == GeomAbs_Cone:
                r = ga.Cone().RefRadius()
                ang = ga.Cone().SemiAngle()
                extras["cone_ref_radius"].append(float(r))
                extras["cone_semi_angle_deg"].append(float(ang * 180.0 / 3.141592653589793))
            elif t == GeomAbs_Torus:
                rmaj = ga.Torus().MajorRadius()
                rmin = ga.Torus().MinorRadius()
                extras["torus_major_radius"].append(float(rmaj))
                extras["torus_minor_radius"].append(float(rmin))
        except Exception:
            counts["other"] += 1

    return counts, extras


def fmt_num(x, digits=6):
    if x is None:
        return "n/a"
    return f"{x:.{digits}g}"


def summarize_list(vals, name, digits=6):
    if not vals:
        return f"- {name}: n/a\n"
    import numpy as np
    a = np.array(vals, dtype=float)
    return (
        f"- {name} (n={len(vals)}): "
        f"min={a.min():.{digits}g}, "
        f"median={np.median(a):.{digits}g}, "
        f"max={a.max():.{digits}g}\n"
    )


def make_report_for_solid(solid, solid_idx, total_solids, filename):
    faces, edges, vertices, shells = count_topology(solid)
    valid, closed = is_valid_closed(solid)
    vol = volume(solid)
    area = surface_area(solid)
    bb = bbox(solid)
    ft_counts, extras = analyze_face_types(solid)

    ft_lines = []
    if ft_counts:
        total_faces = sum(ft_counts.values())
        for k, v in sorted(ft_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            pct = (100.0 * v / total_faces) if total_faces else 0.0
            ft_lines.append(f"    - {k}: {v} ({pct:.1f}%)")
    else:
        ft_lines.append("    - n/a")

    bb_line = "n/a"
    if bb:
        xmin, ymin, zmin, xmax, ymax, zmax = bb
        bb_line = f"[{fmt_num(xmin)}, {fmt_num(ymin)}, {fmt_num(zmin)}] – [{fmt_num(xmax)}, {fmt_num(ymax)}, {fmt_num(zmax)}]"

    rep = []
    rep.append(f"# Report: {filename}  —  solid {solid_idx+1}/{total_solids}\n")
    rep.append("## Summary\n")
    rep.append(f"- valid: **{valid}**, closed: **{closed}**\n")
    rep.append(f"- faces: {faces}, edges: {edges}, vertices: {vertices}, shells: {shells}\n")
    rep.append(f"- volume: {fmt_num(vol)}, surface area: {fmt_num(area)}\n")
    rep.append(f"- bbox: {bb_line}\n")
    rep.append("\n## Face types\n")
    rep.extend(line + "\n" for line in ft_lines)

    # extras
    rep.append("\n## Geometric stats\n")
    rep.append(summarize_list(extras.get("cylinder_radius", []), "cylinder radius"))
    rep.append(summarize_list(extras.get("sphere_radius", []), "sphere radius"))
    rep.append(summarize_list(extras.get("cone_ref_radius", []), "cone ref radius"))
    rep.append(summarize_list(extras.get("cone_semi_angle_deg", []), "cone semi-angle (deg)"))
    rep.append(summarize_list(extras.get("torus_major_radius", []), "torus major radius"))
    rep.append(summarize_list(extras.get("torus_minor_radius", []), "torus minor radius"))

    rep.append("\n> Note: metrics are computed on exact B-Rep (no meshing). "
               "Closed & valid come from BRepCheck_Analyzer; volumes/areas from GProp.\n")
    return "".join(rep)


def make_report_for_file(filename, solids):
    rep = []
    rep.append(f"# File summary: {filename}\n\n")
    rep.append(f"Total solids: {len(solids)}\n\n")
    # global stats
    vols = [volume(s) or 0.0 for s in solids]
    if vols:
        rep.append(f"Volumes: min={fmt_num(min(vols))}, median={fmt_num(sorted(vols)[len(vols)//2])}, max={fmt_num(max(vols))}\n\n")
    for i, s in enumerate(solids):
        rep.append(make_report_for_solid(s, i, len(solids), filename))
        rep.append("\n\n")
    return "".join(rep)


# ---------- Color palette for face types ----------
# map type_name -> ((r,g,b) floats 0..1, hex string)
PALETTE = {
    "plane":      ((0.75, 0.75, 0.75), "#bfbfbf"),
    "cylinder":   ((0.00, 1.00, 1.00), "#00ffff"),
    "cone":       ((1.00, 0.55, 0.00), "#ff8c00"),
    "sphere":     ((0.00, 1.00, 0.00), "#00ff00"),
    "torus":      ((1.00, 0.00, 1.00), "#ff00ff"),
    "bspline":    ((1.00, 1.00, 0.00), "#ffff00"),
    "bezier":     ((1.00, 0.20, 0.20), "#ff3333"),
    "revolution": ((0.20, 0.40, 1.00), "#3366ff"),
    "extrusion":  ((0.50, 0.50, 1.00), "#8080ff"),
    "offset":     ((0.60, 0.40, 0.25), "#996640"),
    "other":      ((1.00, 1.00, 1.00), "#ffffff"),
}


def color_for_type(tp_name):
    rgb, _ = PALETTE.get(tp_name, PALETTE["other"])
    return Quantity_Color(rgb[0], rgb[1], rgb[2], Quantity_TOC_RGB)


def legend_html(counts):
    if not counts:
        return ""
    total = sum(counts.values()) or 1
    parts = []
    for name, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        _, hexcol = PALETTE.get(name, PALETTE["other"])
        pct = 100.0 * cnt / total
        square = f"<span style='display:inline-block;width:12px;height:12px;background:{hexcol};margin-right:6px;border:1px solid #333'></span>"
        parts.append(f"{square}{name} — {cnt} ({pct:.1f}%)")
    return " &nbsp;|&nbsp; ".join(parts)


# ----------------------- GUI -----------------------

class StepBrowser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STEP Browser + Color & Reports")
        self.resize(1500, 950)

        # Left: open/search/list
        self.btn_open = QPushButton("Открыть папку…")
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText("Поиск по имени файла…")
        self.list_files = QListWidget()

        left = QVBoxLayout()
        left.addWidget(self.btn_open)
        left.addWidget(self.ed_search)
        left.addWidget(self.list_files)
        leftw = QWidget(); leftw.setLayout(left)

        # Right: controls + report + viewer
        # Top controls
        self.cmb_solids = QComboBox()
        self.cmb_solids.setMinimumWidth(320)
        self.cb_hide_small = QCheckBox("Скрыть мелкие тела < X% от max объёма")
        self.spn_hide_pct = QSpinBox(); self.spn_hide_pct.setRange(0, 100); self.spn_hide_pct.setValue(1)
        self.cb_color_types = QCheckBox("Раскрасить по типам граней")
        self.btn_export = QPushButton("Экспорт отчёта…")
        self.btn_export_all = QPushButton("Экспорт сводного отчёта…")

        top_controls = QHBoxLayout()
        top_controls.addWidget(QLabel("Тело:"))
        top_controls.addWidget(self.cmb_solids)
        top_controls.addSpacing(12)
        top_controls.addWidget(self.cb_hide_small)
        top_controls.addWidget(self.spn_hide_pct)
        top_controls.addSpacing(12)
        top_controls.addWidget(self.cb_color_types)
        top_controls.addStretch(1)
        top_controls.addWidget(self.btn_export)
        top_controls.addWidget(self.btn_export_all)

        # Legend + Report
        self.lbl_legend = QLabel()
        self.lbl_legend.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_legend.setWordWrap(True)
        self.report = QPlainTextEdit(); self.report.setReadOnly(True)
        self.report.setPlaceholderText("Здесь мини-отчёт по выбранному телу…")

        # Viewer + toggles
        self.viewer = qtViewer3d(self)
        self.display = self.viewer._display
        # set background color in a version-tolerant way
        try:
            self.display.set_bg_color(Quantity_NOC_BLACK)
        except Exception:
            try:
                self.display.View.SetBackgroundColor(Quantity_NOC_BLACK)
            except Exception:
                pass

                self.cb_faces = QCheckBox("Грани (shaded)"); self.cb_faces.setChecked(True)
                self.cb_edges = QCheckBox("Рёбра (boundaries)"); self.cb_edges.setChecked(True)
                self.cb_vertices = QCheckBox("Вершины (points)"); self.cb_vertices.setChecked(False)

                toggles = QHBoxLayout()
                toggles.addWidget(self.cb_faces)
                toggles.addWidget(self.cb_edges)
                toggles.addWidget(self.cb_vertices)
                toggles.addStretch(1)

                # Layout right
                right = QVBoxLayout()
                right.addLayout(top_controls)
                right.addWidget(self.lbl_legend)
                right.addWidget(self.report, 2)
                right.addLayout(toggles)
                right.addWidget(self.viewer, 5)
                rightw = QWidget(); rightw.setLayout(right)

                root = QHBoxLayout()
                root.addWidget(leftw, 0)
                root.addWidget(rightw, 1)
                cw = QWidget(); cw.setLayout(root)
                self.setCentralWidget(cw)

                # State
                self.current_dir = None
                self.file_to_solids = {}  # filename -> list of solids
                self.file_volumes = {}
                self.current_file = None
                self.current_solid_idx = -1
                self.ais_shape = None
                self.vertex_points = []

                # Signals
                self.btn_open.clicked.connect(self.choose_dir)
                self.ed_search.textChanged.connect(self.apply_filter)
                self.list_files.itemSelectionChanged.connect(self.on_file_selected)
                self.cmb_solids.currentIndexChanged.connect(self.on_solid_changed)
                self.cb_hide_small.toggled.connect(self.rebuild_solid_combo)
                self.spn_hide_pct.valueChanged.connect(self.rebuild_solid_combo)
                self.cb_color_types.toggled.connect(self.rebuild_ais)
                self.btn_export.clicked.connect(self.export_report)
                self.btn_export_all.clicked.connect(self.export_all_reports)

                self.cb_faces.toggled.connect(self.refresh_view)
                self.cb_edges.toggled.connect(self.refresh_view)
                self.cb_vertices.toggled.connect(self.refresh_view)

    # ------------- File list -------------

    def choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Выберите папку с STEP моделями")
        if not d:
            return
        self.current_dir = Path(d)
        self.populate_files()

    def populate_files(self):
        self.list_files.clear()
        self.file_to_solids.clear()
        self.file_volumes.clear()
        if not self.current_dir:
            return
        for p in sorted(self.current_dir.glob("*")):
            if p.suffix.lower() in (".stp", ".step"):
                self.list_files.addItem(QListWidgetItem(p.name))
        self.apply_filter()

    def apply_filter(self):
        text = self.ed_search.text().lower().strip()
        for i in range(self.list_files.count()):
            it = self.list_files.item(i)
            it.setHidden(text not in it.text().lower())

    # ------------- Selection -------------

    def on_file_selected(self):
        items = self.list_files.selectedItems()
        if not items or not self.current_dir:
            return
        filename = items[0].text()
        self.current_file = self.current_dir / filename
        self.load_file_solids(self.current_file)
        self.rebuild_solid_combo()

    def load_file_solids(self, path: Path):
        fname = path.name
        if fname in self.file_to_solids:
            return
        shape, err = read_step_shape(path)
        if err:
            QMessageBox.warning(self, "Ошибка чтения", f"{fname}:\n{err}")
            self.file_to_solids[fname] = []
            self.file_volumes[fname] = []
            return
        solids = split_into_solids(shape)
        if not solids:
            solids = [shape]
        vols = []
        for s in solids:
            v = volume(s)
            vols.append(v if v is not None else 0.0)
        self.file_to_solids[fname] = solids
        self.file_volumes[fname] = vols

    def rebuild_solid_combo(self):
        self.cmb_solids.blockSignals(True)
        self.cmb_solids.clear()
        if not self.current_file:
            self.cmb_solids.blockSignals(False)
            return
        fname = self.current_file.name
        solids = self.file_to_solids.get(fname, [])
        vols = self.file_volumes.get(fname, [])
        if not solids:
            self.cmb_solids.addItem("(нет тел)")
            self.cmb_solids.blockSignals(False)
            return

        # filter small bodies
        idx_map = list(range(len(solids)))
        if self.cb_hide_small.isChecked() and any(v is not None for v in vols):
            vmax = max(vols) if vols else 0.0
            thr = vmax * (self.spn_hide_pct.value() / 100.0)
            idx_map = [i for i, v in enumerate(vols) if (v or 0.0) >= thr]
            if not idx_map:
                idx_map = [int(vols.index(vmax))]
        self._idx_map = idx_map

        for i in idx_map:
            faces, edges, vertices, shells = count_topology(solids[i])
            v = vols[i] if vols else None
            self.cmb_solids.addItem(f"solid {i+1}  | faces={faces}  vol={fmt_num(v)}")

        self.cmb_solids.setCurrentIndex(0)
        self.cmb_solids.blockSignals(False)
        self.on_solid_changed()

    def on_solid_changed(self):
        if not self.current_file:
            return
        fname = self.current_file.name
        solids = self.file_to_solids.get(fname, [])
        vols = self.file_volumes.get(fname, [])
        if not solids:
            self.report.setPlainText("Нет тел для отображения.")
            self.display.EraseAll()
            self.vertex_points.clear()
            self.lbl_legend.setText("")
            return

        view_idx = self.cmb_solids.currentIndex()
        if not hasattr(self, "_idx_map"):
            self._idx_map = list(range(len(solids)))
        if view_idx < 0 or view_idx >= len(self._idx_map):
            view_idx = 0
        i = self._idx_map[view_idx]
        self.current_solid_idx = i

        self.show_solid(solids[i])
        rep = make_report_for_solid(solids[i], i, len(solids), fname)
        self.report.setPlainText(rep)

        # Legend (face type counts)
        counts, _ = analyze_face_types(solids[i])
        self.lbl_legend.setText(legend_html(counts))

    # ------------- Viewer -------------

    def clear_vertices(self):
        if self.vertex_points:
            for p in self.vertex_points:
                try:
                    self.display.Context.Remove(p, True)
                except Exception:
                    pass
        self.vertex_points = []

    def make_ais(self, solid):
        """Create AIS object; face-type coloring optional. Do NOT touch drawer here for compatibility."""
        if self.cb_color_types.isChecked():
            ais = AIS_ColoredShape(solid)
            top = TopologyUtils.TopologyExplorer(solid, ignore_orientation=True)
            for f in top.faces():
                try:
                    surf = BRep_Tool.Surface(f)
                    ga = GeomAdaptor_Surface(surf)
                    tname = face_type_name(ga.GetType())
                except Exception:
                    tname = "other"
                col = color_for_type(tname)
                try:
                    ais.SetCustomColor(f, col)
                except Exception:
                    pass
        else:
            ais = AIS_Shape(solid)
        return ais

    def show_solid(self, solid):
        self.display.EraseAll()
        self.clear_vertices()

        self.ais_shape = self.make_ais(solid)
        self.display.Context.Display(self.ais_shape, True)
        self.display.FitAll()
        self.refresh_view()

    def rebuild_ais(self):
        # rebuild AIS for current solid when color mode toggles
        if not self.current_file or self.current_solid_idx is None:
            return
        fname = self.current_file.name
        solids = self.file_to_solids.get(fname, [])
        if not solids:
            return
        i = self.current_solid_idx
        if i < 0 or i >= len(solids):
            return
        self.show_solid(solids[i])

    def refresh_view(self):
        if not self.ais_shape:
            return
        # Set display mode via context (portable)
        mode = AIS_DisplayMode.AIS_Shaded if self.cb_faces.isChecked() else AIS_DisplayMode.AIS_WireFrame
        try:
            self.display.Context.SetDisplayMode(self.ais_shape, mode, True)
        except Exception:
            try:
                self.ais_shape.SetDisplayMode(mode)
            except Exception:
                pass
        # Face boundary draw if drawer available
        try:
            drawer = self.ais_shape.Attributes()
            if drawer is not None:
                drawer.SetFaceBoundaryDraw(self.cb_edges.isChecked())
        except Exception:
            pass
        # Set overall color (only when not per-face coloring)
        if not self.cb_color_types.isChecked():
            try:
                self.display.Context.SetColor(self.ais_shape, Quantity_NOC_WHITE, False)
            except Exception:
                pass
        # Vertices
        self.clear_vertices()
        if self.cb_vertices.isChecked():
            top = TopologyUtils.TopologyExplorer(self.ais_shape.Shape(), ignore_orientation=True)
            cnt = 0
            for v in top.vertices():
                try:
                    p = BRep_Tool.Pnt(v)
                    ap = AIS_Point(p)
                    ap.SetColor(Quantity_NOC_WHITE)
                    self.display.Context.Display(ap, False)
                    self.vertex_points.append(ap)
                except Exception:
                    pass
                cnt += 1
                if cnt > 20000:
                    break
        try:
            self.display.Context.UpdateCurrentViewer()
        except Exception:
            pass
        try:
            self.display.View.SetBackgroundColor(Quantity_NOC_BLACK)
        except Exception:
            pass

    # ------------- Export -------------

    def export_report(self):
        if not self.current_file:
            QMessageBox.information(self, "Экспорт", "Нет выбранного файла.")
            return
        fname = self.current_file.name
        solids = self.file_to_solids.get(fname, [])
        if not solids:
            QMessageBox.information(self, "Экспорт", "Нет тел для отчёта.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчёт", f"{fname}.md", "Markdown (*.md);;Text (*.txt)")
        if not path:
            return
        idx = self.current_solid_idx if self.current_solid_idx is not None else 0
        idx = max(0, min(idx, len(solids) - 1))
        rep = make_report_for_solid(solids[idx], idx, len(solids), fname)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(rep)
        except Exception as e:
            QMessageBox.warning(self, "Ошибка сохранения", str(e))
        else:
            QMessageBox.information(self, "Готово", f"Отчёт сохранён:\n{path}")

    def export_all_reports(self):
        if not self.current_file:
            QMessageBox.information(self, "Экспорт", "Нет выбранного файла.")
            return
        fname = self.current_file.name
        solids = self.file_to_solids.get(fname, [])
        if not solids:
            QMessageBox.information(self, "Экспорт", "Нет тел для отчёта.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить сводный отчёт", f"{fname}_summary.md", "Markdown (*.md);;Text (*.txt)")
        if not path:
            return
        rep = make_report_for_file(fname, solids)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(rep)
        except Exception as e:
            QMessageBox.warning(self, "Ошибка сохранения", str(e))
        else:
            QMessageBox.information(self, "Готово", f"Сводный отчёт сохранён:\n{path}")


def main():
    app = QApplication([])
    w = StepBrowser()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
