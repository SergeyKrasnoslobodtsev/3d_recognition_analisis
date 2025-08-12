import numpy as np
from loguru import logger
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph as csgraph
import warnings

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
    GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion, GeomAbs_OtherSurface
)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

from ..dataset import DataModel
from .extractor import FeatureExtractor, FeatureVector


class BrepExtractor(FeatureExtractor):
    
    def __init__(self, rdf_k: int = 256, lbo_k: int = 16, bins: int = 64):
        self.name = "BRep"
        self.rdf_k = rdf_k
        self.lbo_k = lbo_k
        self.bins = bins
        
        # Определяем размерность вектора на основе параметров
        # RDF + h_edge + h_area + h_dih + h_deg + brep_vec + lbo_evals
        self.feature_dim = self.rdf_k + self.bins + self.bins + self.bins + (self.bins // 2) + 10 + self.lbo_k
        
        logger.info(f"Инициализация '{self.name}' экстрактора. Размерность вектора: {self.feature_dim}")

    def extract_single(self, data: DataModel) -> FeatureVector | None:
        """
        Извлекает единый вектор признаков для 3D-модели.
        """
        shape = self._load_step_shape(data.model_path)
        if shape is None or shape.IsNull():
            logger.warning(f"Не удалось загрузить или пустая геометрия: {data.model_path}")
            return None

        try:
            # 1. Триангуляция всей модели
            self._mesh_shape(shape, lin_deflection=0.02, ang_deflection=0.785)
            V, F = self._get_vertices_and_faces(shape)
            logger.debug(f"Триангуляция завершена для {data.model_id}. Вершин: {V.shape[0]}, Граней: {F.shape[0]}")
            if V.shape[0] < 4 or F.shape[0] < 1:
                logger.warning(f"Недостаточно вершин/граней после триангуляции для {data.model_id}")
                return self._create_empty_vector(data)

            # 2. Извлечение признаков на основе триангуляции
            h_edge, h_area, h_dih, h_deg = self._compute_mesh_histograms(V, F)
            rdf = self._compute_rdf(V, F, K=self.rdf_k)

            # 3. Извлечение признаков из исходной B-Rep структуры
            brep_vec = self._brep_surface_type_hist(shape)

            # 4. Извлечение спектральных признаков
            padded_evals = self._compute_lbo_spectrum_features(V, F, k=self.lbo_k)

            # 5. Конкатенация всех признаков в строгом порядке
            feature_vec = np.concatenate([
                rdf, 
                h_edge,
                h_area,
                h_dih,
                h_deg,
                brep_vec,
                padded_evals
            ]).astype(np.float32)

            if feature_vec.shape[0] != self.feature_dim:
                logger.error(f"Неверная размерность вектора для {data.model_id}."
                             f"Ожидалось {self.feature_dim}, получено {feature_vec.shape[0]}")
                return self._create_empty_vector(data)

            return FeatureVector(model_id=data.model_id, vector=feature_vec, label=data.detail_type)

        except Exception as e:
            logger.error(f"Ошибка при извлечении признаков для {data.model_id}: {e}")
            return self._create_empty_vector(data)

    def _create_empty_vector(self, data: DataModel) -> FeatureVector:
        """Создает пустой (нулевой) вектор признаков."""
        return FeatureVector(
            model_id=data.model_id,
            vector=np.zeros(self.feature_dim, dtype=np.float32),
            label=data.detail_type
        )

    # --- Методы извлечения признаков ---

    def _compute_mesh_histograms(self, V: np.ndarray, F: np.ndarray):
        """Вычисляет набор гистограмм на основе сетки."""
        E, L = self.mesh_edge_data(V, F)
        A = self.mesh_face_areas(V, F)
        D = self._mesh_dihedral_angles(V, F)
        deg = self.mesh_vertex_degrees(F, V.shape[0])

        h_edge = self.hist_norm(L, bins=self.bins, log=True)
        h_area = self.hist_norm(A, bins=self.bins, log=True)
        h_dih = self.hist_norm(D, bins=self.bins, rng=(0, np.pi))
        h_deg = self.hist_norm(deg, bins=self.bins // 2, rng=(0, deg.max() if deg.size else 1))
        
        return h_edge, h_area, h_dih, h_deg

    def _compute_lbo_spectrum_features(self, V: np.ndarray, F: np.ndarray, k: int) -> np.ndarray:
        """Вычисляет и дополняет нулями собственные значения оператора Лапласа-Бельтрами."""
        padded_evals = np.zeros(k, dtype=np.float32)
        try:
            evals, _ = self._compute_lbo_spectrum(V, F, k=k)
            num_evals = min(len(evals), k)
            padded_evals[:num_evals] = evals[:num_evals]
        except (RuntimeError, np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Не удалось вычислить спектр: {e}. Вектор будет заполнен нулями.")
        return padded_evals

    @staticmethod
    def _load_step_shape(step_path: str) -> TopoDS_Shape | None:
        """Загружает STEP-файл и возвращает TopoDS_Shape."""
        reader = STEPControl_Reader()
        if reader.ReadFile(step_path) != IFSelect_RetDone:
            logger.error(f"Не удалось прочитать STEP: {step_path}")
            return None
        reader.TransferRoots()
        return reader.OneShape()

    @staticmethod
    def _mesh_shape(shape, lin_deflection=0.05, ang_deflection=0.5, is_relative=True, parallel=True):
        """Создает сетку для заданной формы."""
        BRepMesh_IncrementalMesh(shape, lin_deflection, is_relative, ang_deflection, parallel)

    @staticmethod
    def _get_vertices_and_faces(shape: TopoDS_Shape) -> tuple[np.ndarray, np.ndarray]:
        """Извлекает вершины и грани из заданной формы."""
        verts_chunks, faces_chunks = [], []
        v_off = 0
        topo = TopologyExplorer(shape)
        for face in topo.faces():
            loc = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, loc)
            if triangulation is None:
                continue

            trsf = loc.Transformation()
            nb_nodes = triangulation.NbNodes()
            cur_verts = np.empty((nb_nodes, 3), dtype=np.float64)
            for i in range(1, nb_nodes + 1):
                p = triangulation.Node(i).Transformed(trsf)
                cur_verts[i - 1] = [p.X(), p.Y(), p.Z()]
            verts_chunks.append(cur_verts)

            nb_tris = triangulation.NbTriangles()
            cur_faces = np.empty((nb_tris, 3), dtype=np.int64)
            for i in range(1, nb_tris + 1):
                t = triangulation.Triangle(i)
                i1, i2, i3 = t.Get()
                cur_faces[i - 1] = [v_off + i1 - 1, v_off + i2 - 1, v_off + i3 - 1]
            faces_chunks.append(cur_faces)
            v_off += nb_nodes

        if not verts_chunks:
            return np.empty((0, 3)), np.empty((0, 3))
        
        V = np.vstack(verts_chunks)
        F = np.vstack(faces_chunks)
        return V, F

    @staticmethod
    def mesh_edge_data(V: np.ndarray, F: np.ndarray):
        """Извлекает данные о ребрах из заданной сетки."""
        E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
        E.sort(axis=1)
        E = np.unique(E, axis=0)
        L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        return E, L

    @staticmethod
    def mesh_face_areas(V: np.ndarray, F: np.ndarray):
        """Извлекает площади граней из заданной сетки."""
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    @staticmethod
    def mesh_face_normals(V: np.ndarray, F: np.ndarray):
        """Извлекает нормали граней из заданной сетки."""
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        return n / n_norm

    def _mesh_dihedral_angles(self, V: np.ndarray, F: np.ndarray):
        """Вычисляет диэдральные углы между гранями."""
        from collections import defaultdict
        E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
        E_sorted = np.sort(E, axis=1)
        tri_idx = np.repeat(np.arange(F.shape[0]), 3)
        edge2tris = defaultdict(list)
        for e, t in zip(map(tuple, E_sorted), tri_idx):
            edge2tris[e].append(t)

        N = self.mesh_face_normals(V, F)
        ang = []
        for tris in edge2tris.values():
            if len(tris) == 2:
                i, j = tris
                c = np.clip(np.dot(N[i], N[j]), -1.0, 1.0)
                ang.append(np.arccos(c))
        return np.array(ang, dtype=np.float64)

    @staticmethod
    def mesh_vertex_degrees(F: np.ndarray, Vn: int):
        """Вычисляет степени вершин в заданной сетке."""
        deg = np.zeros(Vn, dtype=np.int32)
        np.add.at(deg, F.ravel(), 1)
        return deg

    @staticmethod
    def hist_norm(x: np.ndarray, bins: int, rng: tuple = None, log: bool = False): # type: ignore
        """Вычисляет нормализованную гистограмму для заданного массива."""
        if x.size == 0:
            return np.zeros(bins, dtype=np.float32)
        vals = np.log10(np.clip(x, 1e-12, None)) if log else x
        H, _ = np.histogram(vals, bins=bins, range=rng, density=True)
        return H.astype(np.float32)

    @staticmethod
    def fibonacci_sphere(n: int):
        """Генерирует точки на сфере с использованием спиральной схемы Фибоначчи."""
        i = np.arange(n, dtype=np.float64)
        phi = (1 + 5 ** 0.5) / 2
        theta = 2 * np.pi * i / phi
        z = 1 - (2 * i + 1) / n
        r = np.sqrt(np.maximum(0.0, 1 - z * z))
        x, y = r * np.cos(theta), r * np.sin(theta)
        return np.stack([x, y, z], axis=1)

    @staticmethod
    def ray_triangle_intersections(orig: np.ndarray, dirv: np.ndarray, V: np.ndarray, F: np.ndarray):
        """Вычисляет пересечения луча с треугольниками в заданной сетке."""
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        eps = 1e-9
        e1, e2 = v1 - v0, v2 - v0
        pvec = np.cross(dirv, e2)
        det = (e1 * pvec).sum(axis=1)
        mask = np.abs(det) > eps
        inv_det = np.zeros_like(det)
        inv_det[mask] = 1.0 / det[mask]
        tvec = orig - v0
        u = (tvec * pvec).sum(axis=1) * inv_det
        qvec = np.cross(tvec, e1)
        v = (dirv * qvec).sum(axis=1) * inv_det
        t = (e2 * qvec).sum(axis=1) * inv_det
        cond = (mask) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > eps)
        t_valid = np.where(cond, t, np.inf)
        return t_valid.min()

    def _compute_rdf(self, V: np.ndarray, F: np.ndarray, K: int):
        """Вычисляет радиальную распределенную функцию (RDF) для заданной сетки."""
        c = V.mean(axis=0)
        rmax = np.linalg.norm(V - c, axis=1).max() + 1e-9
        dirs = self.fibonacci_sphere(K)
        dists = np.array([self.ray_triangle_intersections(c, d, V, F) for d in dirs])
        dists[~np.isfinite(dists)] = rmax
        return (dists / rmax).astype(np.float32)

    @staticmethod
    def face_area(face):
        """Вычисляет площадь грани."""
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        return props.Mass()

    def _brep_surface_type_hist(self, shape: TopoDS_Shape):
        """Извлекает данные о типах поверхностей из заданной сетки."""
        keys = [
            GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
            GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
            GeomAbs_SurfaceOfExtrusion, GeomAbs_OtherSurface
        ]
        type2area = {k: 0.0 for k in keys}
        total_area = 0.0
        topo = TopologyExplorer(shape)
        for face in topo.faces():
            A = self.face_area(face)
            if A > 0:
                surf = BRepAdaptor_Surface(face, True)
                st = surf.GetType()
                total_area += A
                type2area[st] = type2area.get(st, 0.0) + A
        
        if total_area <= 1e-9:
            return np.zeros(len(keys), dtype=np.float32)
        
        return np.array([type2area[k] / total_area for k in keys], dtype=np.float32)

    @staticmethod
    def _compute_lbo_spectrum(V, F, k=32, scale_invariant=True):
        """
        Вычисляет спектр локальных биортогональных функций (LBO) для заданной сетки.
        """
        # Вспомогательные функции для LBO
        def total_area(V, F):
            """Вычисляет общую площадь треугольников в заданной сетке."""
            v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
            return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()

        def clean_mesh(V, F, area_eps=1e-14):
            """
            Очищает сетку, удаляя треугольники с малой площадью.
            """
            if F.size == 0: return V, F
            area2 = np.linalg.norm(np.cross(V[F[:,1]] - V[F[:,0]], V[F[:,2]] - V[F[:,0]]), axis=1)
            F = F[area2 > area_eps]
            if F.size == 0: return V[:0], F
            used = np.unique(F.ravel())
            remap = -np.ones(V.shape[0], dtype=np.int64); remap[used] = np.arange(used.size)
            return V[used], remap[F]

        def keep_largest_component(V, F):
            """
            Сохраняет только крупнейший компонент связности в сетке.
            """
            if F.size == 0: return V, F
            n = V.shape[0]
            adj = sp.csr_matrix((np.ones(F.shape[0]*2), (F[:,[0,1]].ravel(), F[:,[1,0]].ravel())), shape=(n,n))
            ncomp, labels = csgraph.connected_components(adj, directed=False)
            if ncomp <= 1: return V, F
            largest = np.argmax(np.bincount(labels))
            mask_v = labels == largest
            idx_old = np.where(mask_v)[0]
            remap = -np.ones(n, dtype=np.int64); remap[idx_old] = np.arange(idx_old.size)
            Fm = remap[F]; Fm = Fm[(Fm >= 0).all(axis=1)]
            return V[idx_old], Fm

        def build_laplacian_cotan(V, F):
            """
            Строит лапласиан с использованием котангенсной схемы.
            """
            n = V.shape[0]
            i, j, k = F[:,0], F[:,1], F[:,2]
            vi, vj, vk = V[i], V[j], V[k]
            area2 = np.linalg.norm(np.cross(vj - vi, vk - vi), axis=1)
            area2_safe = np.maximum(area2, 1e-15)
            cot_i = ((vj - vi) * (vk - vi)).sum(axis=1) / area2_safe
            cot_j = ((vi - vj) * (vk - vj)).sum(axis=1) / area2_safe
            cot_k = ((vi - vk) * (vj - vk)).sum(axis=1) / area2_safe
            w_ij, w_jk, w_ki = 0.5 * cot_k, 0.5 * cot_i, 0.5 * cot_j
            rows = np.concatenate([i, j, j, k, k, i])
            cols = np.concatenate([j, i, k, j, i, k])
            data = np.concatenate([-w_ij, -w_ij, -w_jk, -w_jk, -w_ki, -w_ki])
            L = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
            L = L - sp.diags(L.sum(axis=1).A1) # type: ignore
            M_diag = np.zeros(n); tri_area = 0.5 * area2
            np.add.at(M_diag, i, tri_area / 3.0); np.add.at(M_diag, j, tri_area / 3.0); np.add.at(M_diag, k, tri_area / 3.0)
            return L, np.maximum(M_diag, 1e-15)

        # Основная логика
        Vc, Fc = clean_mesh(*keep_largest_component(*clean_mesh(V, F)))
        if Vc.shape[0] < 3 or Fc.shape[0] < 1:
            raise RuntimeError("Недостаточно данных для спектра после чистки.")
        
        Vn = Vc
        if scale_invariant:
            A = total_area(Vc, Fc)
            if A > 0: Vn = Vc / np.sqrt(A)

        L, M_diag = build_laplacian_cotan(Vn, Fc)
        n = Vn.shape[0]
        k_solve = min(k, max(1, n - 2))
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The problem size')
            evals, evecs = spla.eigsh(L, k=k_solve, M=sp.diags(M_diag), sigma=1e-8, which='LM', tol=0)
        
        order = np.argsort(evals)
        evals = evals[order]
        pos = evals > 1e-10
        return evals[pos], evecs[:, order][:, pos].astype(np.float64)