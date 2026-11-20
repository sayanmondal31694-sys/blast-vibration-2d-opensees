"""2D plane-strain mesh, materials and boundary conditions (OpenSeesPy).

Public API
----------
build_model(cfg) -> Mesh
    Wipes the OpenSees domain, builds nodes/quads/materials, applies BCs
    (free surface on top; Lysmer-Kuhlemeyer viscous dashpots on left, right and
    bottom edges via zeroLength elements to fixed ground nodes). Saves
    <out_dir>/mesh.npz. The live OpenSees domain is left in place for the solver.
select_source_nodes(mesh, cfg) -> dict(center_tag, tags[4], dirs[4,2])
Mesh.nearest_node(x, y) -> int tag ; Mesh.save(path) ; Mesh.load(path)

Node tag = 1 + i + j*(nx+1), j = 0 is the ground surface row.
Element tag = 1 + ie + je*nx. Dashpot ground nodes tags start at GROUND_BASE.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import openseespy.opensees as ops

from config import Config, wave_speeds

GROUND_BASE = 1_000_000
MAT_BASE = 10          # nDMaterial tags: MAT_BASE + layer index
DASH_MAT_BASE = 1000   # uniaxial Viscous tags
DASH_ELE_BASE = 2_000_000


@dataclass
class Mesh:
    nx: int
    ny: int
    h: float
    node_tags: np.ndarray      # (N,)
    node_xy: np.ndarray        # (N,2) metres
    ele_tags: np.ndarray       # (E,)
    ele_nodes: np.ndarray      # (E,4) CCW
    ele_layer: np.ndarray      # (E,) layer index
    surface_tags: np.ndarray   # (nx+1,) ordered by x
    left_tags: np.ndarray
    right_tags: np.ndarray
    bottom_tags: np.ndarray
    n_dashpots: int

    def nearest_node(self, x: float, y: float) -> int:
        d2 = (self.node_xy[:, 0] - x) ** 2 + (self.node_xy[:, 1] - y) ** 2
        return int(self.node_tags[int(np.argmin(d2))])

    def xy_of(self, tag: int):
        return self.node_xy[int(tag) - 1]       # tags are 1..N contiguous

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @staticmethod
    def load(path) -> "Mesh":
        z = np.load(path)
        kw = {k: z[k] for k in z.files}
        for k in ("nx", "ny", "n_dashpots"):
            kw[k] = int(kw[k])
        kw["h"] = float(kw["h"])
        return Mesh(**kw)


def check_resolution(cfg: Config):
    """Return list of (layer, f_resolved, f_needed, ok)."""
    f_need = cfg.f_max_factor * cfg.pulse_fc
    out = []
    for L in cfg.layers:
        _, vs = wave_speeds(L)
        f_res = vs / (cfg.pts_per_wavelength * cfg.h)
        out.append((L.name, f_res, f_need, f_res >= f_need))
    return out


def build_model(cfg: Config) -> Mesh:
    nx, ny, h = cfg.nx, cfg.ny, cfg.h
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    # materials (plane strain, density carried by the nDMaterial)
    for k, L in enumerate(cfg.layers):
        ops.nDMaterial("ElasticIsotropic", MAT_BASE + k, L.E, L.nu, L.rho)

    # nodes
    N = (nx + 1) * (ny + 1)
    tags = np.arange(1, N + 1)
    xy = np.zeros((N, 2))
    for j in range(ny + 1):
        for i in range(nx + 1):
            t = 1 + i + j * (nx + 1)
            xy[t - 1] = (i * h, -j * h)
            ops.node(t, float(i * h), float(-j * h))

    # elements (quad, CCW: bottom-left, bottom-right, top-right, top-left)
    E = nx * ny
    etags = np.arange(1, E + 1)
    enodes = np.zeros((E, 4), dtype=int)
    elayer = np.zeros(E, dtype=int)
    for je in range(ny):
        depth_c = (je + 0.5) * h
        lay = cfg.layer_at_depth(depth_c)
        for ie in range(nx):
            et = 1 + ie + je * nx
            n = (1 + ie + (je + 1) * (nx + 1), 2 + ie + (je + 1) * (nx + 1),
                 2 + ie + je * (nx + 1), 1 + ie + je * (nx + 1))
            enodes[et - 1] = n
            elayer[et - 1] = lay
            ops.element("quad", et, *n, cfg.thickness, "PlaneStrain", MAT_BASE + lay)

    # boundary node sets
    surface = np.array([1 + i for i in range(nx + 1)])
    left = np.array([1 + j * (nx + 1) for j in range(ny + 1)])
    right = np.array([1 + nx + j * (nx + 1) for j in range(ny + 1)])
    bottom = np.array([1 + i + ny * (nx + 1) for i in range(nx + 1)])

    # ---- boundary conditions ------------------------------------------------
    # Top: free surface (no constraint). Left/right/bottom: Lysmer-Kuhlemeyer
    # dashpots, c_n = rho*Vp*A, c_t = rho*Vs*A, A = tributary length * thickness,
    # attached with zeroLength to fixed ground nodes at the same coordinates.
    ndash = 0

    def add_dashpot(model_tag, ie, je, normal_dir, trib):
        nonlocal ndash
        L = cfg.layers[elayer[(ie + je * nx)]]
        vp, vs = wave_speeds(L)
        A = trib * cfg.thickness
        tang_dir = 2 if normal_dir == 1 else 1
        g = GROUND_BASE + model_tag + (0 if normal_dir == 1 else 500_000)
        x, y = xy[model_tag - 1]
        ops.node(g, float(x), float(y))
        ops.fix(g, 1, 1)
        mN = DASH_MAT_BASE + 2 * ndash
        mT = mN + 1
        ops.uniaxialMaterial("Viscous", mN, float(L.rho * vp * A), 1.0)
        ops.uniaxialMaterial("Viscous", mT, float(L.rho * vs * A), 1.0)
        dirs = [normal_dir, tang_dir]
        ops.element("zeroLength", DASH_ELE_BASE + ndash, g, model_tag,
                    "-mat", mN, mT, "-dir", *dirs)
        ndash += 1

    for j in range(ny + 1):                       # left & right edges
        trib = h / 2 if j in (0, ny) else h
        je = min(j, ny - 1)
        add_dashpot(1 + j * (nx + 1), 0, je, 1, trib)
        add_dashpot(1 + nx + j * (nx + 1), nx - 1, je, 1, trib)
    for i in range(nx + 1):                       # bottom edge
        trib = h / 2 if i in (0, nx) else h
        ie = min(i, nx - 1)
        add_dashpot(1 + i + ny * (nx + 1), ie, ny - 1, 2, trib)

    mesh = Mesh(nx, ny, h, tags, xy, etags, enodes, elayer, surface,
                left, right, bottom, ndash)
    mesh.save(cfg.out / "mesh.npz")
    return mesh


def select_source_nodes(mesh: Mesh, cfg: Config) -> dict:
    """Centre node + its 4 grid neighbours. Pushing them radially outward with
    the same time function gives zero net force (self-equilibrated)."""
    c = mesh.nearest_node(cfg.source_x, -cfg.source_depth)
    nx = mesh.nx
    tags = np.array([c + 1, c - 1, c - (nx + 1), c + (nx + 1)])   # +x, -x, up, down
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    # sanity: geometry must match
    cx, cy = mesh.xy_of(c)
    for t, d in zip(tags, dirs):
        tx, ty = mesh.xy_of(t)
        assert abs((tx - cx) - d[0] * mesh.h) < 1e-9 and abs((ty - cy) - d[1] * mesh.h) < 1e-9
    return {"center_tag": c, "tags": tags, "dirs": dirs}
