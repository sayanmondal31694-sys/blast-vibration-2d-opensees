"""Shared configuration and data schema for the 2D blast-vibration workflow.

UNITS (SI everywhere, nothing is converted inside the solver):
    length m | time s | mass kg | force N (per metre of out-of-plane thickness
    for the 2D plane-strain model, i.e. N/m) | stress Pa | velocity m/s |
    acceleration m/s^2 | frequency Hz.
    PPV is *reported* in mm/s (field name suffix `_mms`).

COORDINATES: x horizontal (0 at left edge), y vertical, ground surface at y = 0,
depth is negative y (bottom of model at y = -depth).

The blast is NOT modelled physically (no explosive chemistry / detonation /
charge design). The source is a prescribed synthetic transient pulse applied as
a self-equilibrated radial force cluster.

FILES (all under Config.out_dir):
    mesh.npz       written by model_setup.build_model
    results.npz    written by dynamic_analysis.run_dynamic
    ppv.csv, ppv_summary.json, signal_summary.json, *.png   downstream outputs
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class Layer:
    name: str
    thickness: float   # m (the last layer is extended to the model bottom)
    E: float           # Pa
    nu: float          # -
    rho: float         # kg/m^3


@dataclass
class Config:
    # --- geometry / mesh -------------------------------------------------
    width: float = 240.0          # m
    depth: float = 100.0          # m
    h: float = 2.0                # m, square element size
    thickness: float = 1.0        # m, out-of-plane (plane strain, per metre)

    # --- ground model (horizontal layers, top to bottom) ------------------
    layers: list = field(default_factory=lambda: [
        Layer("weathered", 10.0, 2.0e9, 0.30, 2000.0),
        Layer("rock", 90.0, 20.0e9, 0.25, 2600.0),
    ])

    # --- synthetic source pulse (prescribed, not physical) ----------------
    source_x: float = 120.0       # m
    source_depth: float = 15.0    # m below surface
    pulse_fc: float = 15.0        # Hz, Ricker centre frequency
    pulse_t0: float = 0.1         # s, pulse delay (>= 1.5/fc to start at ~0)
    pulse_peak_force: float = 2.0e6   # N/m on each of the 4 cluster nodes

    # --- time integration --------------------------------------------------
    dt: float = 0.0005            # s
    duration: float = 1.2         # s
    damp_ratio: float = 0.01      # Rayleigh target ratio
    damp_f1: float = 5.0          # Hz
    damp_f2: float = 50.0         # Hz

    # --- receivers: (name, x_offset_from_source [m], depth [m]) -----------
    receivers: list = field(default_factory=lambda: [
        ("S15", 15.0, 0.0), ("S30", 30.0, 0.0), ("S50", 50.0, 0.0),
        ("S75", 75.0, 0.0), ("S95", 95.0, 0.0),
        ("D50", 50.0, 20.0),
    ])
    n_snapshots: int = 8

    # --- meshing quality rule ---------------------------------------------
    pts_per_wavelength: float = 8.0
    f_max_factor: float = 2.5     # highest usable frequency = factor * fc

    # --- I/O ----------------------------------------------------------------
    out_dir: str = "results"

    # helpers ---------------------------------------------------------------
    @property
    def out(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def nx(self) -> int:
        return int(round(self.width / self.h))

    @property
    def ny(self) -> int:
        return int(round(self.depth / self.h))

    @property
    def nsteps(self) -> int:
        return int(round(self.duration / self.dt))

    def layer_at_depth(self, depth_m: float) -> int:
        """Index of the layer containing a depth (m, positive down)."""
        z = 0.0
        for i, L in enumerate(self.layers):
            z += L.thickness
            if depth_m < z or i == len(self.layers) - 1:
                return i
        return len(self.layers) - 1

    def to_json(self, path):
        d = asdict(self)
        Path(path).write_text(json.dumps(d, indent=2))


def wave_speeds(L: Layer):
    """(Vp, Vs) in m/s for plane-strain constrained/shear moduli."""
    G = L.E / (2.0 * (1.0 + L.nu))
    lam = L.E * L.nu / ((1 + L.nu) * (1 - 2 * L.nu))
    return ((lam + 2 * G) / L.rho) ** 0.5, (G / L.rho) ** 0.5
