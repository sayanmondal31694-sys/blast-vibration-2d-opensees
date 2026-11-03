"""Transient (Newmark) analysis on the domain left by model_setup.build_model.

Public API
----------
run_dynamic(cfg, mesh) -> Path to <out_dir>/results.npz

results.npz schema (SI units):
    time        (nt,)            s, time[0] = 0
    rec_tags    (nr,)            OpenSees node tags recorded (surface row + receivers)
    rec_xy      (nr,2)           m
    surface_idx (ns,)            indices into rec_tags of the surface row, ordered by x
    rcv_names   (nrc,) str       configured receiver names
    rcv_idx     (nrc,)           indices into rec_tags
    vel, acc, disp  (nt,nr,2)    m/s, m/s^2, m  (component 0 = x, 1 = y)
    src_tag, src_xy              source centre node
    pulse_t, pulse_f             forcing history (unit peak) and pulse_peak [N/m]
    snap_t (nsn,), snap_vel (nsn,N,2)   full-field velocity snapshots
    dt, damp_alpha, damp_beta
"""
from pathlib import Path
import numpy as np
import openseespy.opensees as ops

from config import Config
from model_setup import Mesh, select_source_nodes
from excitation import ricker_pulse


def run_dynamic(cfg: Config, mesh: Mesh) -> Path:
    # the live domain must be the one build_model made (no rebuilding here)
    live = set(ops.getNodeTags())
    assert set(int(t) for t in mesh.node_tags) <= live, "domain != mesh; call build_model first"

    # ---- recorded node set: surface row + named receivers -----------------
    src = select_source_nodes(mesh, cfg)
    sx, _ = mesh.xy_of(src["center_tag"])
    rcv_tags, rcv_names = [], []
    for name, dx, depth in cfg.receivers:
        rcv_tags.append(mesh.nearest_node(sx + dx, -depth))
        rcv_names.append(name)
    rec_tags = list(dict.fromkeys([int(t) for t in mesh.surface_tags] + rcv_tags))
    idx = {t: k for k, t in enumerate(rec_tags)}
    surface_idx = np.array([idx[int(t)] for t in mesh.surface_tags])
    rcv_idx = np.array([idx[t] for t in rcv_tags])

    # ---- loading: Path time series, self-equilibrated radial cluster ------
    pulse = ricker_pulse(cfg)
    ops.timeSeries("Path", 1, "-dt", cfg.dt, "-values", *pulse["f"].tolist(), "-factor", 1.0)
    ops.pattern("Plain", 1, 1)
    for tag, d in zip(src["tags"], src["dirs"]):
        ops.load(int(tag), *(pulse["peak"] * d).tolist())

    # ---- damping (Rayleigh, target ratio between f1 and f2) ----------------
    w1, w2 = 2 * np.pi * cfg.damp_f1, 2 * np.pi * cfg.damp_f2
    alpha = 2 * cfg.damp_ratio * w1 * w2 / (w1 + w2)
    beta = 2 * cfg.damp_ratio / (w1 + w2)
    ops.rayleigh(alpha, beta, 0.0, 0.0)

    # ---- analysis ------------------------------------------------------------
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandSPD")                 # effective Newmark tangent is symmetric positive definite
    ops.test("NormUnbalance", 1e-6, 20)
    ops.algorithm("Linear", "-factorOnce")  # linear model + constant dt: factorize once
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    nt, nr, N = cfg.nsteps + 1, len(rec_tags), len(mesh.node_tags)
    vel = np.zeros((nt, nr, 2)); acc = np.zeros_like(vel); disp = np.zeros_like(vel)
    time = np.zeros(nt)
    snap_steps = set(np.linspace(0, cfg.nsteps, cfg.n_snapshots + 1)[1:].astype(int).tolist())
    snap_t, snap_vel = [], []

    for n in range(1, nt):
        if ops.analyze(1, cfg.dt) != 0:
            raise RuntimeError(f"analysis failed at step {n}, t={n * cfg.dt:.4f}s")
        time[n] = ops.getTime()
        for k, t in enumerate(rec_tags):
            vel[n, k] = ops.nodeVel(t)
            acc[n, k] = ops.nodeAccel(t)
            disp[n, k] = ops.nodeDisp(t)
        if n in snap_steps:
            snap_t.append(time[n])
            snap_vel.append(np.array([ops.nodeVel(int(t)) for t in mesh.node_tags]))

    path = cfg.out / "results.npz"
    np.savez(path,
             time=time, rec_tags=np.array(rec_tags), rec_xy=mesh.node_xy[np.array(rec_tags) - 1],
             surface_idx=surface_idx, rcv_names=np.array(rcv_names), rcv_idx=rcv_idx,
             vel=vel, acc=acc, disp=disp,
             src_tag=src["center_tag"], src_xy=mesh.xy_of(src["center_tag"]),
             pulse_t=pulse["t"], pulse_f=pulse["f"], pulse_peak=pulse["peak"],
             snap_t=np.array(snap_t), snap_vel=np.array(snap_vel),
             dt=cfg.dt, damp_alpha=alpha, damp_beta=beta)
    return path
