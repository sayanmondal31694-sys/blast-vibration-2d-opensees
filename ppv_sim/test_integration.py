"""Integration test: node selection, output dimensions, timestamps, finite values,
units. Run:  python test_integration.py   (re-runs the model on a short duration)
"""
import numpy as np

from config import Config, wave_speeds
import model_setup as ms
import dynamic_analysis as da
import ppv_analysis as pa
import signal_analysis as sa


def main():
    cfg = Config(duration=0.5, out_dir="results_test", n_snapshots=4)
    mesh = ms.build_model(cfg)
    da.run_dynamic(cfg, mesh)
    res = pa.load_results(cfg)
    mesh2 = ms.Mesh.load(cfg.out / "mesh.npz")      # downstream reads the saved mesh
    t, nt = res["time"], cfg.nsteps + 1

    # --- node selection: tags/coordinates match the mesh actually built -----
    for tag, xy in zip(res["rec_tags"], res["rec_xy"]):
        assert np.allclose(mesh2.xy_of(tag), xy)
    sx, sy = res["src_xy"]
    for name, dx, depth in cfg.receivers:
        k = list(res["rcv_names"]).index(name)
        x, y = res["rec_xy"][res["rcv_idx"][k]]
        assert abs(x - (sx + dx)) <= cfg.h / 2 + 1e-9 and abs(y + depth) <= cfg.h / 2 + 1e-9, name
    assert np.allclose(res["rec_xy"][res["surface_idx"], 1], 0.0)          # surface at y=0
    assert abs(sx - cfg.source_x) <= cfg.h and abs(sy + cfg.source_depth) <= cfg.h

    # --- output dimensions -----------------------------------------------------
    nr = len(res["rec_tags"])
    for k in ("vel", "acc", "disp"):
        assert res[k].shape == (nt, nr, 2), (k, res[k].shape)
    assert res["snap_vel"].shape[1:] == (len(mesh2.node_tags), 2)

    # --- timestamps --------------------------------------------------------------
    assert t.shape == (nt,) and t[0] == 0.0
    assert np.allclose(np.diff(t), cfg.dt, atol=1e-9)
    assert abs(t[-1] - cfg.duration) < 1e-9
    assert np.allclose(res["pulse_t"], t)

    # --- finite values, non-trivial response --------------------------------------
    for k in ("vel", "acc", "disp", "snap_vel"):
        assert np.all(np.isfinite(res[k])), k
    ppv = pa.compute_ppv(res)
    assert ppv["ppv_vec_mms"].max() > 0

    # --- units consistency: v ~ d(disp)/dt and a ~ d(vel)/dt -----------------------
    i = res["rcv_idx"][0]
    v_num = np.gradient(res["disp"][:, i, 1], t)
    v_rec = res["vel"][:, i, 1]
    assert np.max(np.abs(v_num - v_rec)) < 0.05 * np.max(np.abs(v_rec)), "vel vs disp mismatch"
    a_num = np.gradient(res["vel"][:, i, 1], t)
    a_rec = res["acc"][:, i, 1]
    assert np.max(np.abs(a_num - a_rec)) < 0.15 * np.max(np.abs(a_rec)), "acc vs vel mismatch"

    # --- physics: arrival times consistent with wave speeds --------------------------
    # Surface arrivals must be later than the straight-ray P time in the *fastest*
    # layer and earlier than the straight-ray time at the slowest S speed.
    vp_max = max(wave_speeds(L)[0] for L in cfg.layers)
    vs_min = min(wave_speeds(L)[1] for L in cfg.layers)
    for k, name in enumerate(res["rcv_names"]):
        j = res["rcv_idx"][k]
        R = np.hypot(*(res["rec_xy"][j] - res["src_xy"]))
        tarr = pa.first_arrival(t, np.linalg.norm(res["vel"][:, j, :], axis=1), 0.02)
        t_min = cfg.pulse_t0 * 0.3 + R / vp_max * 0.9        # cannot beat fastest P
        t_max = cfg.pulse_t0 * 2.0 + R / vs_min * 1.2
        assert t_min < tarr < t_max, (name, R, tarr, t_min, t_max)

    # --- signal analysis returns finite numbers ----------------------------------------
    s = sa.analyze_signal(t, res["vel"][:, i, 1])
    assert all(np.isfinite(v) for v in s.values()), s
    print("integration test PASSED")
    print(f"  nodes={len(mesh2.node_tags)}  recorded={nr}  steps={nt - 1}  "
          f"max PPV={ppv['ppv_vec_mms'].max():.2f} mm/s")


if __name__ == "__main__":
    main()
