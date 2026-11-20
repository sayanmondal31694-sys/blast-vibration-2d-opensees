"""End-to-end driver:  python main.py   (outputs in ./results)"""
import json
import time
import numpy as np

from config import Config, wave_speeds
import model_setup as ms
import dynamic_analysis as da
import ppv_analysis as pa
import signal_analysis as sa
import visualization as viz


def main(cfg: Config | None = None, plots: bool = True):
    cfg = cfg or Config()
    out = cfg.out
    cfg.to_json(out / "config.json")

    for name, f_res, f_need, ok in ms.check_resolution(cfg):
        print(f"[mesh] {name:10s} resolves {f_res:5.1f} Hz (need {f_need:.1f} Hz) "
              f"{'OK' if ok else 'WARNING: refine mesh or lower pulse_fc'}")

    t0 = time.time()
    mesh = ms.build_model(cfg)
    print(f"[mesh] {len(mesh.node_tags)} nodes, {len(mesh.ele_tags)} quads, "
          f"{mesh.n_dashpots} dashpot pairs")
    da.run_dynamic(cfg, mesh)
    print(f"[solve] {cfg.nsteps} steps in {time.time() - t0:.1f} s")

    res = pa.load_results(cfg)

    # PPV: all surface nodes (attenuation) and named receivers
    ppv = pa.compute_ppv(res)
    pa.write_ppv_csv(ppv, out / "ppv.csv")
    fit = pa.fit_attenuation(ppv["R_m"], ppv["ppv_vec_mms"], r_min=15.0, r_max=100.0)
    rcv = pa.compute_ppv(res, res["rcv_idx"])

    # Time-series analysis on each named receiver (vertical + horizontal)
    t = res["time"]
    sig = {}
    for name, i in zip(res["rcv_names"], res["rcv_idx"]):
        sig[str(name)] = {c: sa.analyze_signal(t, res["vel"][:, i, k])
                          for k, c in enumerate(("x", "y"))}

    summary = {"attenuation_fit": fit,
               "receivers": {str(rcv["tag"][k]): {
                   "name": str(res["rcv_names"][k]), "R_m": float(rcv["R_m"][k]),
                   "ppv_vec_mms": float(rcv["ppv_vec_mms"][k]),
                   "t_arr_s": float(rcv["t_arr_s"][k])} for k in range(len(rcv["tag"]))}}
    (out / "ppv_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "signal_summary.json").write_text(json.dumps(sig, indent=2))

    print("\nReceiver   R(m)  PPV_vec(mm/s)  t_arr(s)  f_dom_y(Hz)  f_zc_y(Hz)  dur_y(s)")
    for k, n in enumerate(res["rcv_names"]):
        s = sig[str(n)]["y"]
        print(f"{str(n):8s} {rcv['R_m'][k]:6.1f} {rcv['ppv_vec_mms'][k]:13.3f} "
              f"{rcv['t_arr_s'][k]:9.3f} {s['dominant_freq_hz']:11.1f} "
              f"{s['zero_cross_freq_hz']:11.1f} {s['sig_duration_s']:9.3f}")
    print(f"\nPPV ~ {fit['K']:.3g} * R^-{fit['n']:.2f}  (R2 = {fit['r2']:.2f}, "
          f"{fit['r_min_m']:.0f} m <= R <= {fit['r_max_m']:.0f} m, {fit['n_points']} nodes)")

    if plots:
        mid = str(res["rcv_names"][len(res["rcv_names"]) // 2])
        viz.plot_model(mesh, cfg, res, out / "01_model.png")
        viz.plot_pulse(res, out / "02_pulse.png")
        viz.plot_waveforms(res, out / "03_waveforms_vertical.png", comp=1)
        viz.plot_waveforms(res, out / "04_waveforms_horizontal.png", comp=0)
        viz.plot_ppv_distance(ppv, fit, out / "05_ppv_distance.png")
        viz.plot_spectra(res, out / "06_spectra.png")
        viz.plot_spectrogram(res, mid, out / "07_spectrogram.png")
        viz.plot_energy(res, out / "08_energy.png")
        viz.plot_snapshots(mesh, res, cfg, out / "09_snapshots.png")
    return cfg, mesh, res


if __name__ == "__main__":
    main()
