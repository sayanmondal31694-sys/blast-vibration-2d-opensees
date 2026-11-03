"""PPV extraction from results.npz (reads the solver's exact histories).

Public API
----------
load_results(cfg) -> dict                      (the results.npz arrays)
compute_ppv(res, idx=None) -> dict of arrays   (idx: indices into rec_tags;
                                                default = all surface nodes)
    keys: tag, x, y, R_m, ppv_x_mms, ppv_y_mms, ppv_vec_mms, t_peak_s, t_arr_s
fit_attenuation(R, ppv, r_min, r_max) -> dict(K, n, r2, ...)   PPV[mm/s] = K * R[m]^-n
write_ppv_csv(ppv, path)

2D plane-strain note: only in-plane components exist (x = radial/horizontal,
y = vertical). The out-of-plane transverse component is not modelled, and
plane-strain geometric spreading is cylindrical, so the fitted exponent is
NOT transferable to 3D field data.
"""
import csv
import numpy as np
from config import Config

MMS = 1e3   # m/s -> mm/s


def load_results(cfg: Config) -> dict:
    z = np.load(cfg.out / "results.npz")
    return {k: z[k] for k in z.files}


def first_arrival(t, v_resultant, frac=0.05):
    """First time the resultant velocity exceeds frac * its peak."""
    thr = frac * np.max(v_resultant)
    k = int(np.argmax(v_resultant >= thr))
    return float(t[k])


def compute_ppv(res: dict, idx=None) -> dict:
    if idx is None:
        idx = res["surface_idx"]
    idx = np.asarray(idx)
    t = res["time"]
    v = res["vel"][:, idx, :]                       # (nt, n, 2) m/s
    vec = np.linalg.norm(v, axis=2)                 # (nt, n)
    xy = res["rec_xy"][idx]
    src = res["src_xy"]
    R = np.hypot(xy[:, 0] - src[0], xy[:, 1] - src[1])
    kpk = np.argmax(vec, axis=0)
    return {
        "tag": res["rec_tags"][idx].astype(int),
        "x": xy[:, 0], "y": xy[:, 1], "R_m": R,
        "ppv_x_mms": np.abs(v[:, :, 0]).max(axis=0) * MMS,
        "ppv_y_mms": np.abs(v[:, :, 1]).max(axis=0) * MMS,
        "ppv_vec_mms": vec.max(axis=0) * MMS,
        "t_peak_s": t[kpk],
        "t_arr_s": np.array([first_arrival(t, vec[:, k]) if vec[:, k].max() > 0 else np.nan
                             for k in range(vec.shape[1])]),
    }


def fit_attenuation(R, ppv, r_min=15.0, r_max=100.0) -> dict:
    """Log-log fit over r_min <= R <= r_max. Excludes the overhead peak near the
    source and the nodes close to the viscous side boundaries."""
    m = (R >= r_min) & (R <= r_max) & (ppv > 0)
    A = np.vstack([np.ones(m.sum()), np.log(R[m])]).T
    coef, *_ = np.linalg.lstsq(A, np.log(ppv[m]), rcond=None)
    pred = A @ coef
    ss_res = np.sum((np.log(ppv[m]) - pred) ** 2)
    ss_tot = np.sum((np.log(ppv[m]) - np.log(ppv[m]).mean()) ** 2)
    return {"K": float(np.exp(coef[0])), "n": float(-coef[1]),
            "r2": float(1 - ss_res / ss_tot), "r_min_m": float(r_min),
            "r_max_m": float(r_max), "n_points": int(m.sum())}


def write_ppv_csv(ppv: dict, path):
    keys = list(ppv.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(ppv["tag"])):
            w.writerow([f"{ppv[k][i]:.6g}" for k in keys])
