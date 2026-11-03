"""Matplotlib figures (PNG). Categorical colours in fixed order; thin lines,
recessive grid, legend whenever >= 2 series, text in neutral ink."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import signal_analysis as sa

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 1.2,
})


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model(mesh, cfg, res, path):
    """Layers, boundary conditions, source and receivers."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    lay = mesh.ele_layer.reshape(mesh.ny, mesh.nx)
    cmap = ListedColormap(["#efe4c8", "#c9d3dc", "#d6cfe6"][: len(cfg.layers)])
    ax.imshow(lay, extent=[0, cfg.width, -cfg.depth, 0], cmap=cmap, aspect="equal", alpha=0.8)
    ax.grid(False)
    z = 0.0
    for L in cfg.layers[:-1]:
        z += L.thickness
        ax.axhline(-z, color=INK2, lw=0.6, ls="--")
    for k, L in enumerate(cfg.layers):
        top = -sum(l.thickness for l in cfg.layers[:k])
        ax.text(cfg.width * 0.015, top - 3, L.name, color=INK2, fontsize=8, va="top")
    ax.plot(res["rec_xy"][res["surface_idx"], 0], np.zeros(len(res["surface_idx"])),
            color=INK, lw=2.0)
    ax.text(cfg.width / 2, 2.5, "free surface", ha="center", color=INK2, fontsize=8)
    for xs, ys, lbl, ha in [(0, -cfg.depth / 2, "viscous\nboundary", "right"),
                            (cfg.width, -cfg.depth / 2, "viscous\nboundary", "left")]:
        ax.text(xs, ys, lbl, ha=ha, va="center", color=INK2, fontsize=8)
    ax.text(cfg.width / 2, -cfg.depth - 6, "viscous boundary (bottom)", ha="center",
            color=INK2, fontsize=8)
    ax.scatter(*res["src_xy"], marker="*", s=140, color=SERIES[7], zorder=5, label="source cluster")
    for k, (n, i) in enumerate(zip(res["rcv_names"], res["rcv_idx"])):
        x, y = res["rec_xy"][i]
        ax.scatter(x, y, s=28, color=SERIES[0], edgecolor=SURF, linewidth=1.2, zorder=5,
                   label="receivers" if k == 0 else None)
        ax.annotate(str(n), (x, y), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=7.5, color=INK)
    ax.set_xlim(-25, cfg.width + 25); ax.set_ylim(-cfg.depth - 12, 12)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Model: layers, boundary conditions, source and receivers")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, path)


def plot_pulse(res, path):
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(res["pulse_t"], res["pulse_f"] * res["pulse_peak"] / 1e6, color=SERIES[0])
    ax.set_xlabel("time (s)"); ax.set_ylabel("force per node (MN/m)")
    ax.set_title("Prescribed source pulse (Ricker)")
    return _save(fig, path)


def plot_waveforms(res, path, comp=1):
    names, idx = res["rcv_names"], res["rcv_idx"]
    fig, axes = plt.subplots(len(idx), 1, figsize=(8, 1.5 * len(idx) + 0.8), sharex=True)
    lab = "vertical" if comp == 1 else "horizontal"
    for k, (a, n, i) in enumerate(zip(np.atleast_1d(axes), names, idx)):
        a.plot(res["time"], res["vel"][:, i, comp] * 1e3, color=SERIES[k % 8])
        a.set_ylabel(f"{n}\n(mm/s)", rotation=0, ha="right", va="center")
        a.axhline(0, color=GRID, lw=0.8)
    a.set_xlabel("time (s)")
    np.atleast_1d(axes)[0].set_title(f"Particle velocity, {lab} component")
    fig.tight_layout()
    return _save(fig, path)


def plot_ppv_distance(ppv, fit, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    m = ppv["R_m"] > 0
    ax.loglog(ppv["R_m"][m], ppv["ppv_vec_mms"][m], ".", ms=4, color=SERIES[0], label="surface nodes")
    r = np.linspace(fit["r_min_m"], fit["r_max_m"], 50)
    ax.axvspan(fit["r_min_m"], fit["r_max_m"], color=GRID, alpha=0.5, lw=0)
    ax.loglog(r, fit["K"] * r ** (-fit["n"]), color=SERIES[1], lw=1.4,
              label=f"fit {fit['r_min_m']:.0f}–{fit['r_max_m']:.0f} m: "
                    f"{fit['K']:.3g}·R^-{fit['n']:.2f}  (R²={fit['r2']:.2f})")
    ax.set_xlabel("distance from source R (m)"); ax.set_ylabel("resultant PPV (mm/s)")
    ax.set_title("PPV vs distance (2D plane strain)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_spectra(res, path, band=None):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    t = res["time"]
    for k, (n, i) in enumerate(zip(res["rcv_names"], res["rcv_idx"])):
        v = sa.preprocess(t, res["vel"][:, i, 1], band)
        f, a = sa.fft_spectrum(t, v)
        m = f <= 80
        ax.plot(f[m], a[m] * 1e3, color=SERIES[k % 8], label=str(n))
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("amplitude (mm/s · s)")
    ax.set_title("Fourier amplitude spectra, vertical velocity")
    ax.legend(ncol=3, fontsize=8)
    return _save(fig, path)


def plot_spectrogram(res, name, path):
    t = res["time"]
    k = list(res["rcv_names"]).index(name)
    i = res["rcv_idx"][k]
    f, ts, S = sa.spectrogram(t, sa.preprocess(t, res["vel"][:, i, 1]))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    m = f <= 80
    im = ax.pcolormesh(ts, f[m], 10 * np.log10(S[m] + 1e-30), cmap="viridis",
                       vmin=10 * np.log10(S.max() + 1e-30) - 50, shading="auto")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="power (dB, relative)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"Spectrogram, {name} vertical velocity")
    return _save(fig, path)


def plot_energy(res, path):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    t = res["time"]
    for k, (n, i) in enumerate(zip(res["rcv_names"], res["rcv_idx"])):
        ed = sa.energy_duration(t, sa.preprocess(t, res["vel"][:, i, 1]))
        ax.plot(t, np.nan_to_num(ed["cum"]), color=SERIES[k % 8], label=str(n))
    ax.axhline(0.05, color=GRID, lw=0.8); ax.axhline(0.95, color=GRID, lw=0.8)
    ax.set_xlabel("time (s)"); ax.set_ylabel("normalised cumulative v² energy")
    ax.set_title("Cumulative energy (5–95 % significant duration)")
    ax.legend(ncol=3, fontsize=8)
    return _save(fig, path)


def plot_snapshots(mesh, res, cfg, path, ncols=4):
    sv, st = res["snap_vel"], res["snap_t"]
    n = len(st)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 1.9 * nrows + 0.6), squeeze=False)
    mag = np.linalg.norm(sv, axis=2) * 1e3
    vmax = np.percentile(mag, 99.5)
    for k, a in enumerate(axes.ravel()):
        if k >= n:
            a.axis("off"); continue
        im = a.imshow(mag[k].reshape(mesh.ny + 1, mesh.nx + 1),
                      extent=[0, cfg.width, -cfg.depth, 0], cmap="magma_r",
                      vmin=0, vmax=vmax, aspect="equal")
        a.grid(False)
        a.set_title(f"t = {st[k]:.2f} s", fontsize=8)
        a.tick_params(labelsize=7)
    fig.colorbar(im, ax=axes, label="|v| (mm/s)", shrink=0.8)
    fig.suptitle("Wavefield snapshots (velocity magnitude)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    return _save(fig, path)
