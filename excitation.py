"""Prescribed synthetic transient pulse (Ricker wavelet).

This is a mathematical forcing function only: no explosive chemistry,
detonation physics or charge design is represented.

Public API
----------
ricker_pulse(cfg) -> dict(t [s], f [-], peak [N/m])
    t = 0..duration inclusive at cfg.dt, f normalised to unit peak magnitude.
    Actual nodal force = cfg.pulse_peak_force * f(t).
"""
import numpy as np
from config import Config


def ricker_pulse(cfg: Config) -> dict:
    t = np.arange(0, cfg.nsteps + 1) * cfg.dt
    a = (np.pi * cfg.pulse_fc * (t - cfg.pulse_t0)) ** 2
    f = (1.0 - 2.0 * a) * np.exp(-a)
    f = f / np.max(np.abs(f))
    return {"t": t, "f": f, "peak": cfg.pulse_peak_force}
