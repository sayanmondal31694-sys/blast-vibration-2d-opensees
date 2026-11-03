"""Time-series (frequency / energy) analysis of a recorded velocity history.

Public API  (v in m/s, t in s, uniform spacing)
----------
preprocess(t, v, band=None) -> v_clean        detrend + taper (+ optional bandpass)
fft_spectrum(t, v)          -> f [Hz], amp [m/s * s = m]   one-sided amplitude
dominant_frequency(f, amp)  -> Hz
zero_crossing_frequency(t, v) -> Hz           frequency of the cycle containing the peak
psd(t, v)                   -> f, Pxx         Welch PSD
spectrogram(t, v)           -> f, ts, Sxx     STFT power
energy_duration(t, v)       -> dict(t5, t95, duration, energy)   cumulative v^2 integral
analyze_signal(t, v, band=None) -> dict summary (floats only)
"""
import numpy as np
from scipy import signal


def preprocess(t, v, band=None, taper=0.05):
    v = signal.detrend(np.asarray(v, float), type="linear")
    v = v * signal.windows.tukey(len(v), alpha=taper)
    if band is not None:
        fs = 1.0 / (t[1] - t[0])
        sos = signal.butter(4, band, btype="band", fs=fs, output="sos")
        v = signal.sosfiltfilt(sos, v)
    return v


def fft_spectrum(t, v, nfft=None):
    dt = t[1] - t[0]
    n = nfft or int(2 ** np.ceil(np.log2(len(v))) * 2)       # zero-pad for smoother curve
    amp = np.abs(np.fft.rfft(v, n)) * dt
    return np.fft.rfftfreq(n, dt), amp


def dominant_frequency(f, amp, fmin=1.0):
    m = f >= fmin
    return float(f[m][np.argmax(amp[m])])


def zero_crossing_frequency(t, v):
    """Frequency (Hz) from the zero crossings bracketing the absolute peak,
    as used in blast-monitoring practice (2 x half-cycle duration)."""
    kp = int(np.argmax(np.abs(v)))
    s = np.sign(v)
    cross = np.where(s[:-1] * s[1:] < 0)[0]
    before = cross[cross < kp]
    after = cross[cross >= kp]
    if len(before) == 0 or len(after) == 0:
        return float("nan")
    t_before = t[before[-1]]
    t_after = t[after[0]]
    return float(1.0 / (2.0 * (t_after - t_before)))


def psd(t, v):
    fs = 1.0 / (t[1] - t[0])
    return signal.welch(v, fs=fs, nperseg=min(len(v), 512))


def spectrogram(t, v):
    fs = 1.0 / (t[1] - t[0])
    nper = 256        # 0.128 s window at 2 kHz -> ~7.8 Hz resolution (was 128 -> 15.6 Hz)
    f, ts, S = signal.spectrogram(v, fs=fs, nperseg=nper, noverlap=int(nper * 0.9),
                                  nfft=1024, scaling="spectrum")
    return f, ts + t[0], S


def energy_duration(t, v):
    e = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] ** 2 + v[:-1] ** 2) * np.diff(t))])
    tot = e[-1]
    if tot <= 0:
        return {"t5": np.nan, "t95": np.nan, "duration": np.nan, "energy": 0.0, "cum": e}
    c = e / tot
    t5, t95 = float(t[np.argmax(c >= 0.05)]), float(t[np.argmax(c >= 0.95)])
    return {"t5": t5, "t95": t95, "duration": t95 - t5, "energy": float(tot), "cum": c}


def analyze_signal(t, v, band=None) -> dict:
    vc = preprocess(t, v, band)
    f, amp = fft_spectrum(t, vc)
    ed = energy_duration(t, vc)
    return {
        "peak_mms": float(np.max(np.abs(v)) * 1e3),
        "dominant_freq_hz": dominant_frequency(f, amp),
        "zero_cross_freq_hz": zero_crossing_frequency(t, vc),
        "spectral_centroid_hz": float(np.sum(f * amp) / np.sum(amp)),
        "t5_s": ed["t5"], "t95_s": ed["t95"], "sig_duration_s": ed["duration"],
        "energy_m2_per_s": ed["energy"],
    }
