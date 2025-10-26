#!/usr/bin/env python
"""
Fetch acceleration data, integrate twice, and plot ground displacement.
"""

import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime, Stream, Trace
from obspy.clients.fdsn import Client
from scipy import signal

def fetch_acceleration_trace():
    """Try to fetch a short acceleration record, or make synthetic data."""
    try:
        client = Client("IRIS")
        t0 = UTCDateTime("2022-12-20T10:34:24")
        st = client.get_waveforms("CI", "FMP", "*", "HNZ",
                                  t0 - 60, t0 + 240)
        if st:
            print("Fetched real data:", st)
            return st[0]
    except Exception as e:
        print("Could not fetch real data:", e)

    # --- synthetic fallback: 60 s at 100 Hz, two wavelets ---
    sr = 100.0
    t = np.arange(0, 60, 1/sr)
    a = (0.5*np.exp(-(t-10)**2/0.5)*np.sin(2*np.pi*10*(t-10)) +
         1.5*np.exp(-(t-15)**2/1.0)*np.sin(2*np.pi*5*(t-15)) +
         0.05*np.random.randn(len(t))) * 100.0  # cm/s²
    tr = Trace(a)
    tr.stats.sampling_rate = sr
    tr.stats.station = "SYN"
    tr.stats.channel = "HNZ"
    print("Using synthetic acceleration data.")
    return tr

def integrate_to_displacement(tr):
    """Band‑limit and integrate twice to get displacement (cm)."""
    tr = tr.copy()
    tr.detrend("linear")
    tr.detrend("demean")
    tr.filter("bandpass", freqmin=0.1, freqmax=20.0, corners=4)

    dt = 1.0 / tr.stats.sampling_rate
    a = tr.data
    # integrate a→v→d with trapezoidal rule
    v = np.cumsum(0.5 * (a[1:] + a[:-1])) * dt
    v = np.insert(v, 0, 0.0)
    v = signal.detrend(v)
    d = np.cumsum(0.5 * (v[1:] + v[:-1])) * dt
    d = np.insert(d, 0, 0.0)
    d = signal.detrend(d)
    out = tr.copy()
    out.data = d
    return out

def plot_displacement(tr_disp):
    """Simple time‑displacement plot."""
    sr = tr_disp.stats.sampling_rate
    t = np.arange(tr_disp.stats.npts) / sr
    plt.figure(figsize=(12, 5))
    plt.plot(t, tr_disp.data, 'b-', lw=1)
    plt.fill_between(t, 0, tr_disp.data, color='steelblue', alpha=0.3)
    plt.title(f"Ground Displacement  {tr_disp.stats.station}.{tr_disp.stats.channel}")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (cm)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ground_displacement.png", dpi=150)
    plt.show()
    print("Saved plot to ground_displacement.png")

def main():
    tr_acc = fetch_acceleration_trace()
    tr_disp = integrate_to_displacement(tr_acc)
    print(f"Peak displacement: {np.max(np.abs(tr_disp.data)):.3f} cm")
    plot_displacement(tr_disp)

if __name__ == "__main__":
    main()
