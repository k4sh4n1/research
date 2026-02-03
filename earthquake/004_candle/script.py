import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from obspy import UTCDateTime
from obspy.clients.fdsn import Client


def fetch_waveform(client, net, sta, loc, cha, t0, t1):
    return client.get_waveforms(net, sta, loc, cha, t0, t1)[0]


def to_candles(trace, minutes=1):
    sr = trace.stats.sampling_rate
    step = int(minutes * 60 * sr)
    data = trace.data

    candles = []
    for i in range(len(data) // step):
        seg = data[i * step : (i + 1) * step]
        candles.append((i, seg[0], seg.max(), seg.min(), seg[-1]))

    return candles


def plot_candles(candles, title, outfile):
    fig, ax = plt.subplots(figsize=(14, 5))

    for x, o, h, l, c in candles:
        color = "green" if c >= o else "red"
        ax.plot([x, x], [l, h], color=color, lw=0.8)

        body_y = min(o, c)
        body_h = abs(c - o) or (h - l) * 0.01
        ax.add_patch(
            Rectangle((x - 0.35, body_y), 0.7, body_h, facecolor=color, edgecolor=color)
        )

    ax.set(xlabel="Minutes from start", ylabel="Amplitude (counts)", title=title)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)


def main():
    client = Client("IRIS")

    event = UTCDateTime("2025-12-08T14:15:00")
    t0, t1 = event - 60, event + 20 * 60

    stations = [
        ("IU", "MAJO", "00", "BHZ"),
        ("II", "ERM", "00", "BHZ"),
        ("IU", "INCN", "00", "BHZ"),
        ("G", "INU", "00", "BHZ"),
    ]

    for net, sta, loc, cha in stations:
        print(f"Processing {net}.{sta}.{cha}")

        tr = fetch_waveform(client, net, sta, loc, cha, t0, t1)
        tr.detrend("demean")
        tr.filter("bandpass", freqmin=0.01, freqmax=1.0)

        candles = to_candles(tr)

        title = (
            f"{net}.{sta}.{cha} — M7.6 Japan EQ (2025-12-08)\n"
            f"1‑Minute Candles | {tr.stats.sampling_rate} Hz"
        )
        outfile = f"candle_{net}_{sta}_{cha}.png"

        plot_candles(candles, title, outfile)

    print("✓ All figures saved")


if __name__ == "__main__":
    main()
