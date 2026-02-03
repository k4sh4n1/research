import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from obspy import UTCDateTime
from obspy.clients.fdsn import Client


def fetch_waveform(client, net, sta, loc, cha, t0, t1):
    return client.get_waveforms(net, sta, loc, cha, t0, t1)[0]


def get_station_coords(client, net, sta):
    """Fetch station coordinates from IRIS."""
    inv = client.get_stations(network=net, station=sta, level="station")
    station = inv[0][0]
    return station.latitude, station.longitude


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


def plot_station_map(station_coords, epicenter, outfile):
    """Plot stations and epicenter on a map."""
    eq_lat, eq_lon = epicenter

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Map extent: [lon_min, lon_max, lat_min, lat_max]
    ax.set_extent([125, 150, 30, 50], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    # Plot epicenter
    ax.plot(
        eq_lon,
        eq_lat,
        marker="*",
        markersize=20,
        color="red",
        transform=ccrs.PlateCarree(),
        label=f"Epicenter M7.6\n({eq_lat}°N, {eq_lon}°E)",
    )

    # Plot stations
    colors = ["blue", "green", "orange", "purple"]
    for i, (name, (lat, lon)) in enumerate(station_coords.items()):
        ax.plot(
            lon,
            lat,
            marker="^",
            markersize=12,
            color=colors[i % len(colors)],
            transform=ccrs.PlateCarree(),
            label=f"{name} ({lat:.2f}°N, {lon:.2f}°E)",
        )
        ax.text(
            lon + 0.5,
            lat + 0.5,
            name.split(".")[1],
            transform=ccrs.PlateCarree(),
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(
        "M7.6 Japan Earthquake (2025-12-08 14:15 UTC)\nEpicenter & Recording Stations",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"✓ Map saved: {outfile}")


def main():
    client = Client("IRIS")

    event = UTCDateTime("2025-12-08T14:15:00")
    t0, t1 = event - 60, event + 20 * 60

    # Epicenter: 80 km off Aomori coast, depth 50 km
    epicenter = (41.0, 143.0)

    stations = [
        ("IU", "MAJO", "00", "BHZ"),
        ("II", "ERM", "00", "BHZ"),
        ("IU", "INCN", "00", "BHZ"),
        ("G", "INU", "00", "BHZ"),
    ]

    station_coords = {}

    for net, sta, loc, cha in stations:
        name = f"{net}.{sta}.{cha}"
        print(f"Processing {name}")

        # Fetch coordinates
        lat, lon = get_station_coords(client, net, sta)
        station_coords[name] = (lat, lon)
        print(f"  Location: {lat:.2f}°N, {lon:.2f}°E")

        # Fetch waveform and create candles
        tr = fetch_waveform(client, net, sta, loc, cha, t0, t1)
        tr.detrend("demean")
        tr.filter("bandpass", freqmin=0.01, freqmax=1.0)

        candles = to_candles(tr)

        title = (
            f"{name} — M7.6 Japan EQ (2025-12-08)\n"
            f"1-Minute Candles | {tr.stats.sampling_rate} Hz"
        )
        outfile = f"candle_{net}_{sta}_{cha}.png"

        plot_candles(candles, title, outfile)

    # Generate station map
    plot_station_map(station_coords, epicenter, "station_map.png")

    print("\n✓ All figures saved!")


if __name__ == "__main__":
    main()
