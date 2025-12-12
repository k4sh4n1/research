#!/usr/bin/env python
"""
Tectonic Correlation Analysis using ObsPy
Investigates seismic record correlations across major plate boundaries
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.cross_correlation import correlate, xcorr_max
from scipy import signal

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION: Strategic Tectonic Stations
# =============================================================================

TECTONIC_STATIONS = {
    "Turkey_Collision": {
        "network": "II",
        "station": "KIV",  # Kiver, Caucasus region (II network version)
        "location": "*",
        "channel": "BHZ",
        "coords": (43.956, 42.690),
        "tectonic_setting": "Arabian-Eurasian Collision",
    },
    "Japan_Subduction": {
        "network": "II",
        "station": "ERM",  # Erimo, Japan
        "location": "*",
        "channel": "BHZ",
        "coords": (42.015, 143.157),
        "tectonic_setting": "Pacific-Eurasian Subduction",
    },
    "Alaska_Subduction": {
        "network": "II",
        "station": "KDAK",  # Kodiak Island, Alaska
        "location": "*",
        "channel": "BHZ",
        "coords": (57.782, -152.583),
        "tectonic_setting": "Pacific-North American Subduction",
    },
    "Iceland_MidAtlantic": {
        "network": "II",
        "station": "BORG",  # Borgarnes, Iceland
        "location": "*",
        "channel": "BHZ",
        "coords": (64.747, -21.327),
        "tectonic_setting": "Mid-Atlantic Ridge Divergence",
    },
    "California_Transform": {
        "network": "II",
        "station": "PFO",  # Piñon Flat Observatory
        "location": "*",
        "channel": "BHZ",
        "coords": (33.611, -116.455),
        "tectonic_setting": "Pacific-North American Transform",
    },
    "Indonesia_Collision": {
        "network": "II",
        "station": "KAPI",  # Kappang, Sulawesi, Indonesia
        "location": "*",
        "channel": "BHZ",
        "coords": (-5.014, 119.752),
        "tectonic_setting": "Australian-Eurasian Collision/Subduction",
    },
}


# =============================================================================
# DATA ACQUISITION FUNCTIONS
# =============================================================================


def fetch_seismic_data(stations_dict, starttime, endtime, client_name="IRIS"):
    """
    Fetch seismic waveforms from FDSN web services
    """
    client = Client(client_name)
    streams = {}

    print(f"\n{'=' * 60}")
    print(f"Fetching data from {starttime} to {endtime}")
    print(f"{'=' * 60}\n")

    for region, params in stations_dict.items():
        try:
            print(f"📍 Fetching: {region} ({params['network']}.{params['station']})")
            st = client.get_waveforms(
                network=params["network"],
                station=params["station"],
                location=params["location"],
                channel=params["channel"],
                starttime=starttime,
                endtime=endtime,
            )

            # Merge traces if fragmented
            st.merge(method=1, fill_value="interpolate")
            streams[region] = st
            print(f"   ✓ Success: {len(st)} trace(s), {st[0].stats.npts} samples")

        except Exception as e:
            print(f"   ✗ Failed: {str(e)[:50]}")
            streams[region] = None

    return streams


def preprocess_streams(streams, freqmin=0.01, freqmax=1.0):
    """
    Preprocess seismic data for correlation analysis
    """
    processed = {}

    print(f"\n{'=' * 60}")
    print(f"Preprocessing: Bandpass {freqmin}-{freqmax} Hz")
    print(f"{'=' * 60}\n")

    for region, st in streams.items():
        if st is None:
            processed[region] = None
            continue

        try:
            st_copy = st.copy()

            # Remove mean and trend
            st_copy.detrend("demean")
            st_copy.detrend("linear")

            # Taper edges
            st_copy.taper(max_percentage=0.05)

            # Bandpass filter (focus on long-period for teleseismic)
            st_copy.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4)

            # Normalize
            for tr in st_copy:
                tr.data = tr.data / np.max(np.abs(tr.data))

            processed[region] = st_copy
            print(f"   ✓ {region}: Preprocessed successfully")

        except Exception as e:
            print(f"   ✗ {region}: {str(e)[:40]}")
            processed[region] = None

    return processed


# =============================================================================
# CORRELATION ANALYSIS FUNCTIONS
# =============================================================================


def compute_cross_correlations(streams, stations_dict):
    """
    Compute pairwise cross-correlations between all station pairs
    """
    regions = [r for r, s in streams.items() if s is not None]
    n_regions = len(regions)

    # Initialize correlation matrix
    corr_matrix = np.zeros((n_regions, n_regions))
    lag_matrix = np.zeros((n_regions, n_regions))

    print(f"\n{'=' * 60}")
    print("Computing Cross-Correlations")
    print(f"{'=' * 60}\n")

    for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions):
            if i == j:
                corr_matrix[i, j] = 1.0
                lag_matrix[i, j] = 0.0
                continue

            if j < i:  # Matrix is symmetric
                corr_matrix[i, j] = corr_matrix[j, i]
                lag_matrix[i, j] = -lag_matrix[j, i]
                continue

            try:
                tr1 = streams[region1][0]
                tr2 = streams[region2][0]

                # Ensure same sampling rate
                if tr1.stats.sampling_rate != tr2.stats.sampling_rate:
                    tr2_copy = tr2.copy()
                    tr2_copy.resample(tr1.stats.sampling_rate)
                    data2 = tr2_copy.data
                else:
                    data2 = tr2.data

                data1 = tr1.data

                # Trim to same length
                min_len = min(len(data1), len(data2))
                data1 = data1[:min_len]
                data2 = data2[:min_len]

                # Cross-correlation using ObsPy
                cc = correlate(data1, data2, shift=int(min_len / 4))
                shift, max_corr = xcorr_max(cc)

                corr_matrix[i, j] = max_corr
                lag_matrix[i, j] = shift / tr1.stats.sampling_rate  # Convert to seconds

                print(
                    f"   {region1[:15]:15} ↔ {region2[:15]:15}: "
                    f"r={max_corr:.3f}, lag={lag_matrix[i, j]:.1f}s"
                )

            except Exception as e:
                print(f"   Error {region1} ↔ {region2}: {str(e)[:30]}")
                corr_matrix[i, j] = np.nan
                lag_matrix[i, j] = np.nan

    return corr_matrix, lag_matrix, regions


def compute_spectral_coherence(
    raw_streams, processed_streams, freq_min=0.01, freq_max=0.5, nperseg=8192
):
    """
    Compute spectral coherence using RAW data (only detrended)
    This preserves the spectral characteristics needed for coherence estimation

    Parameters:
    -----------
    raw_streams : dict
        Raw (unfiltered) stream data
    processed_streams : dict
        Processed streams (used only to identify valid regions)
    freq_min, freq_max : float
        Frequency range of interest (Hz)
    nperseg : int
        Segment length for Welch's method (larger = better freq resolution)
    """
    # Get regions that have valid data in both raw and processed
    regions = [
        r
        for r in processed_streams.keys()
        if processed_streams[r] is not None and raw_streams.get(r) is not None
    ]

    coherence_results = {}

    print(f"\n{'=' * 60}")
    print("Computing Spectral Coherence (from raw data)")
    print(f"{'=' * 60}\n")

    for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions):
            if j <= i:
                continue

            try:
                # Use RAW streams, only detrend
                tr1 = raw_streams[region1][0].copy()
                tr2 = raw_streams[region2][0].copy()

                # Minimal preprocessing - just remove trend
                tr1.detrend("demean")
                tr1.detrend("linear")
                tr2.detrend("demean")
                tr2.detrend("linear")

                # Resample to common rate if needed
                fs1 = tr1.stats.sampling_rate
                fs2 = tr2.stats.sampling_rate

                if fs1 != fs2:
                    target_fs = min(fs1, fs2)
                    if fs1 > target_fs:
                        tr1.resample(target_fs)
                    if fs2 > target_fs:
                        tr2.resample(target_fs)

                fs = tr1.stats.sampling_rate

                # Get same-length data
                min_len = min(len(tr1.data), len(tr2.data))
                data1 = tr1.data[:min_len]
                data2 = tr2.data[:min_len]

                # Use large nperseg for good low-frequency resolution
                # For 0.01 Hz resolution at 20 Hz sampling, need nperseg >= 2000
                actual_nperseg = min(nperseg, min_len // 8)
                actual_nperseg = max(actual_nperseg, 256)  # Minimum viable

                # Compute coherence with 50% overlap (default)
                f, Cxy = signal.coherence(
                    data1,
                    data2,
                    fs=fs,
                    nperseg=actual_nperseg,
                    noverlap=actual_nperseg // 2,
                )

                # Filter to frequency range of interest
                mask = (f >= freq_min) & (f <= freq_max)
                f_filtered = f[mask]
                Cxy_filtered = Cxy[mask]

                # Calculate statistics in the band of interest
                mean_coh = np.mean(Cxy_filtered) if len(Cxy_filtered) > 0 else 0
                max_coh = np.max(Cxy_filtered) if len(Cxy_filtered) > 0 else 0

                coherence_results[(region1, region2)] = {
                    "frequency": f_filtered,
                    "coherence": Cxy_filtered,
                    "frequency_full": f,
                    "coherence_full": Cxy,
                    "mean_coherence": mean_coh,
                    "max_coherence": max_coh,
                    "nperseg_used": actual_nperseg,
                }

                print(
                    f"   {region1[:15]:15} ↔ {region2[:15]:15}: "
                    f"mean={mean_coh:.3f}, max={max_coh:.3f}"
                )

            except Exception as e:
                print(f"   Error {region1} ↔ {region2}: {str(e)[:40]}")

    return coherence_results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_waveforms(streams, stations_dict, title="Seismic Waveforms"):
    """
    Plot all waveforms for visual comparison
    """
    valid_streams = {k: v for k, v in streams.items() if v is not None}
    n_streams = len(valid_streams)

    if n_streams == 0:
        print("No valid streams to plot!")
        return

    fig, axes = plt.subplots(n_streams, 1, figsize=(14, 2.5 * n_streams), sharex=True)
    if n_streams == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_streams))

    for idx, (region, st) in enumerate(valid_streams.items()):
        tr = st[0]
        times = tr.times()

        axes[idx].plot(times, tr.data, color=colors[idx], linewidth=0.5)
        axes[idx].set_ylabel(region.replace("_", "\n"), fontsize=9)
        axes[idx].set_xlim(times[0], times[-1])
        axes[idx].grid(True, alpha=0.3)

        # Add tectonic setting annotation
        tectonic = stations_dict[region]["tectonic_setting"]
        axes[idx].text(
            0.98,
            0.95,
            tectonic,
            transform=axes[idx].transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("waveforms_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_matrix(corr_matrix, regions, title="Cross-Correlation Matrix"):
    """
    Plot the correlation matrix as a heatmap
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation Coefficient", fontsize=11)

    # Set labels
    short_names = [r.replace("_", "\n") for r in regions]
    ax.set_xticks(range(len(regions)))
    ax.set_yticks(range(len(regions)))
    ax.set_xticklabels(short_names, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(short_names, fontsize=9)

    # Add correlation values as text
    for i in range(len(regions)):
        for j in range(len(regions)):
            if not np.isnan(corr_matrix[i, j]):
                text_color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_spectral_coherence(coherence_results, title="Spectral Coherence Analysis"):
    """
    Plot spectral coherence with proper formatting
    """
    n_pairs = len(coherence_results)
    if n_pairs == 0:
        print("No coherence results to plot!")
        return

    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_pairs))

    for idx, ((r1, r2), data) in enumerate(coherence_results.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        freq = data["frequency"]
        coh = data["coherence"]

        if len(freq) == 0:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"{r1.split('_')[0]} ↔ {r2.split('_')[0]}")
            continue

        # Plot coherence
        ax.plot(freq, coh, color=colors[idx], linewidth=1.2, label="Coherence")
        ax.fill_between(freq, 0, coh, color=colors[idx], alpha=0.3)

        # Add significance threshold line
        ax.axhline(
            y=0.5,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="0.5 threshold",
        )

        # Formatting
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("Coherence", fontsize=9)
        ax.set_title(
            f"{r1.split('_')[0]} ↔ {r2.split('_')[0]}", fontsize=10, fontweight="bold"
        )
        ax.set_xlim([freq.min(), freq.max()])
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, linestyle=":")

        # Add statistics annotation
        mean_coh = data["mean_coherence"]
        max_coh = data["max_coherence"]
        stats_text = f"Mean: {mean_coh:.3f}\nMax: {max_coh:.3f}"
        ax.text(
            0.97,
            0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Highlight frequency bands
        # Teleseismic P-wave: ~0.5-2 Hz (mostly filtered out)
        # Surface waves: 0.01-0.1 Hz
        ax.axvspan(
            0.01, 0.05, alpha=0.1, color="blue", label="Long-period surface waves"
        )
        ax.axvspan(0.05, 0.15, alpha=0.1, color="green", label="Surface waves")

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("spectral_coherence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ Saved: spectral_coherence.png")


def plot_station_map(stations_dict):
    """
    Plot station locations on a world map
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)

        # Plot tectonic plate boundaries (simplified)
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "geography_marine_polys",
                "110m",
                edgecolor="red",
                facecolor="none",
                linewidth=0.5,
            )
        )

        colors = plt.cm.Set1(np.linspace(0, 1, len(stations_dict)))

        for idx, (region, params) in enumerate(stations_dict.items()):
            lat, lon = params["coords"]
            ax.plot(
                lon,
                lat,
                "o",
                markersize=12,
                color=colors[idx],
                transform=ccrs.PlateCarree(),
                markeredgecolor="black",
            )
            ax.text(
                lon + 3,
                lat + 3,
                region.replace("_", "\n"),
                fontsize=8,
                transform=ccrs.PlateCarree(),
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        ax.set_title(
            "Strategic Seismic Stations at Tectonic Plate Boundaries",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig("station_map.png", dpi=150, bbox_inches="tight")
        plt.show()

    except ImportError:
        print("Cartopy not installed. Skipping map visualization.")
        print("Install with: pip install cartopy")


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """
    Main function to run the tectonic correlation analysis
    """
    print("\n" + "=" * 70)
    print("   TECTONIC CORRELATION ANALYSIS USING ObsPy")
    print("   Investigating seismic correlations across plate boundaries")
    print("=" * 70)

    # 7.6-magnitude earthquake in Japan on December 8, 2025, at 11:15 p.m. local time (1415 GMT)
    # Time window for the 2025-12-08 Japan earthquake
    starttime = UTCDateTime("2025-12-08T13:00:00")
    endtime = UTCDateTime("2025-12-08T17:00:00")

    print(f"\n📅 Analysis Period:")
    print(f"   Start: {starttime}")
    print(f"   End:   {endtime}")
    print(f"   Duration: {(endtime - starttime) / 3600:.1f} hours")

    # =================================================================
    # STEP 2: Fetch seismic data
    # =================================================================
    raw_streams = fetch_seismic_data(TECTONIC_STATIONS, starttime, endtime)

    # =================================================================
    # STEP 3: Preprocess data (for correlation and waveform plots)
    # =================================================================
    processed_streams = preprocess_streams(raw_streams, freqmin=0.01, freqmax=0.5)

    # =================================================================
    # STEP 4: Plot station map
    # =================================================================
    print("\n📍 Plotting station locations...")
    plot_station_map(TECTONIC_STATIONS)

    # =================================================================
    # STEP 5: Plot waveforms
    # =================================================================
    print("\n📊 Plotting waveforms...")
    plot_waveforms(
        processed_streams,
        TECTONIC_STATIONS,
        title=f"Filtered Seismic Waveforms\n{starttime.date}",
    )

    # =================================================================
    # STEP 6: Compute cross-correlations (uses processed/filtered data)
    # =================================================================
    corr_matrix, lag_matrix, regions = compute_cross_correlations(
        processed_streams, TECTONIC_STATIONS
    )

    # =================================================================
    # STEP 7: Plot correlation matrix
    # =================================================================
    print("\n📊 Plotting correlation matrix...")
    plot_correlation_matrix(
        corr_matrix, regions, title=f"Cross-Correlation Matrix\n{starttime.date}"
    )

    # =================================================================
    # STEP 8: Compute spectral coherence (uses RAW data)
    # =================================================================
    print("\n📊 Computing spectral coherence from raw data...")
    coherence_results = compute_spectral_coherence(
        raw_streams, processed_streams, freq_min=0.01, freq_max=0.5, nperseg=8192
    )

    # =================================================================
    # STEP 9: Plot spectral coherence
    # =================================================================
    print("\n📊 Plotting spectral coherence...")
    plot_spectral_coherence(
        coherence_results, title=f"Spectral Coherence Analysis\n{starttime.date}"
    )

    # =================================================================
    # STEP 10: Summary statistics
    # =================================================================
    print("\n" + "=" * 70)
    print("   ANALYSIS SUMMARY")
    print("=" * 70)

    print(
        f"\n📊 Stations successfully processed: "
        f"{len([s for s in processed_streams.values() if s])}/{len(TECTONIC_STATIONS)}"
    )

    print(f"\n📈 Correlation Statistics:")
    valid_corr = corr_matrix[~np.isnan(corr_matrix) & (corr_matrix != 1.0)]
    if len(valid_corr) > 0:
        print(f"   • Maximum correlation: {np.max(valid_corr):.3f}")
        print(f"   • Minimum correlation: {np.min(valid_corr):.3f}")
        print(f"   • Mean correlation: {np.mean(valid_corr):.3f}")
        print(f"   • Std deviation: {np.std(valid_corr):.3f}")

    print(f"\n📐 Coherence Statistics:")
    if coherence_results:
        mean_coherences = [d["mean_coherence"] for d in coherence_results.values()]
        max_coherences = [d["max_coherence"] for d in coherence_results.values()]
        print(f"   • Maximum mean coherence: {np.max(mean_coherences):.3f}")
        print(f"   • Maximum peak coherence: {np.max(max_coherences):.3f}")
        print(f"   • Minimum mean coherence: {np.min(mean_coherences):.3f}")

    print("\n" + "=" * 70)
    print("   Analysis complete! Check generated PNG files.")
    print("=" * 70 + "\n")

    return processed_streams, corr_matrix, coherence_results


# =============================================================================
# RUN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    streams, correlations, coherence = main()
