#!/usr/bin/env python
"""
Minimalist ObsPy script to fetch and visualize real-time seismic data
from various seismology data centers.
"""

import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def fetch_recent_earthquake_data():
    """
    Fetch recent earthquake catalog data from USGS
    """
    print("=" * 60)
    print("Fetching recent earthquake catalog from USGS...")
    print("=" * 60)

    try:
        # Initialize client for USGS data center
        client = Client("USGS")

        # Get earthquakes from the last 7 days with magnitude > 4.5
        endtime = UTCDateTime()
        starttime = endtime - 7 * 24 * 60 * 60  # 7 days ago

        catalog = client.get_events(
            starttime=starttime,
            endtime=endtime,
            minmagnitude=4.5,
            orderby="time"
        )

        print(f"\nFound {len(catalog)} earthquakes in the last 7 days (M > 4.5):")
        print("-" * 60)

        for i, event in enumerate(catalog[:5], 1):  # Show first 5 events
            origin = event.origins[0]
            magnitude = event.magnitudes[0] if event.magnitudes else None

            print(f"\nEvent {i}:")
            print(f"  Time: {origin.time}")
            print(f"  Location: Lat={origin.latitude:.2f}°, Lon={origin.longitude:.2f}°")
            print(f"  Depth: {origin.depth/1000:.1f} km")
            if magnitude:
                print(f"  Magnitude: {magnitude.mag:.1f} {magnitude.magnitude_type}")

        return catalog

    except Exception as e:
        print(f"Error fetching earthquake data: {e}")
        return None

def fetch_realtime_waveform_data():
    """
    Fetch real-time seismic waveform data from IRIS
    """
    print("\n" + "=" * 60)
    print("Fetching real-time waveform data from IRIS...")
    print("=" * 60)

    try:
        # Initialize client for IRIS data center
        client = Client("IRIS")

        # Fetch data from a well-known station
        # Using IU network (Global Seismograph Network)
        network = "IU"
        station = "ANMO"  # Albuquerque, New Mexico
        location = "00"
        channel = "BHZ"  # Broadband High-gain Vertical component

        # Get last hour of data
        endtime = UTCDateTime()
        starttime = endtime - 3600  # 1 hour ago

        print(f"\nFetching data for station {network}.{station}.{location}.{channel}")
        print(f"Time range: {starttime} to {endtime}")

        # Fetch the waveform data
        st = client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime
        )

        print(f"\nSuccessfully fetched {len(st)} trace(s)")

        # Print basic statistics
        for tr in st:
            print(f"\nTrace statistics:")
            print(f"  Network.Station.Location.Channel: {tr.id}")
            print(f"  Sampling rate: {tr.stats.sampling_rate} Hz")
            print(f"  Number of samples: {tr.stats.npts}")
            print(f"  Start time: {tr.stats.starttime}")
            print(f"  End time: {tr.stats.endtime}")
            print(f"  Data range: [{tr.data.min():.2f}, {tr.data.max():.2f}]")

        return st

    except Exception as e:
        print(f"Error fetching waveform data: {e}")
        return None

def visualize_waveforms(stream):
    """
    Create visualizations of the seismic waveform data
    """
    if not stream:
        print("No data to visualize")
        return

    print("\n" + "=" * 60)
    print("Creating visualizations...")
    print("=" * 60)

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Get the trace for direct plotting
    tr = stream[0].copy()
    times = np.arange(0, tr.stats.npts) / tr.stats.sampling_rate

    # Plot 1: Raw waveform
    axes[0].plot(times, tr.data, 'b-', linewidth=0.5)
    axes[0].set_title(f'Raw Seismic Waveform - {tr.id}', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude\n(counts)')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([times[0], times[-1]])

    # Add statistics text
    stats_text = f'Sampling Rate: {tr.stats.sampling_rate} Hz | Samples: {tr.stats.npts} | Duration: {tr.stats.npts/tr.stats.sampling_rate:.1f} s'
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Filtered waveform (bandpass filter)
    tr_filtered = tr.copy()
    tr_filtered.filter('bandpass', freqmin=0.1, freqmax=1.0)
    axes[1].plot(times, tr_filtered.data, 'g-', linewidth=0.5)
    axes[1].set_title('Filtered Waveform (0.1-1.0 Hz Bandpass)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude\n(counts)')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([times[0], times[-1]])

    # Plot 3: High-frequency filtered waveform
    tr_highfreq = tr.copy()
    tr_highfreq.filter('highpass', freq=1.0)
    axes[2].plot(times, tr_highfreq.data, 'r-', linewidth=0.5)
    axes[2].set_title('High-Frequency Content (>1.0 Hz)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Amplitude\n(counts)')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([times[0], times[-1]])

    # Plot 4: Spectrogram
    # Use matplotlib's specgram for more control
    axes[3].specgram(tr.data, Fs=tr.stats.sampling_rate, cmap='viridis',
                     NFFT=256, noverlap=128, scale='dB')
    axes[3].set_title('Spectrogram (Power Spectral Density)', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Frequency (Hz)')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].set_ylim([0, tr.stats.sampling_rate/2])

    # Add colorbar for spectrogram
    cbar = plt.colorbar(axes[3].images[0], ax=axes[3], pad=0.01)
    cbar.set_label('Power (dB)', rotation=270, labelpad=15)

    # Overall title
    start_time = tr.stats.starttime.strftime("%Y-%m-%d %H:%M:%S UTC")
    plt.suptitle(f'Seismic Data Visualization - Station: {tr.stats.station}\n{start_time}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save the figure
    output_file = 'seismic_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

    # Also create a simple ObsPy plot
    print("\nCreating ObsPy native plot...")
    stream.plot(outfile='obspy_plot.png', size=(1200, 600))
    print("ObsPy plot saved to: obspy_plot.png")

    plt.show()

def test_station_availability():
    """
    Test availability of different seismic stations from various networks
    """
    print("\n" + "=" * 60)
    print("Testing station availability from multiple data centers...")
    print("=" * 60)

    # List of data centers to test
    data_centers = ["IRIS", "USGS", "ETH", "GFZ"]

    for dc in data_centers:
        try:
            print(f"\n{dc} Data Center:")
            print("-" * 30)
            client = Client(dc)

            # Get station inventory
            inventory = client.get_stations(
                network="*",
                station="*",
                maxlat=50,
                minlat=40,
                maxlon=-100,
                minlon=-110,
                level="station",
                starttime=UTCDateTime() - 86400,
                endtime=UTCDateTime()
            )

            station_count = sum(len(net) for net in inventory)
            print(f"  Found {len(inventory)} networks with {station_count} stations")

            # Show first few stations
            for net in inventory[:2]:
                for sta in net[:3]:
                    print(f"    {net.code}.{sta.code}: {sta.latitude:.2f}°N, {sta.longitude:.2f}°E")

        except Exception as e:
            print(f"  Error: {e}")

def main():
    """
    Main function to demonstrate ObsPy capabilities
    """
    print("\n")
    print("*" * 60)
    print("ObsPy Real-Time Seismic Data Fetcher and Visualizer")
    print("*" * 60)

    # Display ObsPy version
    print(f"\nObsPy Version: {obspy.__version__}")

    # Test 1: Fetch recent earthquake catalog
    catalog = fetch_recent_earthquake_data()

    # Test 2: Fetch real-time waveform data
    stream = fetch_realtime_waveform_data()

    # Test 3: Visualize the waveforms
    if stream:
        visualize_waveforms(stream)

    # Test 4: Check station availability
    test_station_availability()

    print("\n" + "=" * 60)
    print("Script completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
