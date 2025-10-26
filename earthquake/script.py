#!/usr/bin/env python
"""
Real-time seismic waveform viewer with proper matplotlib integration
"""

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Configuration
STATION = ("IU", "ANMO", "00", "BHZ")  # Station to monitor
UPDATE_INTERVAL = 10000  # milliseconds
WINDOW_LENGTH = 300  # seconds (5 minutes)
LATENCY_BUFFER = 30  # seconds

# Create client
client = Client("IRIS")

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 6))
line, = ax.plot([], [], 'b-', linewidth=0.7)

# Configure axes
ax.set_xlabel('Time (seconds)', fontsize=11)
ax.set_ylabel('Amplitude (counts)', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, WINDOW_LENGTH)

def init():
    """Initialize animation"""
    line.set_data([], [])
    return line,

def fetch_and_update(frame):
    """Fetch new data and update plot"""
    # Calculate time window
    endtime = UTCDateTime() - LATENCY_BUFFER
    starttime = endtime - WINDOW_LENGTH

    try:
        # Fetch waveform
        st = client.get_waveforms(*STATION, starttime=starttime, endtime=endtime)

        # Process data
        st.merge(fill_value='interpolate')
        st.detrend('linear')

        if len(st) > 0:
            tr = st[0]

            # Create time vector
            time_vec = np.linspace(0, WINDOW_LENGTH, len(tr.data))

            # Update line data
            line.set_data(time_vec, tr.data)

            # Auto-scale y-axis
            if len(tr.data) > 0:
                data_min, data_max = tr.data.min(), tr.data.max()
                margin = (data_max - data_min) * 0.1
                ax.set_ylim(data_min - margin, data_max + margin)

            # Update title with timestamp
            ax.set_title(
                f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel} "
                f"- Real-time - Last Update: {UTCDateTime().ctime()}",
                fontsize=12
            )
        else:
            print("No data received")

    except Exception as e:
        print(f"Error: {e}")
        ax.set_title(f"Waiting for data... ({e})", fontsize=12)
        line.set_data([], [])

    return line,

# Print startup message
print(f"Starting real-time monitoring of {'.'.join(STATION)}")
print(f"Window length: {WINDOW_LENGTH} seconds")
print(f"Update interval: {UPDATE_INTERVAL/1000} seconds")
print("\nFetching initial data...")

# Create animation
ani = animation.FuncAnimation(
    fig,
    fetch_and_update,
    init_func=init,
    interval=UPDATE_INTERVAL,
    blit=True,
    cache_frame_data=False
)

# Show plot
plt.tight_layout()
plt.show()
