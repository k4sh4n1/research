#!/usr/bin/env python
"""
Real-time ground displacement viewer with instrument response removal
"""

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from urllib.error import URLError
import socket

# Configuration
STATION = ("IU", "ANMO", "00", "BHZ")  # Station to monitor (BH = broadband high-gain)
UPDATE_INTERVAL = 10000  # milliseconds
WINDOW_LENGTH = 300  # seconds (5 minutes)
LATENCY_BUFFER = 30  # seconds
MAX_RETRIES = 3  # Number of retries on network failure

# Processing configuration
OUTPUT_UNITS = "DISP"  # "DISP" for displacement, "VEL" for velocity, "ACC" for acceleration
PRE_FILTER = (0.005, 0.01, 8.0, 10.0)  # Pre-filter corners in Hz (for stability)

# Create client with timeout
client = Client("IRIS", timeout=30)

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 6))
line, = ax.plot([], [], 'b-', linewidth=0.7)

# Configure axes
ax.set_xlabel('Time (seconds)', fontsize=11)
ax.set_ylabel('Ground Displacement (meters)', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, WINDOW_LENGTH)

# Track consecutive errors
error_count = 0
last_successful_fetch = None
inventory = None  # Cache the inventory

def init():
    """Initialize animation"""
    line.set_data([], [])
    return line,

def get_inventory_cached(network, station, location, channel, starttime, endtime):
    """Get and cache station inventory"""
    global inventory
    if inventory is None:
        try:
            print("Fetching station inventory (instrument response)...")
            inventory = client.get_stations(
                network=network,
                station=station,
                location=location,
                channel=channel,
                starttime=starttime,
                endtime=endtime,
                level="response"
            )
            print("Inventory fetched successfully")
        except Exception as e:
            print(f"Warning: Could not fetch inventory: {e}")
    return inventory

def fetch_with_retry(station, starttime, endtime, max_retries=MAX_RETRIES):
    """Fetch waveforms with retry logic"""
    for attempt in range(max_retries):
        try:
            st = client.get_waveforms(*station, starttime=starttime, endtime=endtime)
            return st, None
        except URLError as e:
            # Network-related errors
            error_msg = f"Network error (attempt {attempt + 1}/{max_retries}): {str(e.reason)}"
            if attempt < max_retries - 1:
                print(f"{error_msg} - Retrying...")
            else:
                return None, error_msg
        except socket.timeout:
            error_msg = f"Timeout (attempt {attempt + 1}/{max_retries})"
            if attempt < max_retries - 1:
                print(f"{error_msg} - Retrying...")
            else:
                return None, error_msg
        except Exception as e:
            # Handle the specific ObsPy error when it tries to parse URLError
            if "'URLError' object has no attribute" in str(e):
                error_msg = "Network connection failed - Server unreachable"
            else:
                error_msg = f"Error: {type(e).__name__}: {str(e)}"

            if attempt < max_retries - 1:
                print(f"{error_msg} - Retrying...")
            else:
                return None, error_msg

    return None, "Max retries exceeded"

def fetch_and_update(frame):
    """Fetch new data and update plot"""
    global error_count, last_successful_fetch

    # Calculate time window
    endtime = UTCDateTime() - LATENCY_BUFFER
    starttime = endtime - WINDOW_LENGTH

    # Fetch waveform with retry logic
    st, error_msg = fetch_with_retry(STATION, starttime, endtime)

    if st is not None:
        try:
            # Process data
            st.merge(fill_value='interpolate')
            st.detrend('linear')

            # Remove instrument response to get ground displacement
            inv = get_inventory_cached(
                STATION[0], STATION[1], STATION[2], STATION[3],
                starttime, endtime
            )

            if inv is not None:
                try:
                    # Remove response - converts to displacement in meters
                    st.remove_response(
                        inventory=inv,
                        output=OUTPUT_UNITS,
                        pre_filt=PRE_FILTER,
                        water_level=60
                    )
                    units_label = "Ground Displacement (meters)"
                except Exception as e:
                    print(f"Warning: Could not remove response: {e}")
                    print("Displaying raw counts instead")
                    units_label = "Amplitude (counts)"
            else:
                # If no inventory, just detrend and show raw data
                units_label = "Amplitude (counts)"

            if len(st) > 0:
                tr = st[0]

                # Create time vector
                time_vec = np.linspace(0, WINDOW_LENGTH, len(tr.data))

                # Update line data
                line.set_data(time_vec, tr.data)
                line.set_alpha(1.0)  # Full opacity for fresh data

                # Auto-scale y-axis
                if len(tr.data) > 0:
                    data_min, data_max = tr.data.min(), tr.data.max()
                    margin = (data_max - data_min) * 0.1
                    if margin == 0:  # Handle flat signal
                        margin = abs(data_max) * 0.1 if data_max != 0 else 1e-9
                    ax.set_ylim(data_min - margin, data_max + margin)

                # Update labels
                ax.set_ylabel(units_label, fontsize=11)

                # Update title with timestamp
                status_text = "Connected"
                if error_count > 0:
                    status_text = f"Reconnected (after {error_count} errors)"
                    error_count = 0

                # Show if we're displaying displacement or raw data
                data_type = "Displacement" if "meters" in units_label else "Raw Data"

                ax.set_title(
                    f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel} "
                    f"- {data_type} - {status_text} - Last Update: {UTCDateTime().ctime()}",
                    fontsize=12
                )

                last_successful_fetch = UTCDateTime()
            else:
                print("No data in stream")

        except Exception as e:
            print(f"Processing error: {e}")
            ax.set_title(f"Processing error: {e}", fontsize=12)
    else:
        # Handle fetch error
        error_count += 1
        print(f"Fetch failed: {error_msg}")

        # Update title with error info
        time_since_last = ""
        if last_successful_fetch:
            seconds_ago = int(UTCDateTime() - last_successful_fetch)
            time_since_last = f" (last success: {seconds_ago}s ago)"

        ax.set_title(
            f"Connection issues (error #{error_count}){time_since_last} - {error_msg}",
            fontsize=12,
            color='red'
        )

        # Keep existing data on display but dim it
        if line.get_alpha() == 1.0:
            line.set_alpha(0.3)  # Dim the line to show it's stale

    return line,

# Print startup message
print(f"Starting real-time ground displacement monitoring of {'.'.join(STATION)}")
print(f"Output units: {OUTPUT_UNITS}")
print(f"Window length: {WINDOW_LENGTH} seconds")
print(f"Update interval: {UPDATE_INTERVAL/1000} seconds")
print("\nFetching initial data and instrument response...")

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
