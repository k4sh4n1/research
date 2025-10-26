#!/usr/bin/env python
"""
Using ObsPy's built-in real-time plotter (Fixed version)
"""

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import time

# Create client
client = Client("IRIS")

# Configuration
STATION = ("IU", "ANMO", "00", "BHZ")  # Station to monitor
UPDATE_INTERVAL = 10  # seconds
WINDOW_LENGTH = 300  # seconds (5 minutes)

print(f"Starting real-time monitoring of {'.'.join(STATION)}")
print("Press Ctrl+C to stop\n")

plt.ion()  # Interactive mode
fig = plt.figure(figsize=(12, 6))

try:
    while True:
        # Get current time window
        endtime = UTCDateTime() - 30  # 30 sec latency buffer
        starttime = endtime - WINDOW_LENGTH

        try:
            # Fetch latest data
            st = client.get_waveforms(*STATION, starttime=starttime, endtime=endtime)

            # Process
            st.merge(fill_value='interpolate')
            st.detrend('linear')

            # Clear and update plot
            plt.clf()

            # Use ObsPy's built-in plot (creates its own axes)
            st.plot(fig=fig, show=False, block=False)

            # Update title
            fig.suptitle(f"Real-Time: {'.'.join(STATION)} - {UTCDateTime().ctime()}",
                        fontsize=14, y=0.98)

            # Refresh display
            plt.draw()
            plt.pause(0.01)

        except Exception as e:
            print(f"Error fetching data: {e}")
            plt.clf()
            plt.text(0.5, 0.5, f'Waiting for data...\n{e}',
                    transform=fig.transFigure,
                    ha='center', va='center', fontsize=12)
            plt.draw()
            plt.pause(0.01)

        # Wait before next update
        time.sleep(UPDATE_INTERVAL)

except KeyboardInterrupt:
    print("\nStopping real-time monitoring...")
    plt.close()
