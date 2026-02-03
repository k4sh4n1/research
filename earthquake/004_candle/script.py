"""
Earthquake Waveform to Candlestick Converter
Fetches seismic data and displays it as financial-style candle charts.
"""

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def fetch_earthquake_waveform(client_name, network, station, location, channel,
                               starttime, endtime):
    """Fetch waveform data from a seismic network."""
    client = Client(client_name)
    try:
        stream = client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime
        )
        return stream
    except Exception as e:
        print(f"Error fetching from {client_name}/{network}.{station}: {e}")
        return None


def waveform_to_candles(trace, candle_minutes=1):
    """Convert seismic trace to OHLC candlestick data."""
    data = trace.data
    sampling_rate = trace.stats.sampling_rate
    samples_per_candle = int(candle_minutes * 60 * sampling_rate)
    
    num_candles = len(data) // samples_per_candle
    candles = []
    
    for i in range(num_candles):
        start_idx = i * samples_per_candle
        end_idx = start_idx + samples_per_candle
        segment = data[start_idx:end_idx]
        
        candles.append({
            'minute': i,
            'open': segment[0],
            'high': np.max(segment),
            'low': np.min(segment),
            'close': segment[-1]
        })
    
    return candles


def plot_candlestick_chart(candles, title, ax=None):
    """Plot candlestick chart from OHLC data."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    for candle in candles:
        x = candle['minute']
        o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
        
        # Bullish (close > open) = green, Bearish = red
        color = 'green' if c >= o else 'red'
        
        # Draw wick (high-low line)
        ax.plot([x, x], [l, h], color=color, linewidth=0.8)
        
        # Draw body (open-close rectangle)
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height == 0:
            body_height = (h - l) * 0.01  # Minimum visible height
        
        rect = Rectangle(
            (x - 0.35, body_bottom),
            0.7, body_height,
            facecolor=color, edgecolor=color, alpha=0.8
        )
        ax.add_patch(rect)
    
    ax.set_xlabel('Time (minutes from start)')
    ax.set_ylabel('Amplitude (counts)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.autoscale()
    
    return ax


def main():
    # Event: M7.6 Japan Earthquake - December 8, 2025, 14:15 UTC
    event_time = UTCDateTime("2025-12-08T14:15:00")
    starttime = event_time - 60  # 1 minute before
    endtime = event_time + 20 * 60  # 20 minutes after
    
    # Station configurations to try (Japanese and global networks)
    stations = [
        ("IRIS", "IU", "MAJO", "00", "BHZ"),  # Matsushiro, Japan
        ("IRIS", "II", "ERM", "00", "BHZ"),   # Erimo, Japan
        ("IRIS", "IU", "INCN", "00", "BHZ"),  # Incheon, South Korea
        ("IRIS", "G", "INU", "00", "BHZ"),    # Inuyama, Japan
    ]
    
    print("=" * 60)
    print("Earthquake Waveform to Candlestick Converter")
    print("=" * 60)
    print(f"Event: M7.6 Japan Earthquake")
    print(f"Date: December 8, 2025, 14:15 UTC")
    print(f"Location: Off coast of Aomori Prefecture, Japan")
    print(f"Time window: {starttime} to {endtime}")
    print("=" * 60)
    
    streams = []
    
    for client_name, net, sta, loc, cha in stations:
        print(f"\nFetching: {net}.{sta}.{loc}.{cha} from {client_name}...")
        stream = fetch_earthquake_waveform(
            client_name, net, sta, loc, cha, starttime, endtime
        )
        if stream:
            streams.append((f"{net}.{sta}.{cha}", stream))
            print(f"  ✓ Success: {len(stream)} trace(s), "
                  f"{stream[0].stats.npts} samples")
    
    if not streams:
        print("\nNo data retrieved. Network may be unavailable.")
        return
    
    # Create candlestick charts
    num_plots = len(streams)
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 5 * num_plots))
    if num_plots == 1:
        axes = [axes]
    
    for idx, (name, stream) in enumerate(streams):
        trace = stream[0]
        trace.detrend('demean')  # Remove mean
        trace.filter('bandpass', freqmin=0.01, freqmax=1.0)  # Filter noise
        
        print(f"\nProcessing {name}...")
        candles = waveform_to_candles(trace, candle_minutes=1)
        print(f"  Generated {len(candles)} candles (1-minute bars)")
        
        title = (f"{name} - M7.6 Japan EQ (2025-12-08)\n"
                f"1-Minute Candlesticks | Sampling: {trace.stats.sampling_rate} Hz")
        plot_candlestick_chart(candles, title, axes[idx])
    
    plt.tight_layout()
    plt.savefig('earthquake_candlesticks.png', dpi=150, bbox_inches='tight')
    print("\n✓ Chart saved to 'earthquake_candlesticks.png'")
    plt.show()


if __name__ == "__main__":
    main()

