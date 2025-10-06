import json
import numpy as np
from PyEMD import CEEMDAN
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def algo(message):
    """
    Main algorithm function optimized for MQL5 integration.

    Args:
        message (str): JSON string containing input data (array of prices)

    Returns:
        str: JSON string containing analysis results formatted for MQL5
    """
    try:
        # Parse input JSON
        prices = json.loads(message)

        # Validate input
        if not isinstance(prices, list) or len(prices) < 10:
            return json.dumps({
                'status': 'error',
                'error_code': -1,
                'error_type': 'ValidationError',
                'message': 'Input must be a list with at least 10 price points',
                'buffers': {
                    'trend_line': [],
                    'upper_band': [],
                    'lower_band': [],
                    'oscillator': [],
                    'signal_line': [],
                    'imf_high_freq': [],
                    'imf_mid_freq': [],
                    'imf_low_freq': []
                }
            })

        # Convert to numpy array
        signal = np.array(prices, dtype=np.float64)
        signal_length = len(signal)

        # Perform CEEMDAN decomposition
        ceemdan = CEEMDAN(trials=100, epsilon=0.005, ext_EMD=None)
        IMFs = ceemdan(signal)

        # Calculate signal statistics
        signal_mean = float(np.mean(signal))
        signal_std = float(np.std(signal))

        # Initialize buffers with proper length
        trend_buffer = np.zeros(signal_length)
        upper_band = np.zeros(signal_length)
        lower_band = np.zeros(signal_length)
        oscillator = np.zeros(signal_length)
        signal_line = np.zeros(signal_length)
        imf_high = np.zeros(signal_length)
        imf_mid = np.zeros(signal_length)
        imf_low = np.zeros(signal_length)

        # Process IMFs
        if len(IMFs) > 0:
            # Trend is typically the last IMF (low frequency)
            trend_buffer = IMFs[-1]

            # Calculate bands based on trend and standard deviation
            band_width = signal_std * 1.5
            upper_band = trend_buffer + band_width
            lower_band = trend_buffer - band_width

            # High frequency component (first IMF if exists)
            if len(IMFs) >= 1:
                imf_high = IMFs[0]

            # Mid frequency component
            if len(IMFs) >= 3:
                imf_mid = IMFs[len(IMFs)//2]

            # Low frequency component (second to last if exists)
            if len(IMFs) >= 2:
                imf_low = IMFs[-2]

            # Calculate oscillator from high frequency components
            if len(IMFs) >= 2:
                # Sum of first few IMFs normalized
                oscillator = np.sum(IMFs[0:min(3, len(IMFs)-1)], axis=0)
                oscillator = (oscillator - np.mean(oscillator)) / (np.std(oscillator) + 1e-10)

            # Signal line (moving average of oscillator)
            window = min(5, signal_length // 4)
            for i in range(window, signal_length):
                signal_line[i] = np.mean(oscillator[i-window:i])
        else:
            # Fallback if CEEMDAN fails
            trend_buffer = signal.copy()
            upper_band = signal + signal_std
            lower_band = signal - signal_std

        # Detect peaks and troughs for signals
        peaks, peak_props = find_peaks(signal, height=signal_mean)
        troughs, trough_props = find_peaks(-signal, height=-signal_mean)

        # Determine current market conditions
        trend_direction = 'neutral'
        trend_strength = 0.0

        if len(trend_buffer) > 0:
            trend_change = trend_buffer[-1] - trend_buffer[0]
            trend_strength = abs(trend_change) / (signal_std + 1e-10)

            if trend_change > signal_std * 0.5:
                trend_direction = 'bullish'
            elif trend_change < -signal_std * 0.5:
                trend_direction = 'bearish'

        # Generate trading signals
        latest_signal = 'hold'
        signal_strength = 0.0

        # Check for buy signal
        if len(oscillator) > 1:
            if oscillator[-1] < -1.0 and oscillator[-1] > oscillator[-2]:
                latest_signal = 'buy'
                signal_strength = min(abs(oscillator[-1]), 2.0) / 2.0
            elif oscillator[-1] > 1.0 and oscillator[-1] < oscillator[-2]:
                latest_signal = 'sell'
                signal_strength = min(abs(oscillator[-1]), 2.0) / 2.0

        # Determine market volatility
        volatility_ratio = signal_std / (signal_mean + 1e-10)
        if volatility_ratio > 0.1:
            volatility = 'high'
        elif volatility_ratio > 0.05:
            volatility = 'medium'
        else:
            volatility = 'low'

        # Support and resistance levels
        support_level = float(np.min(signal[-20:])) if signal_length >= 20 else float(np.min(signal))
        resistance_level = float(np.max(signal[-20:])) if signal_length >= 20 else float(np.max(signal))

        # Convert numpy arrays to lists for JSON serialization
        # Ensure all buffers have the same length
        result = {
            'status': 'success',
            'error_code': 0,
            'algorithm': 'HHT_CEEMDAN',
            'input_length': signal_length,
            'num_imfs': len(IMFs),

            # Current values for display
            'current': {
                'price': float(signal[-1]),
                'trend_value': float(trend_buffer[-1]),
                'oscillator_value': float(oscillator[-1]) if len(oscillator) > 0 else 0.0,
                'signal_line_value': float(signal_line[-1]) if len(signal_line) > 0 else 0.0,
                'trend_direction': trend_direction,
                'trend_strength': float(trend_strength),
                'volatility': volatility,
                'signal': latest_signal,
                'signal_strength': float(signal_strength)
            },

            # Key levels
            'levels': {
                'support': support_level,
                'resistance': resistance_level,
                'mean': float(signal_mean),
                'upper_band': float(upper_band[-1]) if len(upper_band) > 0 else 0.0,
                'lower_band': float(lower_band[-1]) if len(lower_band) > 0 else 0.0
            },

            # Statistics
            'statistics': {
                'mean': float(signal_mean),
                'std': float(signal_std),
                'min': float(np.min(signal)),
                'max': float(np.max(signal)),
                'kurtosis': float(kurtosis(signal)),
                'skewness': float(skew(signal))
            },

            # Buffer arrays for plotting
            'buffers': {
                'trend_line': trend_buffer.tolist(),
                'upper_band': upper_band.tolist(),
                'lower_band': lower_band.tolist(),
                'oscillator': oscillator.tolist(),
                'signal_line': signal_line.tolist(),
                'imf_high_freq': imf_high.tolist(),
                'imf_mid_freq': imf_mid.tolist(),
                'imf_low_freq': imf_low.tolist()
            },

            # Signal points for markers
            'signals': {
                'buy_points': peaks.tolist() if len(peaks) > 0 else [],
                'sell_points': troughs.tolist() if len(troughs) > 0 else []
            }
        }

        return json.dumps(result, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({
            'status': 'error',
            'error_code': -2,
            'error_type': 'JSONDecodeError',
            'message': f'Invalid JSON input: {str(e)}',
            'buffers': {
                'trend_line': [],
                'upper_band': [],
                'lower_band': [],
                'oscillator': [],
                'signal_line': [],
                'imf_high_freq': [],
                'imf_mid_freq': [],
                'imf_low_freq': []
            }
        })

    except Exception as e:
        return json.dumps({
            'status': 'error',
            'error_code': -99,
            'error_type': 'ProcessingError',
            'message': f'Algorithm processing failed: {str(e)}',
            'buffers': {
                'trend_line': [],
                'upper_band': [],
                'lower_band': [],
                'oscillator': [],
                'signal_line': [],
                'imf_high_freq': [],
                'imf_mid_freq': [],
                'imf_low_freq': []
            }
        })
