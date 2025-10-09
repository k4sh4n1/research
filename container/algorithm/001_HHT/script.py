import json
import numpy as np
from PyEMD import CEEMDAN
from scipy.signal import find_peaks, welch
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')


def find_dominant_periods(signal, sampling_rate=1.0):
    """
    Find dominant periods in a signal using multiple methods.

    Args:
        signal: Input signal array
        sampling_rate: Sampling rate (default 1.0 for index-based)

    Returns:
        Dictionary containing dominant periods and frequencies
    """
    results = {}

    # Method 1: FFT for frequency domain analysis
    n = len(signal)
    frequencies = fftfreq(n, d=1/sampling_rate)[:n//2]
    fft_values = np.abs(fft(signal))[:n//2]

    # Find peaks in FFT
    peaks, properties = find_peaks(fft_values, height=np.max(fft_values) * 0.1)

    if len(peaks) > 0:
        # Sort by amplitude
        sorted_indices = np.argsort(fft_values[peaks])[::-1]
        dominant_freqs = frequencies[peaks[sorted_indices]]
        dominant_periods = 1 / (dominant_freqs + 1e-10)  # Avoid division by zero

        # Keep only meaningful periods (filter out extremely large values)
        valid_periods = dominant_periods[dominant_periods < len(signal)]
        results['fft_periods'] = valid_periods[:5].tolist()  # Top 5 periods
    else:
        results['fft_periods'] = []

    # Method 2: CEEMDAN for intrinsic mode functions
    try:
        ceemdan = CEEMDAN(trials=50, epsilon=0.005)
        IMFs = ceemdan(signal)

        imf_periods = []
        for i, imf in enumerate(IMFs[:-1]):  # Exclude residual
            # Find zero crossings to estimate period
            zero_crossings = np.where(np.diff(np.sign(imf)))[0]
            if len(zero_crossings) > 1:
                avg_period = 2 * np.mean(np.diff(zero_crossings))
                imf_periods.append(avg_period)

        results['imf_periods'] = sorted(imf_periods)

    except Exception:
        results['imf_periods'] = []

    # Method 3: Autocorrelation for periodicity
    if len(signal) > 10:
        autocorr = np.correlate(signal - np.mean(signal), signal - np.mean(signal), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]

        # Find peaks in autocorrelation
        peaks, _ = find_peaks(autocorr, height=0.3)
        if len(peaks) > 0:
            results['autocorr_periods'] = peaks[:3].tolist()  # First 3 significant lags
        else:
            results['autocorr_periods'] = []
    else:
        results['autocorr_periods'] = []

    return results


def algo(message):
    """
    Identify dominant periods and cycles in the input signal.

    Args:
        message (str): JSON string containing array of prices

    Returns:
        str: JSON string with identified periods and cycles
    """
    try:
        # Parse input
        prices = json.loads(message)

        # Validate
        if not isinstance(prices, list) or len(prices) < 10:
            return json.dumps({
                'status': 'error',
                'message': 'Input must be a list with at least 10 data points',
                'periods': {}
            })

        # Convert to numpy array
        signal = np.array(prices, dtype=np.float64)

        # Detrend signal for better period detection
        detrended = signal - np.polyval(np.polyfit(range(len(signal)), signal, 1), range(len(signal)))

        # Find dominant periods
        periods = find_dominant_periods(detrended)

        # Combine and deduplicate periods
        all_periods = []
        for key in periods:
            all_periods.extend(periods[key])

        # Remove duplicates and sort
        unique_periods = sorted(list(set(np.round(all_periods, 2))))

        # Filter out unrealistic periods
        valid_periods = [p for p in unique_periods if 2 <= p <= len(signal)/2]

        # Calculate dominant frequencies
        dominant_frequencies = [1/p for p in valid_periods if p > 0]

        result = {
            'status': 'success',
            'dominant_periods': valid_periods[:10],  # Top 10 most dominant
            'dominant_frequencies': dominant_frequencies[:10],
            'details': {
                'fft_based': periods.get('fft_periods', [])[:5],
                'imf_based': periods.get('imf_periods', [])[:5],
                'autocorr_based': periods.get('autocorr_periods', [])[:5]
            },
            'input_length': len(signal)
        }

        return json.dumps(result, indent=2)

    except json.JSONDecodeError:
        return json.dumps({
            'status': 'error',
            'message': 'Invalid JSON input',
            'periods': {}
        })
    except Exception as e:
        return json.dumps({
            'status': 'error',
            'message': f'Processing failed: {str(e)}',
            'periods': {}
        })
