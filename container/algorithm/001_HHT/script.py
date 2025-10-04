import json
import numpy as np
from PyEMD import CEEMDAN
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def algo(message):
    """
    Main algorithm function that must be present in every script.py

    Args:
        message (str): JSON string containing input data

    Returns:
        str: JSON string containing analysis results
    """
    try:
        # Parse input JSON
        prices = json.loads(message)

        # Validate input
        if not isinstance(prices, list) or len(prices) < 10:
            return json.dumps({
                'status': 'error',
                'error_type': 'ValidationError',
                'message': 'Input must be a list with at least 10 price points'
            }, indent=2)

        # Convert to numpy array
        signal = np.array(prices, dtype=np.float64)

        # Perform CEEMDAN decomposition
        ceemdan = CEEMDAN(trials=100, epsilon=0.005, ext_EMD=None)
        IMFs = ceemdan(signal)

        # Analyze each IMF
        imf_stats = []
        for i, imf in enumerate(IMFs):
            # Calculate statistics for each IMF
            imf_mean = float(np.mean(imf))
            imf_std = float(np.std(imf))
            imf_energy = float(np.sum(imf**2))
            imf_kurtosis = float(kurtosis(imf))
            imf_skewness = float(skew(imf))

            # Find peaks and troughs
            peaks = []
            troughs = []
            for j in range(1, len(imf) - 1):
                if imf[j] > imf[j-1] and imf[j] > imf[j+1]:
                    peaks.append({'index': j, 'value': float(imf[j])})
                elif imf[j] < imf[j-1] and imf[j] < imf[j+1]:
                    troughs.append({'index': j, 'value': float(imf[j])})

            imf_stats.append({
                'imf_index': i,
                'mean': imf_mean,
                'std': imf_std,
                'energy': imf_energy,
                'kurtosis': imf_kurtosis,
                'skewness': imf_skewness,
                'num_peaks': len(peaks),
                'num_troughs': len(troughs),
                'peaks': peaks[:5],  # Top 5 peaks
                'troughs': troughs[:5],  # Top 5 troughs
                'values': imf.tolist()  # Full IMF values
            })

        # Calculate trend (last IMF typically represents trend)
        trend = IMFs[-1] if len(IMFs) > 0 else signal
        trend_direction = 'upward' if trend[-1] > trend[0] else 'downward'
        trend_strength = float(abs(trend[-1] - trend[0]) / np.mean(np.abs(signal)))

        # Normalize the signal for pattern detection
        scaler = StandardScaler()
        normalized_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

        # Detect potential patterns
        patterns = []
        window_size = min(20, len(signal) // 3)
        for i in range(len(normalized_signal) - window_size):
            window = normalized_signal[i:i+window_size]
            window_std = np.std(window)
            window_mean = np.mean(window)

            # Check for various patterns
            if window_std < 0.1:
                patterns.append({
                    'type': 'consolidation',
                    'start_index': i,
                    'end_index': i + window_size,
                    'confidence': float(1 - window_std)
                })
            elif np.all(np.diff(window) > 0):
                patterns.append({
                    'type': 'strong_uptrend',
                    'start_index': i,
                    'end_index': i + window_size,
                    'confidence': float(np.mean(np.diff(window)))
                })
            elif np.all(np.diff(window) < 0):
                patterns.append({
                    'type': 'strong_downtrend',
                    'start_index': i,
                    'end_index': i + window_size,
                    'confidence': float(abs(np.mean(np.diff(window))))
                })

        # Remove duplicate/overlapping patterns
        unique_patterns = []
        for pattern in patterns:
            if not any(p['start_index'] == pattern['start_index']
                      and p['type'] == pattern['type']
                      for p in unique_patterns):
                unique_patterns.append(pattern)

        # Calculate overall signal statistics
        signal_stats = {
            'mean': float(np.mean(signal)),
            'std': float(np.std(signal)),
            'min': float(np.min(signal)),
            'max': float(np.max(signal)),
            'range': float(np.max(signal) - np.min(signal)),
            'kurtosis': float(kurtosis(signal)),
            'skewness': float(skew(signal))
        }

        # Compile results
        result = {
            'status': 'success',
            'algorithm': 'HHT_CEEMDAN',
            'input_length': len(prices),
            'signal_statistics': signal_stats,
            'num_imfs': len(IMFs),
            'imf_analysis': imf_stats[:5],  # Limit to first 5 IMFs for brevity
            'trend': {
                'direction': trend_direction,
                'strength': trend_strength,
                'start_value': float(trend[0]),
                'end_value': float(trend[-1]),
                'values': trend.tolist()
            },
            'patterns_detected': unique_patterns[:10],  # Top 10 patterns
            'recommendation': {
                'signal_quality': 'good' if len(IMFs) > 3 else 'poor',
                'trend_confidence': 'high' if trend_strength > 0.1 else 'low',
                'volatility': 'high' if signal_stats['std'] / signal_stats['mean'] > 0.1 else 'low'
            }
        }

        return json.dumps(result, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({
            'status': 'error',
            'error_type': 'JSONDecodeError',
            'message': f'Invalid JSON input: {str(e)}'
        }, indent=2)

    except Exception as e:
        return json.dumps({
            'status': 'error',
            'error_type': 'ProcessingError',
            'message': f'Algorithm processing failed: {str(e)}'
        }, indent=2)
