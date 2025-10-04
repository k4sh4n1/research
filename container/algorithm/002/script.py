import json
import numpy as np
from scipy.fft import fft, fftfreq

import sys
from pathlib import Path
# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def algo(message):
    """
    Simple FFT analysis algorithm.

    Args:
        message (str): JSON string containing a list of numerical values

    Returns:
        str: JSON string with FFT analysis results
    """
    try:
        # Parse input
        data = json.loads(message)

        if not isinstance(data, list) or len(data) < 2:
            return json.dumps({
                'status': 'error',
                'error_type': 'ValidationError',
                'message': 'Input must be a list with at least 2 values'
            }, indent=2)

        # Convert to numpy array
        signal = np.array(data, dtype=np.float64)

        # Perform FFT
        fft_values = fft(signal)
        frequencies = fftfreq(len(signal))

        # Get magnitude spectrum
        magnitude = np.abs(fft_values)

        # Find dominant frequencies
        dominant_freq_indices = np.argsort(magnitude)[-5:][::-1]  # Top 5
        dominant_frequencies = [
            {
                'frequency': float(frequencies[i]),
                'magnitude': float(magnitude[i]),
                'phase': float(np.angle(fft_values[i]))
            }
            for i in dominant_freq_indices
        ]

        # Calculate power spectrum
        power_spectrum = magnitude ** 2
        total_power = float(np.sum(power_spectrum))

        result = {
            'status': 'success',
            'algorithm': 'FFT',
            'input_length': len(data),
            'dominant_frequencies': dominant_frequencies,
            'total_power': total_power,
            'dc_component': float(fft_values[0].real),
            'statistics': {
                'mean': float(np.mean(signal)),
                'std': float(np.std(signal)),
                'max_frequency_magnitude': float(np.max(magnitude))
            }
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            'status': 'error',
            'error_type': 'ProcessingError',
            'message': str(e)
        }, indent=2)
