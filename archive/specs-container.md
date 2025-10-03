I have the following folder structure.

```
├───container/
    │
    └───algorithm/
        │
        └───001_HHT
        │   │   code.mq5
        │   │   README.md
        │   │   requirements.txt
        │   │   run.bat
        │   │   script.py
        │   │
        │   └───virtual_env/
        │
        └───002_FFT
        │   │   code.mq5
        │   │   README.md
        │   │   requirements.txt
        │   │   run.bat
        │   │   script.py
        │   │
        │   └───virtual_env/
        │
        └───003_mass_spring
            │   code.mq5
            │   README.md
            │   requirements.txt
            │   run.bat
            │   script.py
            │
            └───virtual_env/
```

Inside the `container` folder, I intend to create these:

* A Podman `ContainerFile` based on a minimal Python container image
* A Python script to launch a HTTP server inside the container
   * Receives JSON data by HTTP requests
   * According to the URL and parameters of the request:
      * For a request with this route `001`:
         * The server will run the Python function inside `algorithm/001_HHT`
         * The Python script returns a JSON
         * The server will respond with the JSON returned by the Python function
      * For a request with this route `002`:
         * The server will run the Python function inside `algorithm/002_FFT`
         * The Python script returns a JSON
         * The server will respond with the JSON returned by the Python function
      * For a request with this route `003`:
         * The server will run the Python function inside `algorithm/003_mass_spring`
         * The Python script returns a JSON
         * The server will respond with the JSON returned by the Python function

The server has to be able to handle asynchronous and concurrent HTTP requests and respond with returned JSON data.

The Python function for each algorithm has a logic like this:

```python
import json
import numpy as np
from PyEMD import CEEMDAN
from scipy import stats
from sklearn.preprocessing import StandardScaler

def algo(message):
    try:
        # Parse JSON message
        prices = json.loads(message)
        
        # Validate input
        if not prices:
            raise ValueError("Empty price array")
        
        if not isinstance(prices, (list, tuple)):
            raise TypeError("Prices must be a list or tuple")
        
        # Convert to numpy array and validate numeric types
        prices_array = np.array(prices, dtype=np.float64)
        
        # Check for NaN or infinite values
        if np.any(np.isnan(prices_array)) or np.any(np.isinf(prices_array)):
            raise ValueError("Prices contain NaN or infinite values")
        
        if len(prices_array) < 10:  # CEEMDAN needs sufficient data points
            raise ValueError(f"Insufficient data points ({len(prices_array)}). Need at least 10.")
        
        # Normalize prices for better decomposition (preserve original for analysis)
        scaler = StandardScaler()
        normalized_prices = scaler.fit_transform(prices_array.reshape(-1, 1)).flatten()
        
        # Configure CEEMDAN with optimized parameters
        ceemdan = CEEMDAN(
            trials=100,           # Number of trials for noise-assisted data analysis
            epsilon=0.005,        # Standard deviation of Gaussian noise
            ext_EMD=None,         # Use default EMD
            parallel=True,        # Enable parallel processing for speed
            noise_seed=42         # For reproducibility
        )
        
        # Perform decomposition on normalized data
        IMFs = ceemdan.ceemdan(normalized_prices)
        
        # Denormalize IMFs back to original scale
        IMFs_denormalized = IMFs * scaler.scale_[0] + scaler.mean_[0]
        
        # Calculate trend using linear regression on the residual (last IMF)
        trend_imf = IMFs_denormalized[-1]
        x = np.arange(len(trend_imf))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, trend_imf)
        
        # Determine trend strength
        trend_strength = abs(r_value)  # R-value indicates how linear the trend is
        if trend_strength < 0.3:
            trend_category = 'weak'
        elif trend_strength < 0.7:
            trend_category = 'moderate'
        else:
            trend_category = 'strong'
        
        # Calculate additional metrics
        volatility_components = []
        energy_distribution = []
        
        for i, imf in enumerate(IMFs_denormalized[:-1]):  # Exclude trend component
            volatility = float(np.std(imf))
            energy = float(np.sum(imf**2))
            volatility_components.append(volatility)
            energy_distribution.append(energy)
        
        # Normalize energy distribution to percentages
        total_energy = sum(energy_distribution)
        if total_energy > 0:
            energy_distribution = [e/total_energy * 100 for e in energy_distribution]
        
        # Identify dominant frequency component (IMF with highest energy)
        if energy_distribution:
            dominant_imf_index = np.argmax(energy_distribution)
            dominant_frequency_info = {
                'index': int(dominant_imf_index),
                'energy_percentage': float(energy_distribution[dominant_imf_index]),
                'volatility': float(volatility_components[dominant_imf_index])
            }
        else:
            dominant_frequency_info = None
        
        # Calculate instantaneous metrics from first IMF (highest frequency)
        if len(IMFs_denormalized) > 1:
            first_imf = IMFs_denormalized[0]
            # Zero-crossing rate (indicates frequency)
            zero_crossings = np.where(np.diff(np.sign(first_imf)))[0]
            zero_crossing_rate = len(zero_crossings) / len(first_imf)
            
            # Peak detection for cycle analysis
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(first_imf)
            avg_peak_distance = np.mean(np.diff(peaks)) if len(peaks) > 1 else 0
        else:
            zero_crossing_rate = 0
            avg_peak_distance = 0
        
        # Price change statistics
        price_change = float(prices_array[-1] - prices_array[0])
        price_change_pct = float((prices_array[-1] / prices_array[0] - 1) * 100)
        
        # Construct comprehensive result
        result = {
            'status': 'success',
            'input_stats': {
                'num_points': len(prices_array),
                'min_price': float(np.min(prices_array)),
                'max_price': float(np.max(prices_array)),
                'mean_price': float(np.mean(prices_array)),
                'std_dev': float(np.std(prices_array)),
                'price_change': price_change,
                'price_change_pct': price_change_pct
            },
            'decomposition': {
                'num_imfs': len(IMFs_denormalized),
                'imfs': IMFs_denormalized.tolist()
            },
            'trend_analysis': {
                'direction': 'up' if slope > 0 else 'down',
                'strength': trend_category,
                'slope': float(slope),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'trend_line': {
                    'slope': float(slope),
                    'intercept': float(intercept)
                }
            },
            'frequency_analysis': {
                'volatility_components': volatility_components,
                'energy_distribution_pct': energy_distribution,
                'dominant_component': dominant_frequency_info,
                'high_freq_metrics': {
                    'zero_crossing_rate': float(zero_crossing_rate),
                    'avg_cycle_length': float(avg_peak_distance)
                }
            },
            'market_regime': {
                'volatility_level': 'high' if np.std(prices_array) > np.mean(prices_array) * 0.02 else 'low',
                'trend_following': trend_strength > 0.5,
                'mean_reverting': zero_crossing_rate > 0.4
            }
        }
        
        return json.dumps(result, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            'status': 'error',
            'error_type': 'JSONDecodeError',
            'message': f'Invalid JSON format: {str(e)}'
        }, indent=2)
    
    except ValueError as e:
        return json.dumps({
            'status': 'error',
            'error_type': 'ValueError',
            'message': str(e)
        }, indent=2)
    
    except TypeError as e:
        return json.dumps({
            'status': 'error',
            'error_type': 'TypeError',
            'message': str(e)
        }, indent=2)
    
    except ImportError as e:
        return json.dumps({
            'status': 'error',
            'error_type': 'ImportError',
            'message': f'Missing required library: {str(e)}'
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            'status': 'error',
            'error_type': type(e).__name__,
            'message': f'Unexpected error: {str(e)}'
        }, indent=2)

```
