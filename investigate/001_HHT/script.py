import zmq
import json
import numpy as np
from PyEMD import CEEMDAN

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
print(f"Microservice on: {endpoint}")

while True:
    # Wait for request
    message = socket.recv_string()
    prices = json.loads(message)

    # Process with HHT
    ceemdan = CEEMDAN()
    IMFs = ceemdan(np.array(prices))

    # Calculate metrics
    result = {
        'num_imfs': len(IMFs),
        'imfs': IMFs.tolist(),
        'trend_direction': 'up' if IMFs[-1][-1] > IMFs[-1][0] else 'down',
        'volatility_components': [np.std(imf) for imf in IMFs[:-1]]
    }

    # Send reply
    socket.send_string(json.dumps(result))
