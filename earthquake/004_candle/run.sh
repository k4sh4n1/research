#!/bin/bash

if [ ! -f "virtual_env/bin/activate" ]; then
    python3 -m venv virtual_env
    source virtual_env/bin/activate
    python3 -m pip install -r requirements.txt
else
    source virtual_env/bin/activate
fi

python3 script.py

