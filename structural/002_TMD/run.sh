#!/bin/bash

if [ ! -f "virtual_env/bin/activate" ]; then
    python3 -m venv virtual_env
    source virtual_env/bin/activate
    python3 -m pip install -r requirements.txt
else
    source virtual_env/bin/activate
fi

python3 script1_modal.py
python3 script1_OpenSees.py
python3 script2_num.py
python3 script2_num_.py
python3 script2_OpenSees.py
python3 script2_OpenSees_.py
python3 script3_OpenSees.py
python3 script4_OpenSees.py
read -p "Press Enter to continue..."
