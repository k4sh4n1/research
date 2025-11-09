@if not exist virtual_env\Scripts\activate.bat (
    python -m venv virtual_env
    call virtual_env\Scripts\activate.bat
    python -m pip install -r requirements.txt
) else (
    call virtual_env\Scripts\activate.bat
)

python script.py
pause
