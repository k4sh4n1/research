@if not exist virtual_env\Scripts\activate.bat (
    python -m venv virtual_env
    call virtual_env\Scripts\activate.bat
    python -m pip install -r test_client_requirements.txt
) else (
    call virtual_env\Scripts\activate.bat
)

python test_client.py
pause
