@echo off
cd /d "%~dp0"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Done. Restart Streamlit after this. If Gemini vision still fails, open the app error "Technical details" expander.
pause
