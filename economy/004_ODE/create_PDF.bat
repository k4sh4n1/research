@echo off
REM Install LaTeX, on Windows: https://miktex.org/download
REM Install pandoc, on Windows: `winget install --source winget --exact --id JohnMacFarlane.Pandoc`
REM Let package installation be done by the pop-up dialogs

echo Creating PDF with fixed image positions and numbered sections...

REM Create a temporary markdown with LaTeX headers
(
echo ---
echo title: "RL book: code"
echo date: "November 2025"
echo author: "majid.kashani@alum.sharif.edu"
echo numbersections: true
echo header-includes:
echo   - \usepackage{float}
echo   - \floatplacement{figure}{H}
echo   - \usepackage{unicode-math}
echo   - \setmonofont{DejaVu Sans Mono}
echo geometry: margin=1in
echo fontsize: 11pt
echo ---
echo.
type README.md
) > temp_readme.md

REM Convert with pandoc
pandoc temp_readme.md -o README.pdf --pdf-engine=xelatex

REM Clean up
del temp_readme.md

echo Done! Check README.pdf
pause
