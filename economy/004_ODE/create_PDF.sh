#!/bin/bash

# Install LaTeX on openSUSE Leap 16:
#   sudo zypper install texlive-scheme-full
# Install pandoc on openSUSE Leap 16:
#   sudo zypper install pandoc
# Install DejaVu fonts:
#   sudo zypper install dejavu-fonts

echo "Creating PDF with fixed image positions and numbered sections..."

# Create a temporary markdown with LaTeX headers
cat << 'EOF' > temp_readme.md
---
title: "RL book: code"
date: "November 2025"
author: "majid.kashani@alum.sharif.edu"
numbersections: true
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \usepackage{unicode-math}
  - \setmonofont{DejaVu Sans Mono}
geometry: margin=1in
fontsize: 11pt
---

EOF

# Append the README.md content
cat README.md >> temp_readme.md

# Convert with pandoc
pandoc temp_readme.md -o README.pdf --pdf-engine=xelatex

# Clean up
rm temp_readme.md

echo "Done! Check README.pdf"

# Optional: wait for user input (equivalent to 'pause')
# read -p "Press Enter to continue..."
