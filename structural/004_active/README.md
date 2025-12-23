---
title: "Problem: active control"
date: "December 2025"
author: "majid.kashani@alum.sharif.edu   1404 84 111 03"
numbersections: true
geometry: margin=1in
fontsize: 11pt
header-includes: |
  \usepackage{float}
  \floatplacement{figure}{H}
---

# Problem statement

![Problem statement](20251221_112818.jpg "Problem statement")

# Result

![Roof displacement](roof_displacement.png "Roof displacement")

![Roof acceleration](roof_acceleration.png "Roof acceleration")

![Base shear](base_shear.png "Base shear")

![Control forces](control_forces.png "Control forces")

# Table

| Earthquake | Response | Uncontrolled | LQR | Instantaneous |
|:-----------|:---------|-------------:|----:|--------------:|
| **El Centro** | Max Roof Disp (m) | 0.1821 | 0.0453 | 0.0438 |
| | Max Roof Accel (m/s²) | 6.33 | 3.98 | 3.18 |
| | Max Base Shear (kN) | 16033.9 | 9594.8 | 9428.8 |
| | Max Floor 1 Ctrl Force (kN) | N/A | 128.2 | 128.2 |
| | Max Floor 8 Ctrl Force (kN) | N/A | 695.4 | 700.3 |
| | Roof Disp Reduction | — | 75.2% | 75.9% |
| **Tabas** | Max Roof Disp (m) | 0.0895 | 0.0227 | 0.0220 |
| | Max Roof Accel (m/s²) | 4.68 | 3.61 | 4.03 |
| | Max Base Shear (kN) | 11242.9 | 11712.4 | 11292.5 |
| | Max Floor 1 Ctrl Force (kN) | N/A | 242.7 | 273.1 |
| | Max Floor 8 Ctrl Force (kN) | N/A | 860.7 | 903.3 |
| | Roof Disp Reduction | — | 74.7% | 75.4% |
