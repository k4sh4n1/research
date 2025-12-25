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
| **El Centro** | Max Roof Disp (m) | 0.1821 | 0.0460 | 0.0464 |
| | Max Roof Accel (m/s²) | 6.33 | 3.50 | 14.01 |
| | Max Base Shear (kN) | 16033.9 | 9462.7 | 8621.6 |
| | Max Floor 1 Ctrl Force (kN) | N/A | 3597.8 | 2940.7 |
| | Max Floor 8 Ctrl Force (kN) | N/A | 652.0 | 4866.7 |
| | Roof Disp Reduction | — | 74.8% | 74.5% |
| **Tabas** | Max Roof Disp (m) | 0.0895 | 0.0227 | 0.0228 |
| | Max Roof Accel (m/s²) | 4.68 | 3.63 | 17.17 |
| | Max Base Shear (kN) | 11242.9 | 11323.1 | 10343.8 |
| | Max Floor 1 Ctrl Force (kN) | N/A | 5544.1 | 3670.9 |
| | Max Floor 8 Ctrl Force (kN) | N/A | 771.3 | 5920.9 |
| | Roof Disp Reduction | — | 74.7% | 74.5% |

# Implication

- LQR:
   * Looks ahead
   * Is considering the future
   * Balances displacement & acceleration naturally
   * Result: reduces both displacement and acceleration
- IOC:
   * Is greedy
   * Only sees one step ahead
   * Applies sudden, large forces to minimize next-step disp
   * Result: low displacement, but high acceleration

IOC essentially “jerks” the building to reduce displacement, but causes acceleration spikes.

# Note

The R matrix is picked as identity matrix for both LQR and IOC approaches.

The Q matrix is diagonal, but different for LQR and IOC approaches. It is referred to the attached Python code to observe how Q matrix is selected differently for LQR and IOC.

IOC heavily penalizes roof displacement, due to its Q matrix selection. It applies large forces directly at the roof to minimize roof displacement immediately, regardless of the acceleration cost.

It should be noted that the actuators are simulated as inter-story to consider coupling through reaction forces.
