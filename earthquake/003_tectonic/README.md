---
title: "Tectonic correlation"
date: "December 2025"
author: "majid.kashani@alum.sharif.edu   1404 84 111 03"
numbersections: true
geometry: margin=1in
fontsize: 11pt
header-includes: |
  \usepackage{float}
  \floatplacement{figure}{H}
---

# Code

Code is available here:

https://github.com/k4sh4n1/research/tree/master/earthquake/003_tectonic

# Objective

Just like the correlation of financial markets, the seismic time histories might be correlated to each other due to correlation of tectonic plates.

Tectonic correlation is investigated between strategic stations around the globe by `ObsPy` module. Temporal and spectral correlations are investigated which could indicate:

* Global wave propagation
* Force transfer between plate boundaries
* Common tectonic forces
* ...?

# Locations

Seismic signals from different tectonic plate boundaries are selected at strategic locations around the globe. Station locations are at major plate boundary intersections, shown on the following map.

# Time window

Investigation time window a few hours before and after the following earthquake event:

> A powerful 7.6-magnitude earthquake struck off the northeastern coast of Japan on December 8, 2025, at 11:15 p.m. local time (1415 GMT), with its epicenter located 80 km (50 miles) off the coast of Aomori Prefecture at a depth of 50 km.

# Map

![Station map](station_map.png "Station map")

# Result plot

![Waveforms comparison](waveforms_comparison.png "Waveforms comparison")

![Correlation matrix](correlation_matrix.png "Correlation matrix")

![Spectral coherence](spectral_coherence.png "Spectral coherence")

# Implication

* Station distribution for the global coverage:
   * Tectonic diversity: major plate boundaries
   * Geographic spread: coverage around the Pacific
   * Tabriz proximity: a station at Turkey/Caucasus is included
* Time hisotry delay from Japan to other stations:
   * Indicates seismic wave propagation across the globe
   * Wave propagation is verified
* Cross-correlation matrix:
   * Significant positive correlations up to more than `0.4`
   * Why Japan shows low correlation with everyone:
      * Japan was close to the earthquake that the signal is extremely different in character
      * Near-field records differ from far observations
   * Turkey-Alaska correlation `0.43` suggests similar response to events at Pacific
   * Negative Alaska-California correlation `-0.50`:
     * Suggests the waves arrive at these stations with opposite polarities
     * Might be due to their different angles/positions relative to the source i.e. Japan
     * Possibly they are on opposite sides of the mechanism planes
* Spectral coherence:
   * Japan has a low coherence with others: expected due to near-field effect
   * Turkey has a strong coherence with Indonesia and California
      * That's interesting. What's the cause?
      * Possibly, similar tectonic mechanisms — both sit on similar continental crust?
      * Possibly, comparable distances from Japan? Similar seismic wave travel?
