---
title: "Modeling earthqke waveforms as candlesticks on financial charts"
date: "January 2026"
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

https://github.com/k4sh4n1/research/tree/master/earthquake/004_candle

# Objective

Fetching earthquake waveforms from reliable seismic observatories by `ObsPy` module and modeling them just like candlestick patterns on financial charts.

For these reasons:

* To be able to re-apply the RL+DL financial methodologies to the seismic time-series
* To compare between financial time-series and seismic waveforms more conveniently
* ...?

# Locations

* IU_MAJO_BHZ: Matsushiro, Japan
* II_ERM_BHZ: Erimo, Japan
* IU_INCN_BHZ: Incheon, South Korea
* G_INU_BHZ: Inuyama, Japan

# Time window

Investigation time window contains the following earthquake event:

> A powerful 7.6-magnitude earthquake struck off the northeastern coast of Japan on December 8, 2025, at 11:15 p.m. local time (1415 GMT), with its epicenter located 80 km (50 miles) off the coast of Aomori Prefecture at a depth of 50 km.

# Map

![Station map](station_map.png "Station map")

# Result plot

![Matsushiro Station: MAJO Candlestick](candle_IU_MAJO_BHZ.png)

![Erimo Station: ERM Candlestick](candle_II_ERM_BHZ.png)

![Incheon Station: INCN Candlestick](candle_IU_INCN_BHZ.png)

![Inuyama Station: INU Candlestick](candle_G_INU_BHZ.png)

# Implication

* Earthquake waveforms actually look like candlesticks on financial charts
* This was an encouraging step for more investigations
