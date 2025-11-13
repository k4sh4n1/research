# Description

An SDOF system is assumed:

* Stiffness $k$
* Unit mass $m=1$
* Damping coefficient $c$
   * Damping ratio $\zeta$ is `5%`
* Natural period of the system $T$ is `0.4 sec`

For seismic protection, an energy dissipating mechanism, i.e. damper, is added to the system with the attached hysteresis properties.

![SDOF system](media/SDOF.png "SDOF system")

![Hysteresis damper](media/hysteresis.png "Hysteresis damper")

The damper hysteresis properties shown on the attached figure are:

* $\bar{k}$ is damper stiffness
* $\bar{F}_y$ is dampler yielding force

# Seismic records

Two attached seismic acceleration time histories are considered. Their max value should be normalized to $0.4g$ before any computation.

# System alone

For SDOF sytesm without hysteresis mechanism and under the two seismic records, compute:

1. System response
   * Displacement and its max value
   * Base shear and its max value
      * Store the lowest of max values for two seismic time history as $F_{bs}$ variable
      * $F_{bs}$ value will be used later
1. System energy
   * Total input energy
   * Damping energy
   * Elastic energy
   * Kinetic energy
   * Chart comparison of energies, maybe by cumulative sum

# System with damper

Add three separate dampers with three hysteresis properties of:

* $\bar{k}=0.1k$
* $\bar{k}=0.5k$
* $\bar{k}=k$

Consider the $\bar{F}_y$ to be equal to $0.4 * F_{bs}$ for all of the three $\bar{k}$ cases above.

The system response and energy are needed for the system equipped with damper:

1. System response
   * Displacement and its max value
   * Base shear and its max value
1. System energy
   * Total input energy
   * Damping energy
   * Elastic energy
   * Kinetic energy
   * Chart comparison of energies, maybe by cumulative sum

# Implications

Evaluate the performance of the damper by comparing:

* System alone
* System with damper

# Charts

These charts mgiht provide an acceptable visualization:

* One chart with only one sub-plot
   * Comparing two scaled seismic records on top of each other
* For system without damper, one chart with 2 sub-plots
   * One sub-plot for comparing energy components
      * Input energy
      * Kinetic energy
      * Damping energy
      * Elastic energy
   * One sub-plot for checking energy balance
      * Input energy
      * Sum of kinetic, damping, and elastic energies
* For each damper case, one chart with 4 sub-plots in a 2x2 arrangement
   * One sub-plot for comparing displacement without and with damper
   * One sub-plot for comparing base shear without and with damper
   * One sub-plot for comparing energy components
      * Input energy
      * Kinetic energy
      * Damping energy
      * Elastic energy
      * Hysteresis energy of damper
   * One sub-plot for checking energy balance
      * Input energy
      * Sum of kinetic, damping, elastic, and damper hysteresis energies
* One chart with 3 sub-plots
   * Each sub-plot corresponds to the hysteresis loop of one of 3 damper cases

# Simulation details

This is how the simulation will be done:

* Two seismic record file names are `seismic1.txt` and `seismic2.txt`
   * Each of the two seismic records have 4 header lines
   * 1st seismic record file has a header line like this: `NPTS=  1192, DT= .02000 SEC`
   * 2nd seismic record file has a header line like this: `NPTS=  4000, DT= .01000 SEC`
   * Seismic record files have to be read line-by-line
   * They can have a variable number of data on each line
* `OpenSeesPy` module is employed for structural analysis
   * A one-dimensional model is created by `OpenSeesPy`
   * Elastic-perfectly-plastic behavior of the hysteresis damper will be modeled
   * Integrator is set to `integrator('Newmark', 0.5, 0.25)` for unconditional stability
   * Method `.reactions()` of `OpenSeesPy` is called before getting node reactions

The code and the plotting functions are kept:

* Minimal
* Clear
* Concise
* Readable
* Maintainable
