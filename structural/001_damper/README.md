An SDOF system with is assumed:

* Stiffness `k`
* Unit mass `m=1`
* Damping coefficient `c`
   * Damping ratio is `5%`
* Natural period of the system is `T=0.4 sec`

For seismic protection, an energy dissipating mechanism, i.e. damper, is added to the system with the attached hysteresis properties.

![SDOF system](media/SDOF.png "SDOF system")

![Hysteresis damper](media/hysteresis.png "Hysteresis damper")

The damper hysteresis properties shown on the attached figure are:

* `k_bar`
   * The stiffness
* `F_bar`
   * The yielding force


# System alone

For SDOF sytesm without hysteresis mechanism and under the two attached seismic time histories, I need:

1. System response
   * Displacement and its max value
   * Base shear and its max value
      * Store the lowest of max values for two seismic time history as `F_bs` variable
      * `F_bs` value will be used later
1. System energy
   * Total input energy
   * Damping energy
   * Elastic energy
   * Inertial energy
   * Chart comparison of energies, maybe by cumulative sum

# System with damper

Add three separate dampers with three hysteresis properties of:

* `k_bar = 0.1 k`
* `k_bar = 0.5 k`
* `k_bar = k `

Consider the `Fy` to be equal to `0.4 * F_bs` for all of the three `k_bar` cases above.

The system response and energy are needed for the system equipped with damper:

1. System response
   * Displacement and its max value
   * Base shear and its max value
1. System energy
   * Total input energy
   * Damping energy
   * Elastic energy
   * Inertial energy
   * Chart comparison of energies, maybe by cumulative sum

# Implications

Evaluate the performance of the damper by comparing:

* System alone
* System with damper
