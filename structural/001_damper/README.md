An SDOF system with is assumed:

* Stiffness `k`
* Unit mass `m=1`
* Damping coefficient `c`
   * Damping ratio is `5%`
* Natural period of the system is `T=0.4 sec`

For seismic protection, an energy dissipating mechanism, i.e. damper, is added to the system with the attached hysteresis properties.

![SDOF system](media/SDOF.png "SDOF system")
![Hysteresis damper](media/hysteresis.png "Hysteresis damper")

# System alone

For SDOF sytesm without hysteresis mechanism and under the two attached seismic time histories, I need:

1. System response
   * Displacement and base shear and also their max value
1. System energy
   * Total input energy
   * Damping energy
   * Elastic energy
   * Inertial energy
   * Chart comparison of energies, maybe by cumulative sum

# System with damper

Add three separate dampers with three hysteresis properties of:

*
