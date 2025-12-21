"""Earthquake record parser for PEER format."""

import numpy as np


def read_peer_record(filename):
    """
    Parse PEER format ground motion file.

    Returns: dt (time step), accel (acceleration in g)
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Parse header (line 4) for NPTS and DT
    header = lines[3]
    npts = int(header.split("NPTS=")[1].split(",")[0])
    dt = float(header.split("DT=")[1].split()[0])

    # Parse acceleration values (starting line 5)
    accel = []
    for line in lines[4:]:
        accel.extend(map(float, line.split()))

    return dt, np.array(accel[:npts])
