"""Configuration parameters for 8-story building and analysis."""

import numpy as np

# Building properties
NUM_STORIES = 8
MASS = 345.6  # tons (per floor)
STIFFNESS = 3.404e5  # kN/m (per story)
DAMPING = 2937.0  # kN·s/m (per story)

# Earthquake records
RECORD_FILES = {"El Centro": "record-ELCENTRO", "Tabas": "record-TABAS"}

# Gravity
G = 9.81  # m/s²

# Control settings
R_MATRIX = np.eye(NUM_STORIES)  # Control weight matrix (R = I)
TARGET_REDUCTION = 0.25  # Target: 25% of uncontrolled response
