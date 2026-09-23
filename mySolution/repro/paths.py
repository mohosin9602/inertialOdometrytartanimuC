"""Shared path resolution for the reproduction scripts.

Everything is derived from this file's location, so the scripts work from any
working directory and contain no hard-coded absolute paths.
"""
import os
import sys

REPRO = os.path.dirname(os.path.abspath(__file__))
MYSOLUTION = os.path.dirname(REPRO)
ROOT = os.path.dirname(MYSOLUTION)

TARTAN = os.path.join(ROOT, "TartanIMU")            # kept official repo here in the root
STARTER = os.path.join(TARTAN, "starter")
DATA = os.path.join(MYSOLUTION, "data", "raw")      # kaggle data: train/ val/ test/ index/
OUT = os.path.join(REPRO, "out")                    # all generated artifacts

WIN = 200                                            # frames per window (1.0 s)
FS = 200.0                                           # IMU rate, Hz


def add_import_paths():
    """Make the official scorer and the tartan_imu library importable."""
    for p in (STARTER, TARTAN, REPRO):
        if p not in sys.path:
            sys.path.insert(0, p)
