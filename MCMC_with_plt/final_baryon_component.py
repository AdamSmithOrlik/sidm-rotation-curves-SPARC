import os

import numpy as np
import pandas as pd

G = 4.30091e-6  # kpc (km/s)^2 / Msun


# Hernquist
def phiH(Mb, a):
    "Returns a function Phi(r, th) for Hernquist potential"

    def Phi_hernquist(r, th=0):
        r = np.asarray(r)
        return -G * Mb / (r + a)

    return Phi_hernquist


# CSV Path and Import
def get_csv_path(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, filename)


def load_fits(fit_file="hernquist_fits_component.csv"):

    filepath = get_csv_path(fit_file)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")

    return pd.read_csv(filepath)


# Return potential for galaxy
def hernquist_potentials_from_fit(galaxyID, fit_file="hernquist_fits_component.csv"):
    """
    Returns:
        - Phi_diskgas
        - Phi_bulge
    """
    df = load_fits(fit_file)
    row = df[df["GalaxyID"] == galaxyID]
    if row.empty:
        raise ValueError(f"{galaxyID} not found in fit file")

    M_diskgas = row["M_diskgas"].values[0]
    a_diskgas = row["a_diskgas"].values[0]

    # M_bulge   = row["M_bulge"].values[0]
    # a_bulge   = row["a_bulge"].values[0]

    Phi_diskgas = phiH(M_diskgas, a_diskgas)
    # Phi_bulge   = phiH(M_bulge, a_bulge)

    return Phi_diskgas  # ,Phi_bulge
