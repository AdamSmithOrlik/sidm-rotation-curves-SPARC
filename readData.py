import numpy as np
import pandas as pd
import os
import glob

path = os.getcwd() + "/data/"


def get_galaxy_ids():
    # get the galaxy names from the filenames
    filenames = glob.glob(path + "Rotmod_LTG/*.dat")
    galaxy_IDs = [os.path.basename(f).split("_")[0] for f in filenames]
    return galaxy_IDs


def get_rc_data(galaxy_id):
    filename = path + "Rotmod_LTG/" + galaxy_id + "_rotmod.dat"
    with open(filename) as f:
        lines = f.readlines()

    distance = float(lines[0].lstrip("# Distance = ").strip().split()[0])
    colnames = lines[1].lstrip("# ").split()
    units = lines[2].lstrip("# ").split()

    df = pd.read_csv(
        filename,
        comment="#",
        sep=r"\s+",
        names=colnames,
        skiprows=2,  # skip the two # header lines
    )

    # stack colnames and units to turn into a new df
    units_df = pd.DataFrame([units], columns=colnames)

    return df, units_df, distance


def get_mass_data(galaxy_id):

    filename = path + "SPARC_Lelli2016c.mrt"

    names = [
        "Galaxy",
        "T",
        "D",
        "e_D",
        "f_D",
        "Inc",
        "e_Inc",
        "L36",
        "e_L36",
        "Reff",
        "SBeff",
        "Rdisk",
        "SBdisk",
        "MHI",
        "RHI",
        "Vflat",
        "e_Vflat",
        "Q",
        "Ref",
    ]

    units = [
        "",  # Galaxy
        "",  # T (Hubble type)
        "Mpc",  # D
        "Mpc",  # e_D
        "",  # f_D
        "deg",  # Inc
        "deg",  # e_Inc
        "1e9 Lsun",  # L36
        "1e9 Lsun",  # e_L36
        "kpc",  # Reff
        "Lsun/pc^2",  # SBeff
        "kpc",  # Rdisk
        "Lsun/pc^2",  # SBdisk
        "1e9 Msun",  # MHI
        "kpc",  # RHI
        "km/s",  # Vflat
        "km/s",  # e_Vflat
        "",  # Q
        "",  # Ref
    ]

    df = pd.read_csv(
        filename,
        sep=r"\s+",
        skiprows=98,
        header=None,
        names=names,
        engine="python",
    )

    units_df = pd.DataFrame([units], columns=names)

    galaxy_row = df[df["Galaxy"] == galaxy_id].reset_index(drop=True)

    return galaxy_row, units_df


# for testing
def main():
    d, u = get_mass_data("NGC5371")
    print(d)
    print(u)

    return None


if __name__ == "__main__":
    main()
