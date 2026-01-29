import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

path = os.getcwd() + "/data/Rotmod_LTG/"
savepath = os.getcwd() + "/plots/"

# get the galaxy names from the filenames
filenames = glob.glob(path + "*.dat")
galaxy_IDs = [os.path.basename(f).split(".")[0] for f in filenames]


# Helper function to read in data
def get_data(filename):
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


# plot the rotation curves for each galaxy
for galaxy in galaxy_IDs:
    df, units_df, distance = get_data(path + galaxy + ".dat")

    vc_baryons = np.sqrt(df["Vdisk"] ** 2 + df["Vbul"] ** 2 + df["Vgas"] ** 2)

    plt.figure(figsize=(6, 4))
    plt.scatter(df["Rad"], vc_baryons, label="Baryon Total", color="k", s=10, zorder=3)

    plt.xlabel(f"Radius (kpc)")
    plt.ylabel(f"Velocity (km/s)")
    plt.title(f"Galaxy {galaxy} at Distance {distance} Mpc")
    plt.legend()
    plt.grid(zorder=0)
    plt.savefig(savepath + f"{galaxy}_baryon_RC.png")
    plt.close()
