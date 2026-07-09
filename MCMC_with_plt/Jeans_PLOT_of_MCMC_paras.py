import importlib
import os

import emcee
import jeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import final_baryon_component as fbc

importlib.reload(fbc)


def get_data(filename):
    with open(filename) as f:
        lines = f.readlines()

    distance = float(lines[0].lstrip("# Distance = ").strip().split()[0])
    colnames = lines[1].lstrip("# ").split()
    units_list = lines[2].lstrip("# ").split()

    df = pd.read_csv(filename, comment="#", sep=r"\s+", names=colnames, skiprows=2)
    units_df = pd.DataFrame([units_list], columns=colnames)
    return df, units_df, distance


def Jeans_MCMC_plot(galaxy_name):
    # Define Path
    filename_h5 = f"chains_{galaxy_name}_rotmod.h5"
    path_chain = os.path.join(os.getcwd(), "chains", filename_h5)

    path_data = os.path.join(
        os.getcwd(), "data", "Rotmod_LTG", f"{galaxy_name}_rotmod.dat"
    )

    # Check for bugs:
    if not os.path.exists(path_chain):
        print(f"DEBUG: File NOT found!")
        print(f"Sourched path: {path_chain}")
        print(f"does the folder 'chains' exist?: {os.path.exists('chains')}")
        return

    if not os.path.exists(path_data):
        print(f"Error: Data-file not found: {path_data}")
        return
    df, units, dist = get_data(path_data)

    # MCMC-Parameter from median of simulated
    backend = emcee.backends.HDFBackend(path_chain, read_only=True)

    # 30 % Burnin
    burnin = int(0.3 * backend.iteration)
    flat_samples = backend.get_chain(discard=burnin, flat=True)

    # Median extraction of: [r1, log10(M200), c]
    medians = np.median(flat_samples, axis=0)

    r1_best = medians[0]
    M200_best = 10 ** medians[1]  # log10 -> linear
    c_best = medians[2]

    print(f"{galaxy_name} Fit-Results")
    print(f"r1:   {r1_best:.3f} kpc")
    print(f"M200: {M200_best:.2e} M_sun")
    print(f"c:    {c_best:.3f}")

    # Jeans-Modell with Baryon-Potential
    Phi_diskgas = fbc.hernquist_potentials_from_fit(galaxy_name)
    profile = jeans.squashed(
        r1_best, M200_best, c_best, Phi_b=Phi_diskgas, verbose=False
    )

    # Plot
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        df["Rad"],
        df["Vobs"],
        yerr=df["errV"],
        fmt="o",
        color="black",
        markersize=4,
        capsize=2,
        label=f"Data ({galaxy_name})",
        alpha=0.7,
    )

    v_model = profile.V(df["Rad"], Lmax=0)
    plt.plot(df["Rad"], v_model, color="royalblue", lw=2, label="Jeans Model (MCMC)")
    plt.xlabel(f"Radius ({units.at[0, 'Rad']})")
    plt.ylabel(f"Velocity ({units.at[0, 'Vobs']})")
    plt.title(f"Rotation Curve: {galaxy_name} (Distance: {dist} Mpc)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


# How to use for 'DDO170_rotmod.dat':
# Jeans_MCMC_plot('DDO170')
