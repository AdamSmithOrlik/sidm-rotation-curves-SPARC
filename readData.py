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


def get_ML_data(galaxy_id):

    filename = path + f"Fits/ByGalaxy/Table/{galaxy_id}.mrt"

    names = [
        "Model",
        "Ydisk",
        "e_Ydisk",
        "Ybul",
        "e_Ybul",
        "D",
        "e_D",
        "inc",
        "e_inc",
        "V200",
        "e_V200",
        "C200",
        "e_C200",
        "rs",
        "e_rs",
        "log_rhos",
        "e_log_rhos",
        "log_M200",
        "e_log_M200",
        "alpha",
        "e_alpha",
        "Chi",
    ]

    units = [
        "",  # Model
        "",  # Ydisk
        "",  # e_Ydisk
        "",  # Ybul
        "",  # e_Ybul
        "Mpc",  # D
        "Mpc",  # e_D
        "deg",  # inc
        "deg",  # e_inc
        "km/s",  # V200
        "km/s",  # e_V200
        "",  # C200
        "",  # e_C200
        "kpc",  # rs
        "kpc",  # e_rs
        "Msun/pc^3",  # log_rhos
        "Msun/pc^3",  # e_log_rhos
        "Msun",  # log_M200
        "Msun",  # e_log_M200
        "",  # alpha
        "",  # e_alpha
        "",  # Chi
    ]

    df = pd.read_csv(
        filename,
        sep=r"\s+",
        skiprows=34,
        header=None,
        names=names,
        engine="python",
    )

    units_df = pd.DataFrame([units], columns=names)

    return df, units_df


def Gamma_disk(galaxy):
    """
    Calculates disk mass-to-light ratio Gamma from Eq 10 arXiv:2601.17118
    using catalogs from arXiv:2001.10538 accessed from SPARC database.
    Takes:
        galaxy : galaxy ID string
    Returns:
        gamma : mass-to-light ratio scaling factor for disk
        sigma_eff : uncertainty in gamma
    """
    df, _ = get_ML_data(galaxy)
    numerator = np.sum(df["Ydisk"] / df["e_Ydisk"] ** 2)
    denominator = np.sum(1 / df["e_Ydisk"] ** 2)
    gamma = np.divide(numerator, denominator)

    # uncertainty in gamma
    sigma_gamma_sq = 1 / denominator

    # add spread in Ydisk values since Gamma_i are correlated
    w = 1 / df["e_Ydisk"] ** 2
    spread_sq = np.sum(w * (df["Ydisk"] - gamma) ** 2) / np.sum(w)
    sigma_eff = np.sqrt(sigma_gamma_sq + spread_sq)

    return gamma, sigma_eff


def Gamma_bulge(galaxy):
    """
    Calculates bulge mass-to-light ratio Gamma from Eq 10 arXiv:2601.17118
    using catalogs from arXiv:2001.10538 accessed from SPARC database.
    Takes:
        galaxy : galaxy ID string
    Returns:
        gamma : mass-to-light ratio scaling factor for bulge
        sigma_eff : uncertainty in gamma
    """
    df, _ = get_ML_data(galaxy)
    if df["Ybul"].all() == 0:
        return 0.0, 0.0
    numerator = np.sum(df["Ybul"] / df["e_Ybul"] ** 2)
    denominator = np.sum(1 / df["e_Ybul"] ** 2)
    gamma = np.divide(numerator, denominator)

    # uncertainty in gamma
    sigma_gamma_sq = 1 / denominator

    # add spread in Ybulge values since Gamma_i are correlated
    w = 1 / df["e_Ybul"] ** 2
    spread_sq = np.sum(w * (df["Ybul"] - gamma) ** 2) / np.sum(w)
    sigma_eff = np.sqrt(sigma_gamma_sq + spread_sq)

    return gamma, sigma_eff


# for testing
def main():
    gd, ed = Gamma_disk("NGC5371")
    gb, eb = Gamma_bulge("NGC5371")
    print(f"Disk Gamma: {gd} +/- {ed}")
    print(f"Bulge Gamma: {gb} +/- {eb}")

    return None


if __name__ == "__main__":
    main()
