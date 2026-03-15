import importlib
import os

import emcee
import jeans
import numpy as np
import pandas as pd
from pathos.multiprocessing import (
    ProcessingPool as Pool,  # if you use the slower version just: from multiprocessing import Pool
)

import final_baryon_component  # should be the name of the Baryon_Pot file (final_baryon_component.py)

importlib.reload(final_baryon_component)  # again same name

# --- DATEN-FUNKTIONEN ---


def get_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data {filename} not found.")
    with open(filename) as f:
        lines = f.readlines()
    distance = float(lines[0].lstrip("# Distance = ").strip().split()[0])
    colnames = lines[1].lstrip("# ").split()
    units = lines[2].lstrip("# ").split()
    df = pd.read_csv(filename, comment="#", sep=r"\s+", names=colnames, skiprows=2)
    return df, units, distance


def c_MCR(M200):
    """Konzentrationsparameter nach MCR Relation."""
    a, b, h = 0.905, -0.101, 0.7
    return 10 ** (a + b * np.log10(h * M200 / 1e12))


# To get the files
#

# MCMC core Logistics


def logc_prior(theta):
    """
    Use:
        Includes an error factor from the mass-concentration relation with a Gaussian prior on log(c) centered on log(c) from the mass-concentration relation with a width of 0.11 dex.
    Takes:
        log(c)
    Returns:
        The likelihood for log(c)
    """
    logM200, c = theta
    M200 = 10**logM200
    cMCR = c_MCR(M200)

    log_c_0 = np.log10(cMCR)
    log_c = np.log10(c)
    dex_c = 0.11

    chi_squared_c = ((log_c - log_c_0) / dex_c) ** 2

    lnp = -0.5 * chi_squared_c

    return lnp


def log_prior(theta):
    # CDM nutzt nur logM und c
    logM, c = theta

    # 1. Check: Harte Grenzen (Flat Prior)
    if 8.0 < logM < 16.0 and 0.0 < c < 50.0:
        # 2. Check: Gauß-Prior auf log(c) aufrufen (!)
        return logc_prior(theta)

    return -np.inf


def log_likelihood(theta, df, Phi_b):
    logM, c = theta
    M200 = 10**logM
    try:
        profile = jeans.cdm(M200, c, Phi_b=Phi_b)
        if profile is None:
            return -np.inf
    except Exception as e:
        print(f"Error in jeans-profile at {theta}, with Error {e}")
        return -np.inf

    try:
        v_model = profile.V(df["Rad"], Lmax=0)
    except Exception as e:
        print(f"Error in rotation curve at {theta}, with Error {e}")
        return -np.inf
        # 1. Check: Sind die Modellwerte valide Zahlen?
    if not np.all(np.isfinite(v_model)):
        return -np.inf

        # Chi-2 Calc
    diff = df["Vobs"] - v_model
    chi2 = np.sum((diff / df["errV"]) ** 2)

    # 2. Check: Ist das Ergebnis von Chi2 valide? (Wichtig!)
    if not np.isfinite(chi2) or np.isnan(chi2):
        return -np.inf

    return -0.5 * chi2


def log_probability(theta, df, Phi_b):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, df, Phi_b)


# Main Function of the MCMC
# Change variables as you see fit
def MCMC(
    galaxy_name,
    nwalkers=16,
    nsteps=100,
    resume=False,
    start_logM=12.0,
    c_start=10.0,
    initial_volume=0.1,
    savename="",
    num_cores=4,
):

    # Path Setup
    path_data = os.path.join(os.getcwd(), "data", "Rotmod_LTG")
    path_chains = os.path.join(os.getcwd(), "chains")

    if not os.path.exists(path_chains):
        os.makedirs(path_chains)
        print(f"Opened New Folder: {path_chains}")

    # reverse extract galaxy name (e.g. IC4202->IC4202_rotmod.dat)
    data_filename = f"{galaxy_name}_rotmod.dat"
    full_data_path = os.path.join(path_data, data_filename)

    # NEW: file-name with nwalkers (nw) & nsteps (ns)
    filename = f"{savename}_{galaxy_name}_nw_{nwalkers}.h5"
    print(f"Files saved as: {filename}")
    backend_filename = os.path.join(
        path_chains,
        filename,  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    )

    # retrieve Baryon potential function from Baryon Team
    Phi_diskgas = final_baryon_component.hernquist_potentials_from_fit(galaxy_name)

    # Load Data: get_data - Function
    try:
        df, units_info, dist = get_data(full_data_path)
        print(f"Data for {galaxy_name} could be loaded.")
    except Exception as e:
        print(f"Error while loading Data: {e}")
        return None

    # MCMC Setup
    theta = (start_logM, c_start)
    ndim = len(theta)
    initial = np.array(theta)
    p0 = [initial + initial_volume * np.random.randn(ndim) for i in range(nwalkers)]

    # Backend Setup
    backend = emcee.backends.HDFBackend(backend_filename)

    if resume:
        # Important: Exists this file even?
        if not os.path.exists(backend_filename):
            print(f"no Backend found in {backend_filename}. Starting new...")

        else:
            print(f"Resume MCMC for {galaxy_name}...")
            p0 = None

    if not resume:
        # Kompletly new start: File deliting / newing
        print(f"Start new MCMC-Run for {galaxy_name}...")
        backend.reset(nwalkers, ndim)
        # Initiale Positionen for new file

    with Pool(nodes=num_cores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            args=(df, Phi_diskgas),
            pool=pool,
            backend=backend,
        )
        sampler.run_mcmc(p0, nsteps, progress=True)

    print(f"MCMC ended. Endnumber of Backends iterations: {backend.iteration}")

    return None
