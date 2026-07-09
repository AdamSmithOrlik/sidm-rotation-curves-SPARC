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


# To get the files


# MCMC core Logistics


def log_prior(theta):
    r1, logM, c = theta
    # Updated prior ranges with third variable: 0 < r1 < 100, 8 < logM < 16, 0 < c < 50
    if 0.0 < r1 < 100.0 and 8.0 < logM < 16.0 and 0.0 < c < 50.0:
        return 0.0
    return -np.inf


def log_likelihood(theta, df, Phi_b):
    r1, logM, c = theta
    M200 = 10**logM
    try:
        # pass the potential function handle directly to Phi_b which is defined in MCMC-Function
        profile = jeans.squashed(r1, M200, c, Phi_b=Phi_b)

        if profile is None:
            return -np.inf

        v_model = profile.V(df["Rad"], Lmax=0)

        if not np.all(np.isfinite(v_model)):
            return -np.inf

        # Chi-2 Calc
        chi2 = np.sum(((df["Vobs"] - v_model) / df["errV"]) ** 2)
        return -0.5 * chi2
    except:
        return -np.inf


def log_probability(theta, df, Phi_b):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, df, Phi_b)


# Main Function of the MCMC
# Change variables as you see fit
def MCMC(
    galaxy_name,
    nwalkers=8,
    nsteps=100,
    initial_r1=20.0,
    resume=False,
    start_logM=10.0,
    c_start=10.0,
    initial_volume=0.1,
):

    # Path Setup
    path_data = os.path.join(os.getcwd(), "../data", "Rotmod_LTG")
    path_chains = os.path.join(os.getcwd(), "chains")

    if not os.path.exists(path_chains):
        os.makedirs(path_chains)
        print(f"Opened New Folder: {path_chains}")

    # reverse extract galaxy name (e.g. IC4202->IC4202_rotmod.dat)
    data_filename = f"{galaxy_name}_rotmod.dat"
    full_data_path = os.path.join(path_data, data_filename)
    backend_filename = os.path.join(path_chains, f"chains_{galaxy_name}_rotmod.h5")

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
    theta = (initial_r1, start_logM, c_start)
    ndim = len(theta)
    initial = np.array(theta)
    p0 = [initial + initial_volume * np.random.randn(ndim) for i in range(nwalkers)]

    # Backend Setup
    backend = emcee.backends.HDFBackend(backend_filename)
    if not resume:
        backend.reset(nwalkers, ndim)

    with Pool(nodes=4) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            args=(df, Phi_diskgas),
            pool=pool,
            backend=backend,
        )
        sampler.run_mcmc(p0, nsteps, progress=True)
    # Extracting Results
    af = np.mean(sampler.acceptance_fraction)

    flat_samples = sampler.get_chain(flat=True)
    log_probs = sampler.get_log_prob(flat=True)

    best_idx = np.argmax(log_probs)
    theta_best = flat_samples[best_idx]

    r1_best, logM_best, c_best = theta_best
    M200_best = 10**logM_best
    return
