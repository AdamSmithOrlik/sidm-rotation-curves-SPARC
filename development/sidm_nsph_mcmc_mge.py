import numpy as np
import pandas as pd
import os
import emcee
import jeans
from mge_potential import BaryonicModel
from multiprocessing import Pool
import time as t

from readData import get_rc_data, log_M200_weighted_mean
import argparse

# GalaxyID = "F568-3" # test galaxy
# Command line input for the GalaxyID
parser = argparse.ArgumentParser()
parser.add_argument(
    'GALAXYID', 
    type=str, 
    help='The Galaxy ID for the SPARC dataset.')

args = parser.parse_args()
GALAXYID = args.GALAXYID

parser.add_argument(
    'QDISK', 
    type=float, 
    help='The disk flattening ratio.')

args = parser.parse_args()
QDISK = args.QDISK

# # Load baryon fit parameters for potential calculation
# baryon_fit_data = pd.read_csv(os.getcwd() + "/../data/baryon-fits/hern_fits.csv")
# baryon_data = baryon_fit_data[baryon_fit_data["GalaxyID"] == GALAXYID]
# M_b_fit = baryon_data["M_b_fit"].values[0]  # solar masses
# a_fit = baryon_data["a_fit"].values[0]  # kpc

# # Load the rotation curve data
# data, _, _ = get_rc_data(GALAXYID)
# r_data = data["Rad"].values  # kpc
# array_size = len(r_data)  # for blobs
# min_r = np.min(r_data)  # likelihood to 0 below min radius in data
# v_data = data["Vobs"].values  # km/s
# v_err = data["errV"].values  # km/s


######################################################################
######################### HELPER FUNCTIONS ###########################
######################################################################
def transform(theta):
    r1, logM200, logc, q0, log_upsilon_disk, log_upsilon_bulge, inclination, distance = theta
    M200 = 10**logM200
    # c = c_MCR(M200) * (10**dlogc)
    c = 10**logc
    upsilon_disk = 10**log_upsilon_disk
    upsilon_bulge = 10**log_upsilon_bulge
    
    return r1, M200, c, q0, upsilon_disk, upsilon_bulge, inclination, distance


# c from mass-concentration relation (Dutton & Maccio, 2014)
def c_MCR(M200):
    """
    Returns:
        c from the mass-concentration relations without error factor delta(log(c))
    """
    a, b = 0.905, -0.101
    h = 0.7
    c = 10 ** (a + b * np.log10(h * M200 / 1e12))

    return c

def logc_prior(theta):
    """
    Use:
        Includes an error factor from the mass-concentration relation with a Gaussian prior on log(c) centered on log(c) from the mass-concentration relation with a width of 0.11 dex.
    Takes:
        log(c)
    Returns:
        The likelihood for log(c)
    """
    _, logM200, log_c, _, _, _, _ = theta
    M200 = 10**logM200
    cMCR = c_MCR(M200)
    
    log_c_0 = np.log10(cMCR)
    dex_c = 0.11

    chi_squared_c = ((log_c - log_c_0) / dex_c) ** 2

    lnp = -0.5 * chi_squared_c

    return lnp

def log_upsilon_disk_prior(theta):
    _, _, _, _, log_upsilon_disk, _, _, _ = theta
    log_upsilon_disk_0 = np.log10(0.5)
    dex_upsilon_disk = 0.1
    
    chi_squared_upsilon_disk = ((log_upsilon_disk - log_upsilon_disk_0) / dex_upsilon_disk) ** 2
    
    lnp = -0.5 * chi_squared_upsilon_disk
    
    return lnp

def log_upsilon_bulge_prior(theta):
    _, _, _, _, _, log_upsilon_bulge, _, _ = theta
    log_upsilon_bulge_0 = np.log10(0.7)
    dex_upsilon_bulge = 0.1
    
    chi_squared_upsilon_bulge = ((log_upsilon_bulge - log_upsilon_bulge_0) / dex_upsilon_bulge) ** 2
    
    lnp = -0.5 * chi_squared_upsilon_bulge
    
    return lnp

# gaussian priors on inclination 
def inclination_prior(theta, bar):
    _, _, _, _, _, _, inclination, _ = theta
    inclination_0 = bar.incl
    sigma_inclination = bar.incl_err
    
    chi_squared_inclination = ((inclination - inclination_0) / sigma_inclination) ** 2
    
    lnp = -0.5 * chi_squared_inclination
    
    return lnp

def distance_prior(theta, bar):
    _, _, _, _, _, _, _, distance = theta
    distance_0 = bar.dist
    sigma_distance = bar.dist_err
    
    chi_squared_distance = ((distance - distance_0) / sigma_distance) ** 2
    
    lnp = -0.5 * chi_squared_distance
    
    return lnp


######################################################################
########################## MCMC FUNCTIONS ############################
######################################################################
def model(theta, bar, **kwargs):
    r1, M200, c, q0, upsilon_disk, upsilon_bulge, inclination, distance = transform(theta)

    phi = bar.potential_function(Upsilond=upsilon_disk, Upsilonb=upsilon_bulge, D=distance)   # phi(r, th)

    profile = jeans.squashed(r1, M200, c, q0=q0, Phi_b=phi, **kwargs)

    return profile


def likelihood(theta, bar, **kwargs):
    try:
        profile = model(theta, bar, **kwargs)
        if profile is None:
            print(f"Jeans model returned None for theta: {theta}")
            return -np.inf, None, None, None
    except Exception as e:
        print(f"Jeans model error for theta: {theta}\nError: {e}")
        return -np.inf, None, None, None
    
    r1, M200, c, q0, upsilon_disk, upsilon_bulge, inclination, distance = transform(theta)

    # total rotation curve with inclination and distance
    _, v_data, v_err = bar.vobs_at_inclination(inclination)
    r_data = bar.data_radii(D=distance)
    
    # Calculate the model rotation curve at the data radii
    try:
        v_model = profile.V(r_data, Lmax=2) # Lmax=2 for nonspherical Jeans model
    except Exception as e:
        print(f"Rotation curve calculation error for theta: {theta}\nError: {e}")
        return -np.inf, None, None, None

    # Chi-squared
    chi_squared = np.sum(((v_data - v_model) / v_err) ** 2)

    log_likelihood = -0.5 * chi_squared

    cross_section = profile.cross_section()
    
    vrel = profile.inner.vrel # km/s

    return log_likelihood, cross_section, v_model, vrel


def ln_prior(theta, bar):
    r1, logM200, dlogc, q0, log_upsilon_disk, log_upsilon_bulge, inclination, distance = theta
    
    c = 10**dlogc

    r1_condition = min_r < r1 < 500.0
    
    # uniform prior on logM200 between 1e8 and 1e15 Msun
    logM200_condition = 8.0 < logM200 < 15.0
    
    c_condition = 1.0 < c < 30.0  # corresponds to logc between 0 and 2
    
    inclination_condition = 0.0 < inclination <= 90.0
    
    # c between 1 and 100, which corresponds to logc between 0 and 2
    # logc_condition = 0.0 < dlogc < 2.0

    conditions = r1_condition and logM200_condition and c_condition and inclination_condition

    if not conditions:
        return -np.inf
    # else:
    #     return 0.0  # flat prior within bounds, -inf outside bounds

    # gaussian prior on logc
    lp_logc = logc_prior(theta)
    return lp_logc


def ln_prob(theta, bar, **kwargs):

    lp = ln_prior(theta, bar) 
    if not np.isfinite(lp):
        return -np.inf, None, None, None

    try:
        lk, cross_section, v_model, vrel = likelihood(theta, bar,**kwargs)
        # lk = likelihood(theta, **kwargs)
    except Exception as e:
        print(f"Likelihood calculation error for theta: {theta}: {e}")
        return -np.inf, None, None, None

    return lp + lk, cross_section, v_model, vrel


######################################################################
########################## EMCEE FUNCTION ############################
######################################################################
def run_emcee(
    theta,
    nwalkers=20,
    niter=100,
    baryonic_model=None,
    initial_volume=1e-3,
    p0=None,
    multiprocessing=False,
    save=False,
    filename=None,
    savepath=None,
    cores=4,
    relaxation_kwargs=None,
    emcee_kwargs=None,
):
    # Handle empty kwargs
    if relaxation_kwargs is None:
        relaxation_kwargs = {}
    if emcee_kwargs is None:
        emcee_kwargs = {}

    # Filename for saving
    if savepath is None:
        savepath = os.path.join(os.getcwd(), "MCMC_results/")
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        else:
            pass

    if save:
        if filename is None:
            filename = savepath + "test_run" + f"_nwalkers_{nwalkers}_niter_{niter}.h5"
        else:
            filename = filename

    if p0 is None:
        # Set up the initial conditions for the walkers
        initial = np.array(theta)
        ndim = len(theta)

        p0 = [initial + initial_volume * np.random.randn(ndim) for i in range(nwalkers)]
    else:
        ndim = p0.shape[1]

    # Boolean checking if the file already exists
    resume = os.path.exists(filename)
    if resume and save:
        print(f"Resuming from existing file: {filename}")

    dtypes = [
        ("cross-section", float),
        ("rotation-curve", float, (array_size,)),
        ("vrel", float),
    ]

    if multiprocessing:
        with Pool(processes=cores) as pool:
            print(f"Utilizing {cores} cores...")
            if save:
                print(filename)
                backend = emcee.backends.HDFBackend(filename)
                if resume:
                    print("Continuing previous run...")
                    p0 = None  # continue from the last position
                else:
                    print("Setting up new backend...")
                    backend.reset(nwalkers, ndim)
                sampler = emcee.EnsembleSampler(
                    nwalkers,
                    ndim,
                    ln_prob,
                    args=(baryonic_model,),
                    kwargs=relaxation_kwargs,
                    pool=pool,
                    blobs_dtype=dtypes,
                    backend=backend,
                    **emcee_kwargs,
                )
            else:
                sampler = emcee.EnsembleSampler(
                    nwalkers,
                    ndim,
                    ln_prob,
                    args=(baryonic_model,),
                    kwargs=relaxation_kwargs,
                    pool=pool,
                    blobs_dtype=dtypes,
                    **emcee_kwargs,
                )

            print("Running production with multiprocessing...")
            pos, prob, state, blobs = sampler.run_mcmc(
                p0, niter, progress=True, store=True
            )
    else:
        if save:
            print(filename)
            backend = emcee.backends.HDFBackend(filename)
            if resume:
                print("Continuing previous run...")
                p0 = None  # continue from the last position
            else:
                print("Setting up backend...")
                backend.reset(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                ln_prob,
                args=(baryonic_model,),
                kwargs=relaxation_kwargs,
                blobs_dtype=dtypes,
                backend=backend,
                **emcee_kwargs,
            )
        else:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                ln_prob,
                args=(baryonic_model,),
                kwargs=relaxation_kwargs,
                blobs_dtype=dtypes,
                **emcee_kwargs,
            )

        print("Running production...")
        pos, prob, state, blobs = sampler.run_mcmc(p0, niter, progress=True, store=True)

    return sampler, pos, prob, state, blobs


def main():

    start = t.time()

    # relaxation_kwargs = {"AC_prescription": "Cautun"}
    relaxation_kwargs = None
    NWALKERS = 32
    NITER = 5000
    cores = NWALKERS // 2
    
    print(f"Running MCMC for galaxy {GALAXYID} with {NWALKERS} walkers and {NITER} iterations.")
    
    # Generate the baryon potential for the given galaxy
    print(f"Generating baryon potential for galaxy {GALAXYID} with qdisk={QDISK}...")
    baryonic_model = BaryonicModel(GALAXYID, qdisk=QDISK, ngauss=8)
    initial_inclination = baryonic_model.incl
    initial_distance = baryonic_model.dist
    initial_log_upsilon_disk = np.log10(0.5)
    initial_log_upsilon_bulge = np.log10(0.7)

    initial_r1 = 10 # kpc
    initial_log_M200, _ = log_M200_weighted_mean(GALAXYID)
    initial_c = np.log10(c_MCR(10**initial_log_M200))  # from mass-concentration relation
    initial_q0 = 1.0
    
    initial_theta = [initial_r1, initial_log_M200, initial_c, initial_q0, initial_log_upsilon_disk, initial_log_upsilon_bulge, initial_inclination, initial_distance]  # [r1, logM200, logc, q0, log_upsilon_disk, log_upsilon_bulge]

    print(f"Initial theta: {initial_theta}")
    
    savepath = f'/scratch/adamso/sparc-results/mcmc-chains/'
    filename = savepath + f"{GALAXYID}_sph_mcmc_nw_{NWALKERS}_ni_{NITER}.h5"

    run_emcee(
        initial_theta,
        nwalkers=NWALKERS,
        niter=NITER,
        baryonic_model=baryonic_model,
        initial_volume=1e-2,
        multiprocessing=True,
        save=True,
        filename=filename,
        savepath=savepath,
        cores=cores,
        relaxation_kwargs=relaxation_kwargs,
    )

    end = t.time()
    print(f"Elapsed time: {end - start} seconds")


if __name__ == "__main__":
    main()
