import numpy as np
import pandas as pd
import os
import emcee
import jeans
from multiprocessing import Pool
import time as t

from readData import get_rc_data

GalaxyID = "F568-3"

# Load baryon fit parameters for potential calculation
baryon_fit_data = pd.read_csv(
    os.getcwd() + "/data/fit_data/hernquist_fit_results_gamma.csv"
)
baryon_data = baryon_fit_data[baryon_fit_data["GalaxyID"] == GalaxyID]
M_b_fit = baryon_data["M_b_fit"].values[0]  # solar masses
a_fit = baryon_data["a_fit"].values[0]  # kpc

# Load the rotation curve data
data, _, _ = get_rc_data(GalaxyID)
r_data = data["Rad"].values  # kpc
v_data = data["Vobs"].values  # km/s
v_err = data["errV"].values  # km/s


######################################################################
######################### HELPER FUNCTIONS ###########################
######################################################################
def transform(theta):
    r1, logM200, dlogc = theta
    M200 = 10**logM200
    c = c_MCR(M200) * (10**dlogc)

    return r1, M200, c


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


def dlogc_prior(dlog_c):
    """
    Use:
        Includes an error factor from the mass-concentration relation with a Gaussian prior on Delta(log(c)) centered on Delta(log(c))=0 with a width of 0.11 dex.
    Takes:
        delta log(c)
    Returns:
        The likelihood for delta log(c)
    """
    dlog_c_0 = 0
    dex_c = 0.11

    chi_squared_c = ((dlog_c - dlog_c_0) / dex_c) ** 2

    likelihood = -0.5 * chi_squared_c

    return likelihood


######################################################################
###################### HERNQUIST BARYON MODEL ########################
######################################################################
def phi_Hern_sph(M_b=1e10, a=10):
    """
    Spherical Hernquist potential.
    Takes:
        M_b : total baryonic mass [solar masses]
        a : scale radius [kpc]
    Returns:
        phi_Hern_sph : function that takes radius r [kpc] and returns potential [km^2/s^2]
    """
    G = 4.30091e-6  # [kpc * (km/s)^2 / solar mass]

    def potential(r, th):  # theta is a dummy variable for compatibility
        return -G * M_b / (r + a)

    return potential


######################################################################
########################## MCMC FUNCTIONS ############################
######################################################################
def model(theta, **kwargs):
    r1, M200, c = transform(theta)

    phi_b = phi_Hern_sph(M_b=M_b_fit, a=a_fit)

    profile = jeans.squashed(r1, M200, c, Phi_b=phi_b, verbose=False, **kwargs)

    return profile


def likelihood(theta, **kwargs):
    try:
        profile = model(theta, **kwargs)
    except Exception as e:
        print(f"Jeans model error: {e}")
        return -np.inf  # , None, None

    v_model = profile.V(r_data, Lmax=0)

    # Chi-squared
    chi_squared = np.sum(((v_data - v_model) / v_err) ** 2)

    log_likelihood = -0.5 * chi_squared

    # cross_section = profile.cross_section()

    return log_likelihood  # , cross_section, v_model


def ln_prior(theta):
    r1, logM200, dlogc = theta

    r1_condition = 0.0 < r1 < 500
    logM200_condition = (
        9.0 < logM200 < 15.0
    )  # uniform prior on logM200 between 1e9 and 1e15 Msun

    conditions = r1_condition and logM200_condition

    if not conditions:
        return -np.inf

    # gaussian prior on dlogc
    lp_dlogc = dlogc_prior(dlogc)
    return lp_dlogc


def ln_prob(theta, **kwargs):

    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf  # , None, None

    try:
        # lk, v_model, cross_section = likelihood(theta, **kwargs)
        lk = likelihood(theta, **kwargs)
    except Exception as e:
        print(f"Likelihood calculation error: {e}")
        return -np.inf  # , None, None

    return lp + lk  # , cross_section, v_model


######################################################################
########################## EMCEE FUNCTION ############################
######################################################################
def run_emcee(
    theta,
    nwalkers=20,
    niter=100,
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

    # array_size = len(r_data)  # corresponds to the number of radii in the rotation curve

    # dtypes = [
    #     ("cross-section", float),
    #     ("rotation-curve", float, (array_size,)),
    # ]

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
                    kwargs=relaxation_kwargs,
                    pool=pool,
                    # blobs_dtype=dtypes,
                    backend=backend,
                    **emcee_kwargs,
                )
            else:
                sampler = emcee.EnsembleSampler(
                    nwalkers,
                    ndim,
                    ln_prob,
                    kwargs=relaxation_kwargs,
                    pool=pool,
                    # blobs_dtype=dtypes,
                    **emcee_kwargs,
                )

            print("Running production with multiprocessing...")
            pos, prob, state = sampler.run_mcmc(p0, niter, progress=True, store=True)
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
                kwargs=relaxation_kwargs,
                # blobs_dtype=dtypes,
                backend=backend,
                **emcee_kwargs,
            )
        else:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                ln_prob,
                kwargs=relaxation_kwargs,
                # blobs_dtype=dtypes,
                **emcee_kwargs,
            )

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True, store=True)

    return sampler, pos, prob, state


def main():

    start = t.time()

    # relaxation_kwargs = {"AC_prescription": "Cautun"}
    relaxation_kwargs = None
    NWALKERS = 8
    NITER = 10

    initial_theta = [10, 11, 0.0]  # [r1, M200, dlogc]
    savepath = os.getcwd() + "/data/MCMC_results/"
    filename = savepath + f"{GalaxyID}_sph_mcmc_nw_{NWALKERS}_ni_{NITER}.h5"

    run_emcee(
        initial_theta,
        nwalkers=NWALKERS,
        niter=NITER,
        initial_volume=1e-2,
        multiprocessing=True,
        save=True,
        filename=filename,
        savepath=savepath,
        cores=NWALKERS,
        relaxation_kwargs=relaxation_kwargs,
    )

    end = t.time()
    print(f"Elapsed time: {end - start} seconds")


if __name__ == "__main__":
    main()
