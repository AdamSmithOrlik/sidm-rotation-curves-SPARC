import numpy as np
from readData import *

from scipy.optimize import curve_fit
from lmfit import Model
import time as t


###################################################################
######################## HELPER FUNCTIONS #########################
###################################################################
# Hernquist model for circular velocity
def hernquist_model(r, M_b=1e10, a=1):
    """
    Circular velocity squared for Hernquist profile.
    Takes:
        r : radius [kpc] (or array of radii)
        M_b : total baryonic mass [solar masses]
        a : scale radius [kpc]
    Returns:
        Vc : circular velocity at radius r [km/s]
    """
    G = 4.30091e-6  # [kpc * (km/s)^2/ solar mass]
    vc = np.sqrt(G * M_b * r) / (r + a)
    return vc


# Total baryonic circular velocity
def vc_baryons(data, gamma_star=1.0):
    """
    Total baryonic circular velocity.
    Takes:
        data : dataframe with columns 'Vdisk', 'Vbul', 'Vgas'
        gamma_star : mass-to-light ratio scaling factor for disk
    Returns:
        vc_baryons : total baryonic circular velocity [km/s]
        radii : radius [kpc]
    """
    radii = data["Rad"].values
    gamma_bulge = 1.4 * gamma_star  # bulge mass-to-light ratio scaling factor
    vc_disk_sq = gamma_star * data["Vdisk"] ** 2
    vc_bulge_sq = gamma_bulge * data["Vbul"] ** 2
    vc_gas_sq = data["Vgas"] ** 2

    vc_baryons = np.sqrt(vc_disk_sq + vc_bulge_sq + vc_gas_sq)

    return radii, vc_baryons


###################################################################
###################### HERNQUIST FIT CLASS ########################
###################################################################
class HernquistFit:
    def __init__(self, galaxy_id, gamma_star=1.0, method="lmfit"):
        self.galaxy_id = galaxy_id
        self.data, _, _ = get_rc_data(galaxy_id)
        self.gamma_star = gamma_star  # mass-to-light ratio scaling factor
        self.r_vals, self.vc_baryons = vc_baryons(self.data, self.gamma_star)
        self.method = method

        if method not in ["lmfit", "curve_fit"]:
            raise ValueError("HernquistFit 'method' supports 'lmfit' or 'curve_fit'")

        # Perform the fit upon initialization
        self.M_b_fit, self.a_fit = self.perform_fit()

    def perform_fit(self):
        if self.method == "lmfit":

            try:
                # time the fit method
                start = t.time()
                # create the lmfit model
                self.model = Model(hernquist_model)
                params = self.model.make_params(M_b=1e10, a=1)
                r_eval = self.r_vals
                y_eval = self.vc_baryons
                # perform the fit
                self.result = self.model.fit(y_eval, params, r=r_eval)
            except Exception as e:
                print(f"Error fitting galaxy {self.galaxy_id} with lmfit: {e}")
                self.M_b_fit, self.a_fit = np.nan, np.nan
                self.chisqr = np.nan

            print(f"Fitting with lmfit took {t.time() - start:.4f} seconds")

            M_b_fit = self.result.params["M_b"].value
            a_fit = self.result.params["a"].value
            chisqr = self.result.chisqr

            print(f"Fitted parameters for galaxy {self.galaxy_id} using lmfit:")
            print(f"  M_b = {M_b_fit:.2e} solar masses")
            print(f"  a   = {a_fit:.2f} kpc")
            print(f"  chi-squared = {chisqr:.4f}")

        elif self.method == "curve_fit":

            try:
                # time the fit method
                start = t.time()
                # perform the curve fit
                popt, pcov = curve_fit(
                    hernquist_model, self.r_vals, self.vc_baryons, p0=[1e10, 1]
                )

                print(f"Fitting with curve_fit took {t.time() - start:.4f} seconds")

                M_b_fit, a_fit = popt
                self.pcov = pcov
            except RuntimeError as e:
                print(f"Error fitting galaxy {self.galaxy_id} with curve_fit: {e}")
                self.M_b_fit, self.a_fit = np.nan, np.nan
                self.pcov = None

            print(f"Fitted parameters for galaxy {self.galaxy_id} using curve_fit:")
            print(f"  M_b = {M_b_fit:.2e} solar masses")
            print(f"  a   = {a_fit:.2f} kpc")
            print(f"  Parameter covariance matrix:\n{self.pcov}")

        return M_b_fit, a_fit

    # Method to compute vc and phi using fitted parameters
    def vc(self, r):
        """
        Creates the Hernquist circular velocity model using the fitted parameters.
        Takes:
            r : radius [kpc] (or array of radii)
        Returns:
            vc : circular velocity at radius r [km/s] for best fit params
        """
        return hernquist_model(r, M_b=self.M_b_fit, a=self.a_fit)

    # Method to compute gravitational potential using fitted parameters in terms of (r,th) for the jeans model
    def phi(self, r):
        """
        Creates the Hernquist gravitational potential model using the fitted parameters.
        Takes:
            r : radius [kpc] (or array of radii)
        Returns:
            phi : gravitational potential at radius r [km^2/s^2] for best fit params
        """
        G = 4.30091e-6  # [kpc * (km/s)^2/ solar mass]

        def phi_func(r, th):
            return -G * self.M_b_fit / (r + self.a_fit)

        return phi_func

    def density(self, r):
        """
        Creates the Hernquist density profile using the fitted parameters.
        Takes:
            r : radius [kpc] (or array of radii)
        Returns:
            rho : density at radius r [solar masses/kpc^3] for best fit params
        """
        numerator = self.M_b_fit / (2 * np.pi * self.a_fit**3)
        denominator = (r / self.a_fit) * (1 + (r / self.a_fit)) ** 3

        return np.divide(numerator, denominator)

    def mass_enclosed(self, r):
        """
        Creates the Hernquist enclosed mass profile using the fitted parameters.
        Takes:
            r : radius [kpc] (or array of radii)
        Returns:
            M_enc : enclosed mass at radius r [solar masses] for best fit params
        """
        numerator = self.M_b_fit * r**2
        denominator = (r + self.a_fit) ** 2

        return np.divide(numerator, denominator)
