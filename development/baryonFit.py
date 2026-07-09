import numpy as np
from readData import *

from scipy.optimize import curve_fit
from lmfit import Model
import time as t

# Global Constant
G = 4.30091e-6  # [kpc * (km/s)^2 / solar mass]


###################################################################
######################## HELPER FUNCTIONS #########################
###################################################################
# Hernquist model for circular velocity for fitting
def hernquist_model(r, log_M_b=10, log_a=1):
    """
    Circular velocity squared for Hernquist profile.
    Takes:
        r : radius [kpc] (or array of radii)
        log_M_b : log10 of total baryonic mass [solar masses]
        log_a : log10 of scale radius [kpc]
    Returns:
        Vc : circular velocity at radius r [km/s]
    """
    Mb = 10**log_M_b  # convert log10(M_b) to M_b in solar masses
    a = 10**log_a  # convert log10(a) to a in kpc
    vc = np.sqrt(G * Mb * r) / (r + a)
    return vc


# Dehnen model for circular velocity for fitting
def dehnen_model(r, log_M_b=10, log_a=1, gamma=1):
    """
    Circular velocity squared for Dehnen profile.
    Takes:
        r : radius [kpc] (or array of radii)
        log_M_b : log10 of total baryonic mass [solar masses]
        log_a : log10 of scale radius [kpc]
        gamma : inner slope parameter
    Returns:
        Vc : circular velocity at radius r [km/s]
    """
    Mb = 10**log_M_b  # convert log10(M_b) to M_b in solar masses
    a = 10**log_a  # convert log10(a) to a in kpc
    vcsq = G * Mb * ((r ** (2 - gamma)) / (r + a) ** (3 - gamma))
    return np.sqrt(vcsq)

def kuzmin_model(r, log_M_b=10, log_a=1):
    """
    Circular velocity squared for Kuzmin disk profile.
    Takes:
        r : radius [kpc] (or array of radii)
        log_M_b : log10 of total baryonic mass [solar masses]
        log_a : log10 of scale radius [kpc]
    Returns:
        Vc : circular velocity at radius r [km/s]
    """
    
    Mb = 10**log_M_b  # convert log10(M_b) to M_b in solar masses
    a = 10**log_a  # convert log10(a) to a in kpc
    vcsq = G * Mb * r**2 / (r**2 + a**2) ** (3 / 2)
    return np.sqrt(vcsq)


# Total baryonic circular velocity
def vc_disk_gas(data, gamma_star=0.5):
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
    vc_disk_sq = gamma_star * data["Vdisk"] ** 2
    vc_gas_sq = data["Vgas"] ** 2

    vc_baryons = np.sqrt(vc_disk_sq + vc_gas_sq)

    return radii, vc_baryons


def vc_bulge(data, gamma_bulge=0.7):
    """
    Bulge circular velocity.
    Takes:
        data : dataframe with columns 'Vbul'
        gamma_bulge : mass-to-light ratio scaling factor for bulge
    Returns:
        vc_bulge : bulge circular velocity [km/s]
        radii : radius [kpc]
    """
    radii = data["Rad"].values
    vc_bulge = np.sqrt(gamma_bulge * data["Vbul"] ** 2)

    return radii, vc_bulge


###################################################################
###################### HERNQUIST FIT CLASS ########################
###################################################################
class HernquistFit:
    def __init__(self, galaxy_id, gamma_star=0.5, gamma_bulge=0.7, method="lmfit"):
        self.galaxy_id = galaxy_id
        self.data, _, _ = get_rc_data(galaxy_id)
        self.gamma_star = gamma_star  # mass-to-light ratio scaling factor
        self.gamma_bulge = gamma_bulge  # bulge mass-to-light ratio scaling factor
        self.r_vals, self.vc_disk_gas = vc_disk_gas(self.data, self.gamma_star)
        _, self.vc_bulge = vc_bulge(self.data, self.gamma_bulge)
        self.method = method

        if method not in ["lmfit", "curve_fit"]:
            raise ValueError("HernquistFit 'method' supports 'lmfit' or 'curve_fit'")

        # Perform the fit upon initialization
        self.M_b_fit, self.a_fit, self.error = self.perform_fit(
            self.r_vals, self.vc_disk_gas
        )
        # Fit Bulge component separately if it exists
        if np.any(self.vc_bulge > 0):
            print("Bulge component detected, performing separate fit for bulge...")
            self.M_b_fit_bulge, self.a_fit_bulge, self.error_bulge = self.perform_fit(
                self.r_vals, self.vc_bulge
            )
        else:
            self.M_b_fit_bulge, self.a_fit_bulge, self.error_bulge = (
                np.nan,
                np.nan,
                np.nan,
            )

    def perform_fit(self, x, y):
        if self.method == "lmfit":

            try:
                # time the fit method
                start = t.time()
                # create the lmfit model
                model = Model(hernquist_model)
                params = model.make_params(
                    log_M_b=10.0, log_a=0.0
                )  # in log10 space for better convergence
                params["log_M_b"].set(
                    min=6.0, max=13.0
                )  # allow log_M_b to vary between 6 and 13
                params["log_a"].set(
                    min=-4.0, max=2.0
                )  # allow a to vary between 0.1 and 100 kpc
                r_eval = x
                y_eval = y
                # perform the fit
                self.result = model.fit(y_eval, params, r=r_eval)

                M_b_fit = (
                    10 ** self.result.params["log_M_b"].value
                )  # convert log10(M_b) to M_b in solar masses
                a_fit = 10 ** self.result.params["log_a"].value
                error = self.result.chisqr
            except Exception as e:
                print(f"Error fitting galaxy {self.galaxy_id} with lmfit: {e}")
                M_b_fit, a_fit = np.nan, np.nan
                error = np.nan

            print(f"Fitting with lmfit took {t.time() - start:.4f} seconds")

            print(f"Fitted parameters for galaxy {self.galaxy_id} using lmfit:")
            print(f"  M_b = {M_b_fit:.2e} solar masses")
            print(f"  a   = {a_fit:.2f} kpc")
            print(f"  chi-squared = {error:.4f}")

        elif self.method == "curve_fit":

            try:
                # time the fit method
                start = t.time()
                # perform the curve fit
                popt, pcov = curve_fit(hernquist_model, x, y, p0=[10.0, 0.0])

                print(f"Fitting with curve_fit took {t.time() - start:.4f} seconds")

                log_M_b_fit, log_a_fit = popt
                M_b_fit = 10**log_M_b_fit  # convert log10(M_b) to M_b in solar masses
                a_fit = 10**log_a_fit  # convert log10(a) to a in kpc
                error = pcov
            except RuntimeError as e:
                print(f"Error fitting galaxy {self.galaxy_id} with curve_fit: {e}")
                M_b_fit, a_fit = np.nan, np.nan
                error = None

            print(f"Fitted parameters for galaxy {self.galaxy_id} using curve_fit:")
            print(f"  M_b = {M_b_fit:.2e} solar masses")
            print(f"  a   = {a_fit:.2f} kpc")
            print(f"  Parameter covariance matrix:\n{error}")

        return M_b_fit, a_fit, error

    # Method to compute vc and phi using fitted parameters
    def vc(self, r, M_b=1e10, a=1):
        """
        Creates the Hernquist circular velocity model using the fitted parameters.
        Takes:
            r : radius [kpc] (or array of radii)
            M_b : total baryonic mass [solar masses]
            a : scale radius [kpc]
        Returns:
            vc : circular velocity at radius r [km/s] for best fit params
        """
        log_M_b = np.log10(M_b)
        log_a = np.log10(a)
        return hernquist_model(r, log_M_b=log_M_b, log_a=log_a)

    # Method to compute gravitational potential using fitted parameters in terms of (r,th) for the jeans model
    def phi(self, M_b=1e10, a=1):
        """
        Creates the Hernquist gravitational potential model using the fitted parameters.
        Takes:
            r : radius [kpc] (or array of radii)
            M_b : total baryonic mass [solar masses]
            a : scale radius [kpc]
        Returns:
            phi : gravitational potential at radius r [km^2/s^2] for best fit params
        """

        def phi_func(r, th):
            return -G * M_b / (r + a)

        return phi_func

    def density(self, r, M_b=1e10, a=1):
        """
        Creates the Hernquist density profile using the fitted parameters.
        Takes:
            r : radius [kpc] (or array of radii)
        Returns:
            rho : density at radius r [solar masses/kpc^3] for best fit params
        """
        numerator = M_b / (2 * np.pi * a**3)
        denominator = (r / a) * (1 + (r / a)) ** 3

        return np.divide(numerator, denominator)

    def mass_enclosed(self, r, M_b=1e10, a=1):
        """
        Creates the Hernquist enclosed mass profile using the fitted parameters.
        Takes:
            r : radius [kpc] (or array of radii)
        Returns:
            M_enc : enclosed mass at radius r [solar masses] for best fit params
        """
        numerator = M_b * r**2
        denominator = (r + a) ** 2

        return np.divide(numerator, denominator)
    
###################################################################
###################### HERNQUIST FIT CLASS ########################
###################################################################
class KuzminFit:
    def __init__(self, galaxy_id, gamma_star=0.5, gamma_bulge=0.7, method="lmfit"):
        """
        Fits a Kuzmin rotation curve to the baryonic rotation curve of a galaxy.
        Fits a Hernquist rotation curve to the bulge rotation curve of a galaxy if it exists.
        """
        self.galaxy_id = galaxy_id
        self.data, _, _ = get_rc_data(galaxy_id)
        self.gamma_star = gamma_star  # mass-to-light ratio scaling factor
        self.gamma_bulge = gamma_bulge  # bulge mass-to-light ratio scaling factor
        self.r_vals, self.vc_disk_gas = vc_disk_gas(self.data, self.gamma_star)
        _, self.vc_bulge = vc_bulge(self.data, self.gamma_bulge)
        self.method = method

        if method not in ["lmfit", "curve_fit"]:
            raise ValueError("KuzminFit 'method' supports 'lmfit' or 'curve_fit'")

        # Perform the fit upon initialization
        self.M_b_fit, self.a_fit, self.error = self.perform_fit(
            self.r_vals, self.vc_disk_gas
        )
        # Fit Bulge component separately if it exists
        if np.any(self.vc_bulge > 0):
            print("Bulge component detected, performing separate fit for bulge...")
            self.M_b_fit_bulge, self.a_fit_bulge, self.error_bulge = self.perform_fit_hernquist(
                self.r_vals, self.vc_bulge
            )
        else:
            self.M_b_fit_bulge, self.a_fit_bulge, self.error_bulge = (
                np.nan,
                np.nan,
                np.nan,
            )

    def perform_fit(self, x, y):
        if self.method == "lmfit":

            try:
                # time the fit method
                start = t.time()
                # create the lmfit model
                model = Model(kuzmin_model)
                params = model.make_params(
                    log_M_b=10.0, log_a=0.0
                )  # in log10 space for better convergence
                params["log_M_b"].set(
                    min=6.0, max=13.0
                )  # allow log_M_b to vary between 6 and 13
                params["log_a"].set(
                    min=-4.0, max=2.0
                )  # allow a to vary between 0.1 and 100 kpc
                r_eval = x
                y_eval = y
                # perform the fit
                self.result = model.fit(y_eval, params, r=r_eval)

                M_b_fit = (
                    10 ** self.result.params["log_M_b"].value
                )  # convert log10(M_b) to M_b in solar masses
                a_fit = 10 ** self.result.params["log_a"].value
                error = self.result.chisqr
            except Exception as e:
                print(f"Error fitting galaxy {self.galaxy_id} with lmfit: {e}")
                M_b_fit, a_fit = np.nan, np.nan
                error = np.nan

            print(f"Fitting with lmfit took {t.time() - start:.4f} seconds")

            print(f"Fitted parameters for galaxy {self.galaxy_id} using lmfit:")
            print(f"  M_b = {M_b_fit:.2e} solar masses")
            print(f"  a   = {a_fit:.2f} kpc")
            print(f"  chi-squared = {error:.4f}")

        elif self.method == "curve_fit":

            try:
                # time the fit method
                start = t.time()
                # perform the curve fit
                popt, pcov = curve_fit(kuzmin_model, x, y, p0=[10.0, 0.0])

                print(f"Fitting with curve_fit took {t.time() - start:.4f} seconds")

                log_M_b_fit, log_a_fit = popt
                M_b_fit = 10**log_M_b_fit  # convert log10(M_b) to M_b in solar masses
                a_fit = 10**log_a_fit  # convert log10(a) to a in kpc
                error = pcov
            except RuntimeError as e:
                print(f"Error fitting galaxy {self.galaxy_id} with curve_fit: {e}")
                M_b_fit, a_fit = np.nan, np.nan
                error = None

            print(f"Fitted parameters for galaxy {self.galaxy_id} using curve_fit:")
            print(f"  M_b = {M_b_fit:.2e} solar masses")
            print(f"  a   = {a_fit:.2f} kpc")
            print(f"  Parameter covariance matrix:\n{error}")
            
        return M_b_fit, a_fit, error
            
    def perform_fit_hernquist(self, x, y):
        if self.method == "lmfit":

            try:
                # time the fit method
                start = t.time()
                # create the lmfit model
                model = Model(hernquist_model)
                params = model.make_params(
                    log_M_b=10.0, log_a=0.0
                )  # in log10 space for better convergence
                params["log_M_b"].set(
                    min=6.0, max=13.0
                )  # allow log_M_b to vary between 6 and 13
                params["log_a"].set(
                    min=-4.0, max=2.0
                )  # allow a to vary between 0.1 and 100 kpc
                r_eval = x
                y_eval = y
                # perform the fit
                self.result = model.fit(y_eval, params, r=r_eval)

                M_b_fit = (
                    10 ** self.result.params["log_M_b"].value
                )  # convert log10(M_b) to M_b in solar masses
                a_fit = 10 ** self.result.params["log_a"].value
                error = self.result.chisqr
            except Exception as e:
                print(f"Error fitting galaxy {self.galaxy_id} with lmfit: {e}")
                M_b_fit, a_fit = np.nan, np.nan
                error = np.nan

            print(f"Fitting with lmfit took {t.time() - start:.4f} seconds")

            print(f"Fitted parameters for galaxy {self.galaxy_id} using lmfit:")
            print(f"  M_b = {M_b_fit:.2e} solar masses")
            print(f"  a   = {a_fit:.2f} kpc")
            print(f"  chi-squared = {error:.4f}")

        elif self.method == "curve_fit":

            try:
                # time the fit method
                start = t.time()
                # perform the curve fit
                popt, pcov = curve_fit(hernquist_model, x, y, p0=[10.0, 0.0])

                print(f"Fitting with curve_fit took {t.time() - start:.4f} seconds")

                log_M_b_fit, log_a_fit = popt
                M_b_fit = 10**log_M_b_fit  # convert log10(M_b) to M_b in solar masses
                a_fit = 10**log_a_fit  # convert log10(a) to a in kpc
                error = pcov
            except RuntimeError as e:
                print(f"Error fitting galaxy {self.galaxy_id} with curve_fit: {e}")
                M_b_fit, a_fit = np.nan, np.nan
                error = None

            print(f"Fitted parameters for galaxy {self.galaxy_id} using curve_fit:")
            print(f"  M_b = {M_b_fit:.2e} solar masses")
            print(f"  a   = {a_fit:.2f} kpc")
            print(f"  Parameter covariance matrix:\n{error}")

        return M_b_fit, a_fit, error

    # Method to compute vc and phi using fitted parameters
    def vc(self, r, M_b=1e10, a=1):
        """
        Creates the Kuzmin circular velocity model using the fitted parameters.
        Takes:
            r : radius [kpc] (or array of radii)
            M_b : total baryonic mass [solar masses]
            a : scale radius [kpc]
        Returns:
            vc : circular velocity at radius r [km/s] for best fit params
        """
        log_M_b = np.log10(M_b)
        log_a = np.log10(a)
        return kuzmin_model(r, log_M_b=log_M_b, log_a=log_a)

    # # Method to compute gravitational potential using fitted parameters in terms of (r,th) for the jeans model
    # def phi(self, M_b=1e10, a=1):
    #     """
    #     Creates the Kuzmin gravitational potential model using the fitted parameters.
    #     Takes:
    #         r : radius [kpc] (or array of radii)
    #         M_b : total baryonic mass [solar masses]
    #         a : scale radius [kpc]
    #     Returns:
    #         phi : gravitational potential at radius r [km^2/s^2] for best fit params
    #     """

    #     def phi_func(r, th):
    #         return -G * M_b / (r + a)

    #     return phi_func

    # def density(self, r, M_b=1e10, a=1):
    #     """
    #     Creates the Kuzmin density profile using the fitted parameters.
    #     Takes:
    #         r : radius [kpc] (or array of radii)
    #     Returns:
    #         rho : density at radius r [solar masses/kpc^3] for best fit params
    #     """
    #     numerator = M_b / (2 * np.pi * a**3)
    #     denominator = (r / a) * (1 + (r / a)) ** 3

    #     return np.divide(numerator, denominator)

    # def mass_enclosed(self, r, M_b=1e10, a=1):
    #     """
    #     Creates the Hernquist enclosed mass profile using the fitted parameters.
    #     Takes:
    #         r : radius [kpc] (or array of radii)
    #     Returns:
    #         M_enc : enclosed mass at radius r [solar masses] for best fit params
    #     """
    #     numerator = M_b * r**2
    #     denominator = (r + a) ** 2

    #     return np.divide(numerator, denominator)



###################################################################
####################### DEHNEN FIT CLASS ##########################
###################################################################
class DehnenFit:
    def __init__(self, galaxy_id, gamma_star=0.5, gamma_bulge=0.7, method="lmfit"):
        self.galaxy_id = galaxy_id
        self.data, _, _ = get_rc_data(galaxy_id)
        self.gamma_star = gamma_star  # mass-to-light ratio scaling factor
        self.gamma_bulge = gamma_bulge  # bulge mass-to-light ratio scaling factor
        self.r_vals, self.vc_disk_gas = vc_disk_gas(self.data, self.gamma_star)
        _, self.vc_bulge = vc_bulge(self.data, self.gamma_bulge)
        self.method = method

        if method not in ["lmfit", "curve_fit"]:
            raise ValueError("DehnenFit 'method' supports 'lmfit' or 'curve_fit'")

        # Perform the fit upon initialization
        self.M_b_fit, self.a_fit, self.gamma_fit, self.error = self.perform_fit(
            self.r_vals, self.vc_disk_gas
        )

        # Fit Bulge component separately if it exists
        if np.any(self.vc_bulge > 0):
            print("Bulge component detected, performing separate fit for bulge...")
            (
                self.M_b_fit_bulge,
                self.a_fit_bulge,
                self.gamma_fit_bulge,
                self.error_bulge,
            ) = self.perform_fit(self.r_vals, self.vc_bulge)
        else:
            (
                self.M_b_fit_bulge,
                self.a_fit_bulge,
                self.gamma_fit_bulge,
                self.error_bulge,
            ) = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

    def perform_fit(self, x, y):
        if self.method == "lmfit":

            try:
                # time the fit method
                start = t.time()
                # create the lmfit model
                model = Model(dehnen_model)
                params = model.make_params(
                    log_M_b=10.0, log_a=0.0, gamma=1.0
                )  # in log10 space for better convergence
                params["gamma"].set(
                    min=1e-4, max=2.0
                )  # allow gamma to vary between -3 and 3
                params["log_M_b"].set(
                    min=6.0, max=13.0
                )  # allow log_M_b to vary between 6 and 13
                params["log_a"].set(
                    min=-4.0, max=2.0
                )  # allow a to vary between 0.1 and 100 kpc
                r_eval = x
                y_eval = y
                # perform the fit
                self.result = model.fit(y_eval, params, r=r_eval)

                M_b_fit = (
                    10 ** self.result.params["log_M_b"].value
                )  # convert log10(M_b) to M_b in solar masses
                a_fit = 10 ** self.result.params["log_a"].value
                gamma_fit = self.result.params["gamma"].value
                error = self.result.chisqr
            except Exception as e:
                print(f"Error fitting galaxy {self.galaxy_id} with lmfit: {e}")
                M_b_fit, a_fit, gamma_fit = np.nan, np.nan, np.nan
                error = np.nan

            print(f"Fitting with lmfit took {t.time() - start:.4f} seconds")

            print(f"Fitted parameters for galaxy {self.galaxy_id} using lmfit:")
            print(f"  M_b = {M_b_fit:.2e} solar masses")
            print(f"  a   = {a_fit:.2f} kpc")
            print(f"  gamma = {gamma_fit:.2f}")
            print(f"  chi-squared = {error:.4f}")

        elif self.method == "curve_fit":

            try:
                # time the fit method
                start = t.time()
                # perform the curve fit
                popt, pcov = curve_fit(
                    dehnen_model,
                    x,
                    y,
                    p0=[10.0, 0.0, 1.0],
                    bounds=([6.0, -4.0, 0.0], [13.0, 2.0, 2.0]),
                )

                print(f"Fitting with curve_fit took {t.time() - start:.4f} seconds")

                log_M_b_fit, log_a_fit, gamma_fit = popt
                M_b_fit = 10**log_M_b_fit  # convert log10(M_b) to M_b in solar masses
                a_fit = 10**log_a_fit  # convert log10(a) to a in kpc
                error = pcov
            except RuntimeError as e:
                print(f"Error fitting galaxy {self.galaxy_id} with curve_fit: {e}")
                M_b_fit, a_fit, gamma_fit = np.nan, np.nan, np.nan
                error = None

            print(f"Fitted parameters for galaxy {self.galaxy_id} using curve_fit:")
            print(f"  M_b = {M_b_fit:.2e} solar masses")
            print(f"  a   = {a_fit:.2f} kpc")
            print(f"  gamma = {gamma_fit:.2f}")
            print(f"  Parameter covariance matrix:\n{error}")

        return M_b_fit, a_fit, gamma_fit, error

    # Method to compute vc and phi using fitted parameters
    def vc(self, r, M_b=1e0, a=1, gamma=1):
        """
        Creates the Hernquist circular velocity model using the fitted parameters.
        Takes:
            r : radius [kpc] (or array of radii)
            M_b : total baryonic mass [solar masses]
            a : scale radius [kpc]
        Returns:
            vc : circular velocity at radius r [km/s] for best fit params
        """
        log_M_b = np.log10(M_b)
        log_a = np.log10(a)
        return dehnen_model(r, log_M_b=log_M_b, log_a=log_a, gamma=gamma)

    # Method to compute gravitational potential using fitted parameters in terms of (r,th) for the jeans model
    def phi(self, M_b=1e10, a=1, gamma=1):
        """
        Creates the Hernquist gravitational potential model using the fitted parameters.
        Takes:
            r : radius [kpc] (or array of radii)
            M_b : total baryonic mass [solar masses]
            a : scale radius [kpc]
        Returns:
            phi : gravitational potential at radius r [km^2/s^2] for best fit params
        """

        def phi_func(r, th):
            if np.isclose(gamma, 2.0):
                # Jaffe limit
                return -(G * M_b / a) * np.log(r / (r + a))
            else:
                prefactor = -(G * M_b) / (a * (2 - gamma))
                return prefactor * (1 - (r / (r + a)) ** (2 - gamma))

        return phi_func

    def density(self, r, M_b=1e10, a=1, gamma=1):
        """
        Creates the Hernquist density profile using the fitted parameters.
        Takes:
            r : radius [kpc] (or array of radii)
        Returns:
            rho : density at radius r [solar masses/kpc^3] for best fit params
        """

        numerator = (3 - gamma) * M_b * a
        denominator = 4 * np.pi * (r**gamma * (r + a) ** (4 - gamma))

        return np.divide(numerator, denominator)

    def mass_enclosed(self, r, M_b=1e10, a=1, gamma=1):
        """
        Creates the Hernquist enclosed mass profile using the fitted parameters.
        Takes:
            r : radius [kpc] (or array of radii)
        Returns:
            M_enc : enclosed mass at radius r [solar masses] for best fit params
        """
        return M_b * (r / (r + a)) ** (3 - gamma)
