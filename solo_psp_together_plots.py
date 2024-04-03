import os
import numpy as np
import pandas as pd
import glob
from scipy import interpolate
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyreadr
import datetime as dt

from conversions import jd2date, date2jd
from ephemeris import get_approaches
from load_data import encounter_group

from paths import psp_ephemeris_file
from paths import figures_location

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600


class InlaResult:
    """
    The container for INLA results coming in the form of .RData file 
    produced by with "psp_inla_fitting.R" containing
    sampels and prior, posterior functions.

    The usual structure of such .RData file:
    
    odict_keys(['sample_l_a', 'sample_l_b', 'sample_v_b_r', 
                'sample_e_v', 'sample_e_b_r', 'sample_shield_miss_rate',
                'fx_l_a', 'fy_l_a', 
                'fx_l_b', 'fy_l_b', 
                'fx_v_b_r', 'fy_v_b_r', 
                'fx_e_v', 'fy_e_v', 
                'fx_e_b_r', 'fy_e_b_r',
                'fx_shield_miss_rate', 'fy_shield_miss_rate',
                'px_l_a', 'py_l_a', 
                'px_l_b', 'py_l_b', 
                'px_v_b_r', 'py_v_b_r', 
                'px_e_v', 'py_e_v', 
                'px_e_b_r', 'py_e_b_r', 
                'px_shield_miss_rate', 'py_shield_miss_rate',
                'model_definition', 
                'mydata'])

    """
    def __init__(self,
                 datafile,
                 solo_csv_readable,
                 psp_csv_readable,
                 careful=True):
        """
        The creator for the InlaResult. Loads the RData output and 
        the csv input. 

        Parameters
        ----------
        datafile : str
            The path to the RData file with the INLA resutl.
        input_csv_readable : str, optional
            The path to the readable CSV data file (input). 
            The default is paths.readable_data. 
        careful : bool, optional
            Whether to raise exception if the INLA used function 
            is different from the expected one.

        Returns
        -------
        None.

        """
        self.contents = pyreadr.read_r(datafile)
        self.sample_size = len(self.contents["sample_l_a"])
        self.atts = ["l_a",
                     "l_b",
                     "v_b_r",
                     "e_v",
                     "e_b_r",
                     "shield_miss_rate"]
        self.solo_input_df = pd.read_csv(solo_csv_readable)
        self.psp_input_df = pd.read_csv(psp_csv_readable)
        self.careful=careful

        """
        solo_input_df:
        ['Julian date', 'Fluxes [/day]', 'Radial velocity [km/s]',
        'Tangential velocity [km/s]', 'Radial distance [au]',
        'Detection time [hours]', 'Velocity phase angle [deg]',
        'Velocity inclination [deg]', 'V_X (HAE) [km/s]', 'V_Y (HAE) [km/s]',
        'V_Z (HAE) [km/s]']
            
        psp_input_df:
        ['Julian date', 'Count corrected [/day]', 'Radial velocity [km/s]',
        'Tangential velocity [km/s]', 'Radial distance [au]',
        'Detection time [hours]', 'Velocity phase angle [deg]',
        'Velocity inclination [deg]', 'V_X (HAE) [km/s]', 'V_Y (HAE) [km/s]',
        'V_Z (HAE) [km/s]', 'Deviation angle [deg]', 'Area front [m^2]',
        'Area side [m^2]']
        """


    def inla_function(self):
        """
        Extracts the function definition if the RGeneric model used,
        this was obtained as deparse(three_component_model).

        Returns
        -------
        contents["model_definition"] : str
            The original INLA function.

        """
        return self.contents["model_definition"]


    def rate_function(self):
        """
        The function that checks what was the function used in R. If it is 
        a usual rate function, the tranlation is returned. 

        Parameters
        ----------


        Raises
        ------
        Warning
            If the R function does not match the expectation and the 
            self.careful is set to False.

        Exception
            If the R function does not match the expectation and the 
            self.careful is set to True.

        Returns
        -------
        usual_rate : function
            The rate function that is usually used.

        """

        first_row = np.argmax(self.inla_function()[
                                                    "model_definition"
                                                  ].str.find(
                                                    "rate <- function(v_sc_r"
                                                            )>0)

        last_row = np.argmax(self.inla_function()[
                                                    "model_definition"
                                                  ].str.find(
                                                        "return(hourly_rate)"
                                                            )>0)

        r_function = np.array(self.inla_function()[first_row:last_row+1])
        r_function_list = [row[0] for row in r_function]

        usual_r_function_list = [
           '    rate <- function(v_sc_r = feed_c[1], v_sc_t = feed_c[2], ',
           '        r_sc = feed_c[3], area_front = feed_c[4], area_side = feed_c[5], ',
           '        heat_shield = feed_c[6], l_a = feed_h[1], l_b = feed_h[2], ',
           '        v_b_r = feed_h[3], e_v = feed_h[4], e_b_r = feed_h[5], ',
           '        shield_miss_rate = feed_h[6], e_a_r = -1.3, v_b_a = 9, ',
           '        v_earth_a = 0) {',
           '        deg2rad <- function(deg) {',
           '            rad = deg/180 * pi',
           '            return(rad)',
           '        }',
           '        ksi = -2 - (-1.5 - e_b_r)',
           '        r_factor = r_sc/1',
           '        v_factor = (((v_sc_r - (v_b_r * (r_factor^ksi)))^2 + ',
           '            (v_sc_t - (v_b_a * (r_factor^(-1))))^2)^0.5)/(((v_b_r)^2 + ',
           '            (v_earth_a - v_b_a)^2)^0.5)',
           '        radial_impact_velocity = -1 * (v_sc_r - (v_b_r * (r_factor^ksi)))',
           '        azimuthal_impact_velocity = abs(v_sc_t - (v_b_a * (r_factor^(-1))))',
           '        impact_angle = atan(azimuthal_impact_velocity/radial_impact_velocity)',
           '        frontside = radial_impact_velocity > 0',
           '        backside = (frontside != TRUE)',
           '        area = (frontside * (1 - shield_miss_rate * heat_shield) * ',
           '            area_front * cos(impact_angle) + backside * 1 * area_front * ',
           '            cos(impact_angle) + area_side * sin(abs(impact_angle)))',
           '        L_b = l_b * area * (v_factor)^(e_v + 1) * (r_factor)^(-1.5 - ',
           '            e_b_r)',
           '        ksi = -2 - e_a_r',
           '        r_factor = r_sc/1',
           '        v_a_a = 29.8 * (r_factor^(-0.5))',
           '        v_factor = (((v_sc_r)^2 + (v_sc_t - v_a_a)^2)^0.5)/abs(v_earth_a - ',
           '            v_a_a)',
           '        radial_impact_velocity = -1 * (v_sc_r)',
           '        azimuthal_impact_velocity = abs(v_sc_t - v_a_a)',
           '        impact_angle = atan(azimuthal_impact_velocity/radial_impact_velocity)',
           '        frontside = radial_impact_velocity > 0',
           '        backside = (frontside != TRUE)',
           '        area = (frontside * (1 - shield_miss_rate * heat_shield) * ',
           '            area_front * cos(impact_angle) + backside * 1 * area_front * ',
           '            cos(impact_angle) + area_side * sin(abs(impact_angle)))',
           '        L_a = l_a * area * (v_factor)^(e_v + 1) * (r_factor)^e_a_r',
           '        hourly_rate = 3600 * (L_b + L_a)',
           '        return(hourly_rate)']

        alternative_r_function_list = [
            '    rate <- function(v_sc_r = feed_c[1], v_sc_t = feed_c[2], ',
            '        r_sc = feed_c[3], area_front = feed_c[4], area_side = feed_c[5], ',
            '        heat_shield = feed_c[6], l_a = feed_h[1], l_b = feed_h[2], ',
            '        v_b_r = feed_h[3], e_v = feed_h[4], e_b_r = feed_h[5], ',
            '        shield_miss_rate = feed_h[6], e_a_r = -1.3, v_b_a = 9, ',
            '        v_earth_a = 0) {',
            '        deg2rad <- function(deg) {',
            '            rad = deg/180 * pi',
            '            return(rad)',
            '        }',
            '        ksi = -2 - (-1.5 - e_b_r)',
            '        r_factor = r_sc/1',
            '        v_factor = (((v_sc_r - (v_b_r * (r_factor^ksi)))^2 + ',
            '            (v_sc_t - (v_b_a * (r_factor^(-1))))^2)^0.5)/(((v_b_r)^2 + ',
            '            (v_earth_a - v_b_a)^2)^0.5)',
            '        radial_impact_velocity = -1 * (v_sc_r - (v_b_r * (r_factor^ksi)))',
            '        azimuthal_impact_velocity = abs(v_sc_t - (v_b_a * (r_factor^(-1))))',
            '        impact_angle = atan(azimuthal_impact_velocity/radial_impact_velocity)',
            '        frontside = radial_impact_velocity > 0',
            '        backside = (frontside != TRUE)',
            '        area = (frontside * 1 * area_front * cos(impact_angle) + ',
            '            backside * 1 * area_front * cos(impact_angle) + area_side * ',
            '            sin(abs(impact_angle))) * (1 - shield_miss_rate * ',
            '            heat_shield)',
            '        L_b = l_b * area * (v_factor)^(e_v + 1) * (r_factor)^(-1.5 - ',
            '            e_b_r)',
            '        ksi = -2 - e_a_r',
            '        r_factor = r_sc/1',
            '        v_a_a = 29.8 * (r_factor^(-0.5))',
            '        v_factor = (((v_sc_r)^2 + (v_sc_t - v_a_a)^2)^0.5)/abs(v_earth_a - ',
            '            v_a_a)',
            '        radial_impact_velocity = -1 * (v_sc_r)',
            '        azimuthal_impact_velocity = abs(v_sc_t - v_a_a)',
            '        impact_angle = atan(azimuthal_impact_velocity/radial_impact_velocity)',
            '        frontside = radial_impact_velocity > 0',
            '        backside = (frontside != TRUE)',
            '        area = (frontside * 1 * area_front * cos(impact_angle) + ',
            '            backside * 1 * area_front * cos(impact_angle) + area_side * ',
            '            sin(abs(impact_angle))) * (1 - shield_miss_rate * ',
            '            heat_shield)',
            '        L_a = l_a * area * (v_factor)^(e_v + 1) * (r_factor)^e_a_r',
            '        hourly_rate = 3600 * (L_b + L_a)',
            '        return(hourly_rate)']

        if r_function_list == usual_r_function_list:
            print("""the usual rate function is applied""")
            def usual_rate(v_sc_r, v_sc_t,
                           r_sc,
                           area_front, area_side,
                           heat_shield,
                           l_a, l_b,
                           v_b_r,
                           e_v,
                           e_b_r,
                           shield_miss_rate,
                           e_a_r=-1.3,
                           v_b_a=9, v_earth_a=0):
        
                def deg2rad(deg):
                    return deg / 180 * np.pi

                #beta meteoroid contribution
                ksi = -2 - (-1.5-e_b_r)
                r_factor = r_sc/1
                v_factor = ( (
                    ( v_sc_r - ( v_b_r*(r_factor**ksi)  ) )**2
                    + ( v_sc_t - ( v_b_a*(r_factor**(-1)) ) )**2
                  )**0.5
                  ) / ( (
                    ( v_b_r )**2
                    + ( v_earth_a - v_b_a )**2
                  )**0.5
                  )
                radial_impact_velocity = -1* (v_sc_r-(v_b_r*(r_factor**ksi)))
                  #positive is on the heatshield, negative on the tail
                azimuthal_impact_velocity = np.abs(v_sc_t-(v_b_a*(r_factor**(-1))))
                  #always positive, RHS vs LHS plays no role
                impact_angle = np.arctan( azimuthal_impact_velocity
                                    / radial_impact_velocity )
                
                frontside = radial_impact_velocity > 0
                backside = (frontside != True)
                area = (   frontside   * (1 - shield_miss_rate * heat_shield) 
                              * area_front * np.cos(impact_angle)
                           + backside  * 1 
                              * area_front * np.cos(impact_angle)
                           + area_side * np.sin(np.abs(impact_angle)) )
                
                L_b = l_b * area * (v_factor)**(e_v+1) * (r_factor)**(-1.5-e_b_r)
                
                #bound dust contribution
                ksi = -2 - e_a_r
                r_factor = r_sc/1
                v_a_a = 29.8*(r_factor**(-0.5))
                v_factor = ( (
                    ( v_sc_r )**2
                    + ( v_sc_t - v_a_a )**2
                  )**0.5
                  ) / np.abs( v_earth_a - v_a_a )
                radial_impact_velocity = -1* ( v_sc_r ) 
                  #positive is on the heatshield, negative on the tail
                azimuthal_impact_velocity = np.abs( v_sc_t - v_a_a )
                  #always positive, RHS vs LHS plays no role
                impact_angle = np.arctan( azimuthal_impact_velocity
                                          / radial_impact_velocity )
                
                frontside = radial_impact_velocity > 0
                backside = (frontside != True)
                area = (   frontside   * (1 - shield_miss_rate * heat_shield) 
                              * area_front * np.cos(impact_angle)
                           + backside  * 1 
                              * area_front * np.cos(impact_angle)
                           + area_side * np.sin(np.abs(impact_angle)) )
                
                L_a = l_a * area * (v_factor)**(e_v+1) * (r_factor)**e_a_r
                
                #normalization to hourly rate, while L_i are in s^-1
                hourly_rate = 3600 * ( L_b + L_a )

                return hourly_rate
            return usual_rate
        elif r_function_list == alternative_r_function_list:
            print("""the alternative rate function is applied""")
            def alternative_rate(v_sc_r, v_sc_t,
                           r_sc,
                           area_front, area_side,
                           heat_shield,
                           l_a, l_b,
                           v_b_r,
                           e_v,
                           e_b_r,
                           shield_miss_rate,
                           e_a_r=-1.3,
                           v_b_a=9, v_earth_a=0):
        
                def deg2rad(deg):
                    return deg / 180 * np.pi

                #beta meteoroid contribution
                ksi = -2 - (-1.5-e_b_r)
                r_factor = r_sc/1
                v_factor = ( (
                    ( v_sc_r - ( v_b_r*(r_factor**ksi)  ) )**2
                    + ( v_sc_t - ( v_b_a*(r_factor**(-1)) ) )**2
                  )**0.5
                  ) / ( (
                    ( v_b_r )**2
                    + ( v_earth_a - v_b_a )**2
                  )**0.5
                  )
                radial_impact_velocity = -1* (v_sc_r-(v_b_r*(r_factor**ksi)))
                  #positive is on the heatshield, negative on the tail
                azimuthal_impact_velocity = np.abs(v_sc_t-(v_b_a*(r_factor**(-1))))
                  #always positive, RHS vs LHS plays no role
                impact_angle = np.arctan( azimuthal_impact_velocity
                                    / radial_impact_velocity )
                
                frontside = radial_impact_velocity > 0
                backside = (frontside != True)
                area = (   frontside   * 1
                              * area_front * np.cos(impact_angle)
                           + backside  * 1 
                              * area_front * np.cos(impact_angle)
                           + area_side * np.sin(np.abs(impact_angle))
                       ) * (1 - shield_miss_rate * heat_shield)

                L_b = l_b * area * (v_factor)**(e_v+1) * (r_factor)**(-1.5-e_b_r)
                
                #bound dust contribution
                ksi = -2 - e_a_r
                r_factor = r_sc/1
                v_a_a = 29.8*(r_factor**(-0.5))
                v_factor = ( (
                    ( v_sc_r )**2
                    + ( v_sc_t - v_a_a )**2
                  )**0.5
                  ) / np.abs( v_earth_a - v_a_a )
                radial_impact_velocity = -1* ( v_sc_r ) 
                  #positive is on the heatshield, negative on the tail
                azimuthal_impact_velocity = np.abs( v_sc_t - v_a_a )
                  #always positive, RHS vs LHS plays no role
                impact_angle = np.arctan( azimuthal_impact_velocity
                                          / radial_impact_velocity )
                
                frontside = radial_impact_velocity > 0
                backside = (frontside != True)
                area = (   frontside   * 1
                              * area_front * np.cos(impact_angle)
                           + backside  * 1 
                              * area_front * np.cos(impact_angle)
                           + area_side * np.sin(np.abs(impact_angle))
                       ) * (1 - shield_miss_rate * heat_shield)
                
                L_a = l_a * area * (v_factor)**(e_v+1) * (r_factor)**e_a_r
                
                #normalization to hourly rate, while L_i are in s^-1
                hourly_rate = 3600 * ( L_b + L_a )

                return hourly_rate
            return alternative_rate
        elif not self.careful:
            print("""

                  the rate function in R is not the usual one,
                  proceed with caution

                  """)
        else:
            raise Warning("""the rate function in R is not the usual one,
                          proceed with caution""")

        return usual_rate


    def get_detection_errors(self,
                             counts,
                             prob_coverage = 0.9):
        """
        The function to calculate the errorbars for flux 
        assuming Poisson distribution and taking into account
        the number of detections.

        Parameters
        ----------
        counts : array of float
            Counts per day.
        prob_coverage : float, optional
            The coverage of the errobar interval. The default is 0.9.

        Returns
        -------
        err_plusminus_flux : np.array of float
            The errorbars, lower and upper bound, shape (2, n). 
            The unit is [1] (count).

        """

        counts_err_minus = -stats.poisson.ppf(0.5-prob_coverage/2,
                                              mu=counts)+counts
        counts_err_plus  = +stats.poisson.ppf(0.5+prob_coverage/2,
                                              mu=counts)-counts
        err_plusminus_counts = np.array([counts_err_minus,
                                       counts_err_plus])

        return err_plusminus_counts


    def summary_posterior(self):
        """
        Prints a short summary (means and variances) of the posteriors
        for all the attributes.

        Returns
        -------
        None.

        """
        print("Summary posterior:")
        for att in self.atts:
            mean = np.mean(self.contents[f"sample_{att}"].to_numpy())
            stdev = np.std(self.contents[f"sample_{att}"].to_numpy())
            print(f"{att}:\t mean = {mean:.3}\t +- {stdev:.2}")


    def summary_prior(self):
        """
        Prints a short summary (means and variances) of the priors
        for all the attributes.

        Returns
        -------
        None.

        """
        print("Summary prior:")
        for att in self.atts:
            mean = np.average(a=self.contents[f"px_{att}"].to_numpy(),
                              weights=self.contents[f"py_{att}"].to_numpy())
            stdev = (np.average(a=((self.contents[f"px_{att}"].to_numpy())**2),
                                weights=self.contents[f"py_{att}"].to_numpy())
                     - mean**2)**0.5

            print(f"{att}:\t mean = {mean:.3}\t +- {stdev:.2}")


    def sample(self,atts=None,sample_size=None):
        """
        Provides a sample from the posterior of the attribute(s) of choice.

        Parameters
        ----------
        atts : str or list of str or None
            The attrbiute of interest, one or several of self.atts. 
            In None, then the multivariate sample of all 
            the attributes is returned. 
        sample_size : int, optional
            The size of the sample. If None, then the full
            sample as provided by INLA is returned,
            i.e. (sample_size = self.sample_size). 
            The default is None, hence self.sample_size.

        Returns
        -------
        samples : np.ndarray
            If a single attribute is requested, then the sample 
                is 1D array of len sample_size.
            If several (n) attributes are requested, then the sample
                is 2D array of shape (n,sample_size), i.e. the first
                dimension is the attribute order.
            If atts == None, then all the attributes as in self.atts
                are requested and the shape of the 2D array is 
                therefore (len(self.atts),sample_size).

        """
        if sample_size is None:
            sample_size = self.sample_size
            replace = False
        else:
            replace = True

        if atts is None:
            atts = self.atts

        if isinstance(atts, str):
            samples = np.random.choice(
                        self.contents[f"sample_{atts}"].to_numpy().flatten(),
                        size=sample_size,
                        replace=replace)
        else:
            indices = np.random.choice(np.arange(self.sample_size),
                                       size = sample_size,
                                       replace = replace)

            samples = np.zeros((0,sample_size))
            for att in atts:
                row = self.contents[f"sample_{att}"].to_numpy(
                                                     ).flatten()[indices]
                samples = np.vstack((samples,row))

        return samples


    def pdf(self,att,prior=False):
        """
        Returns the PDF, either posterior or prior, for the attribute 
        of choice.

        Parameters
        ----------
        att : str
            The attrbiute of interest, one of self.atts.
        prior : bool, optional
            Whther to return prior. If False, then posterior is returned.
            The default is False.

        Returns
        -------
        interpolated : scipy.interpolate._interpolate.interp1d 
            The pdf function, callable.

        """
        if prior:
            x = self.contents[f"px_{att}"].to_numpy().flatten()
            y = self.contents[f"py_{att}"].to_numpy().flatten()
        else:
            x = self.contents[f"fx_{att}"].to_numpy().flatten()
            y = self.contents[f"fy_{att}"].to_numpy().flatten()
        x = x[y>max(y)/10**4]
        y = y[y>max(y)/10**4]
        interpolated = interpolate.interp1d(x, y,
                                            kind=3,
                                            bounds_error=False,
                                            fill_value=0)
        return interpolated


    def rate_samples(self,
                     sample_size=100):
        """
        Provides sampled mu matrices of shape (sample_size,len(jd)), 
        separately for solo, psp, bound and beta.

        Parameters
        ----------
        sample_size : int, optional
            The sample size, i.e. the number of rows in the poutput arrays. 
            The default is 100.

        Returns
        -------
        solo_prediction_bound : np.array of float, 2D
            The predicted bound component flux detection for SolO [/h].
        solo_prediction_beta : np.array of float, 2D
            The predicted beta component flux detection for SolO [/h].
        psp_prediction_bound : np.array of float, 2D
            The predicted bound component flux detection for PSP [/h].
        psp_prediction_beta : np.array of float, 2D
            The predicted beta component flux detection for PSP [/h].

        """

        sample = self.sample(sample_size=sample_size)
        mu = self.rate_function()
        solo_prediction_bound = np.zeros(
            (0,len(self.solo_input_df["Julian date"])))
        solo_prediction_beta = np.zeros(
            (0,len(self.solo_input_df["Julian date"])))
        psp_prediction_bound = np.zeros(
            (0,len(self.psp_input_df["Julian date"])))
        psp_prediction_beta = np.zeros(
            (0,len(self.psp_input_df["Julian date"])))

        for i,e in enumerate(sample[0][:]):
            solo_prediction_bound = np.vstack((solo_prediction_bound,
                mu(
                    self.solo_input_df["Radial velocity [km/s]"],
                    self.solo_input_df["Tangential velocity [km/s]"],
                    self.solo_input_df["Radial distance [au]"],
                    10.34 * np.ones(len(self.solo_input_df["Julian date"])),
                    8.24  * np.ones(len(self.solo_input_df["Julian date"])),
                    0     * np.ones(len(self.solo_input_df["Julian date"])),
                    sample[0][i],
                    0,
                    sample[2][i],
                    sample[3][i],
                    sample[4][i],
                    sample[5][i]
                )))
            solo_prediction_beta = np.vstack((solo_prediction_beta,
                mu(
                    self.solo_input_df["Radial velocity [km/s]"],
                    self.solo_input_df["Tangential velocity [km/s]"],
                    self.solo_input_df["Radial distance [au]"],
                    10.34 * np.ones(len(self.solo_input_df["Julian date"])),
                    8.24  * np.ones(len(self.solo_input_df["Julian date"])),
                    0     * np.ones(len(self.solo_input_df["Julian date"])),
                    0,
                    sample[1][i],
                    sample[2][i],
                    sample[3][i],
                    sample[4][i],
                    sample[5][i]
                )))

            psp_prediction_bound = np.vstack((psp_prediction_bound,
                mu(
                    self.psp_input_df["Radial velocity [km/s]"],
                    self.psp_input_df["Tangential velocity [km/s]"],
                    self.psp_input_df["Radial distance [au]"],
                    self.psp_input_df["Area front [m^2]"],
                    self.psp_input_df["Area side [m^2]"],
                    1 * np.ones(len(self.psp_input_df["Julian date"])),
                    sample[0][i],
                    0,
                    sample[2][i],
                    sample[3][i],
                    sample[4][i],
                    sample[5][i]
                )))
            psp_prediction_beta = np.vstack((psp_prediction_beta,
                mu(
                    self.psp_input_df["Radial velocity [km/s]"],
                    self.psp_input_df["Tangential velocity [km/s]"],
                    self.psp_input_df["Radial distance [au]"],
                    self.psp_input_df["Area front [m^2]"],
                    self.psp_input_df["Area side [m^2]"],
                    1 * np.ones(len(self.psp_input_df["Julian date"])),
                    0,
                    sample[1][i],
                    sample[2][i],
                    sample[3][i],
                    sample[4][i],
                    sample[5][i]
                )))

        return (solo_prediction_bound, solo_prediction_beta,
                psp_prediction_bound, psp_prediction_beta)


    def plot_prior_posterior(self,
                             atts=None,
                             xrange = [[0,5e-4],
                                       [0,5e-4],
                                       [0,100],
                                       [0,3],
                                       [0,1],
                                       [0,1]],
                             title=None):
        """
        The procedure that tshows the priors and the posteriors as per
        the InlaResult. 

        Parameters
        ----------
        atts : list of str, optional
            The list of attributes to show. If None, then 
            self.atts is used. The default is None.
        xrange : list of lists of float, optional
            The x axis limits. The default is [[0,5e-4],
                                               [0,5e-4],
                                               [0,100],
                                               [0,3],
                                               [0,1],
                                               [0,1]].

        Returns
        -------
        None.

        """

        if atts is None:
            atts = self.atts
        elif isinstance(atts, str):
            atts = [atts]

        fig = plt.figure(figsize=(4,0.666*len(atts)))
        gs = fig.add_gridspec(len(atts), hspace=.6)
        ax = gs.subplots()

        for i,att in enumerate(atts):
            x = np.linspace(xrange[i][0],xrange[i][1],num=100000)
            prior = self.pdf(att,prior=True)(x)
            posterior = self.pdf(att,prior=False)(x)
            ax[i].plot(x,prior/max(prior),label="prior")
            ax[i].plot(x,posterior/max(posterior),label="posterior")
            ax[i].set_xlim(xrange[i])
            ax[i].set_ylim(bottom=0)
            ax[i].set_ylabel(att)
        ax[0].legend(fontsize="x-small")

        if title is not None:
            fig.suptitle(title)

        fig.show()


    def plot_prior_posterior_natural(self,
                                     atts=None,
                                     xrange = [[0,3e-4],
                                               [0,3e-4],
                                               [20,80],
                                               [0,4],
                                               [0,1],
                                               [0,1]],
                                     ylabels = [r"$\lambda_a$",
                                                r"$\lambda_\beta$",
                                                r"$v_{\beta,r}$",
                                                r"$\epsilon_v$",
                                                r"$\epsilon_{\beta,r}$",
                                                r"$\alpha_{shield}$"],
                                     title=None):
        """
        The procedure that tshows the priors and the posteriors as per
        the InlaResult, converted to natural units. 

        Parameters
        ----------
        atts : list of str, optional
            The list of attributes to show. If None, then 
            self.atts is used. The default is None.
        xrange : list of lists of float, optional
            The x axis limits. The default is [[0,5e-4],
                                               [0,5e-4],
                                               [0,100],
                                               [1,3],
                                               [-3,0]].

        Returns
        -------
        None.

        """

        if atts is None:
            atts = self.atts
        elif isinstance(atts, str):
            atts = [atts]

        fig = plt.figure(figsize=(4,0.666*len(atts)))
        gs = fig.add_gridspec(len(atts), hspace=.6)
        ax = gs.subplots()

        for i,att in enumerate(atts):
            x = np.linspace(xrange[i][0],xrange[i][1],num=100000)
            prior = self.pdf(att,prior=True)(x)
            posterior = self.pdf(att,prior=False)(x)
            if i == 3:
                ax[i].plot(x+1,prior/max(prior),label="prior")
                ax[i].plot(x+1,posterior/max(posterior),label="posterior")
                ax[i].set_xlim([1,3])
            elif i == 4:
                ax[i].plot(-x-1.5,prior/max(prior),label="prior")
                ax[i].plot(-x-1.5,posterior/max(posterior),label="posterior")
                ax[i].set_xlim([-2.5,-1.5])
            else:
                ax[i].plot(x,prior/max(prior),label="prior")
                ax[i].plot(x,posterior/max(posterior),label="posterior")
                ax[i].set_xlim(xrange[i])
            ax[i].set_ylim(bottom=0)
            ax[i].set_ylabel(ylabels[i])
        ax[0].legend(fontsize="x-small")

        if title is not None:
            fig.suptitle(title)

        fig.show()


    def overplot(self,
                 sample_size=100,
                 title=None,
                 aspect = 1.333):
        """
        A procedure to plot the detected counts under the INLA fitted rates.
        The rates are decomposed into beta and bound. 

        Parameters
        ----------
        sample_size : int, optional
            The number of sampled fluxes to draw over the measured data.
            The default is 100.
        title : str, optional
            The suptitle of the plot. 
            The default is None, in which case not suptitle is used.
        aspect : float, optional
            The aspect ratio of the plot.

        Returns
        -------
        None.

        """

        fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(aspect*3, 3),
                               sharex=True)

        for dataset, axis, flux_name in zip([self.solo_input_df,
                                                 self.psp_input_df],
                                                [ax[0],ax[1]],
                                                ["Fluxes [/day]",
                                                 "Count corrected [/day]"]):
            for i,df in enumerate([
                    dataset[dataset['Radial distance [au]'] <= 0.4],
                    dataset[dataset['Radial distance [au]'] > 0.4]]):
                point_errors = self.get_detection_errors(df[flux_name])
                axis.errorbar([jd2date(jd) for jd in df["Julian date"]],
                               df[flux_name] / df["Detection time [hours]"],
                               np.vstack((point_errors[0]
                                            / df["Detection time [hours]"],
                                          point_errors[1]
                                            / df["Detection time [hours]"]
                               )),
                               lw=0, elinewidth=0.3, alpha=0.5,
                               c="navajowhite"*(i==0)+"darkorange"*(i==1))

        (solo_prediction_bound,
         solo_prediction_beta,
         psp_prediction_bound,
         psp_prediction_beta) = self.rate_samples(sample_size)
        for prediction, label, color in zip([(solo_prediction_bound
                                               + solo_prediction_beta),
                                             solo_prediction_bound,
                                             solo_prediction_beta],
                                            ["Total","Bound","Beta"],
                                            ["black","blue","red"]):
            lw = 0.5
            if label == "Total":
                lw = 1
            ax[0].plot([jd2date(jd) for jd
                        in self.solo_input_df["Julian date"]],
                        np.mean(prediction, axis = 0),
                        color=color, label=label, lw=lw)
        for prediction, label, color in zip([(psp_prediction_bound
                                               + psp_prediction_beta),
                                             psp_prediction_bound,
                                             psp_prediction_beta],
                                            ["Total","Bound","Beta"],
                                            ["black","blue","red"]):
            lw = 0.5
            if label == "Total":
                lw = 1
            ax[1].plot([jd2date(jd) for jd
                        in self.psp_input_df["Julian date"]],
                        np.mean(prediction, axis = 0),
                        color=color, label=label, lw=lw)

        for a in ax:
            a.set_ylim(bottom=0)
            a.set_ylim(top=30)
            a.set_ylabel(r"Detection rate $[/h]$")
            a.set
        ax[0].legend()
        if title is not None:
            fig.suptitle(title)
        fig.show()


    def radial_fit_profile(self,
                           sample_size=100):
        """
        A procedure to compare the measured points to the prediction,
        as a function of the heliocentric distance.

        Parameters
        ----------
        sample_size : int, optional
            The sample size used to uvaluate the mean model. 
            The default is 100.

        Returns
        -------
        None.

        """

        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(4, 3))

        self.solo_input_df['Radial distance [au]']
        self.psp_input_df['Radial distance [au]']

        (solo_prediction_bound,
         solo_prediction_beta,
         psp_prediction_bound,
         psp_prediction_beta) = self.rate_samples(sample_size)
        solo_prediction = np.mean(solo_prediction_bound
                                  + solo_prediction_beta, axis=0)
        psp_prediction = np.mean(psp_prediction_bound
                                 + psp_prediction_beta, axis=0)

        for (df,
             flux_name,
             prediction,
             color,
             label) in zip([self.solo_input_df,
                            self.psp_input_df],
                           ["Fluxes [/day]",
                            "Count corrected [/day]"],
                           [solo_prediction,
                            psp_prediction],
                           ["red",
                            "blue"],
                           ["SolO",
                            "PSP"]):

            ax.scatter(df['Radial distance [au]'],
                        (df[flux_name]
                         / df["Detection time [hours]"]
                         / prediction),
                        s=1, c=color, alpha=0.2, label=label)

        min_r = min(self.psp_input_df['Radial distance [au]'])
        max_r = max(self.solo_input_df['Radial distance [au]'])

        ax.hlines(1,min_r,max_r,
                  lw=1,color="black")
        ax.set_ylabel(r"Detections / model $[1]$")
        ax.set_xlabel(r"Heliocentric distance $[AU]$")
        ax.set_yscale('log')
        ax.legend()
        fig.show()


    def zoom_psp_maxima(self,
                        sample_size=10,
                        max_perihelia=16,
                        days=14,
                        aspect=2,
                        zoom=1.2,
                        split=False,
                        pointcolor="darkorange",
                        linecolor="black",
                        filename=None):
        """
        A procedure to plot the zoom / crop on the maxima, i.e. near perihelia. 

        Parameters
        ----------
        sample_size : int, optional
            The number of sampled fluxes to draw over the measured data.
            The default is 100.
        max_perihelia : int, optional
            How many perihelia to show, counting from the first. 
            The default is 16.
        aspect : float, optional
            The aspect ratio of the plot.
        zoom : float, optional
            The zoom of the plots, higher number implies larger texts.
        split : bool, optional
            Whether to split the fit line into bound dust and beta or not. 
            The default is False.
        filename : str, optional
            The filename of the .png to be saved. the default is None, in which
            case, the plot is not saved.

        Returns
        -------
        None.

        """

        # Getting the approaches
        approaches = np.linspace(1,max_perihelia,
                                 max_perihelia,
                                 dtype=int)

        approach_dates = np.array([jd2date(a)
                                      for a
                                      in get_approaches(psp_ephemeris_file)
                                      ][:max_perihelia])
        approach_groups = np.array([encounter_group(a)
                                       for a
                                       in approaches])

        df = self.psp_input_df
        dates = np.array([jd2date(jd) for jd in df["Julian date"]])
        post_approach_threshold_passages = np.array([
            np.min(dates[(dates>approach_date)
                         *(df["Radial distance [au]"]>0.4)])
            for approach_date in approach_dates])

        # Evaluate the model
        (solo_prediction_bound,
         solo_prediction_beta,
         psp_prediction_bound,
         psp_prediction_beta) = self.rate_samples(sample_size)

        # Calculate the scatter plot
        point_errors = self.get_detection_errors(df["Count corrected [/day]"])
        duty_hours = df["Detection time [hours]"]
        detecteds = df["Count corrected [/day]"]
        scatter_point_errors = np.vstack((point_errors[0]
                                             / df["Detection time [hours]"],
                                          point_errors[1]
                                             / df["Detection time [hours]"]
                                          ))

        # Caluclate the model lines for beta and bound
        mean_expected_counts = (np.mean(psp_prediction_beta
                                        + psp_prediction_bound,
                                        axis=0))*duty_hours
        eff_rate = mean_expected_counts/duty_hours
        mean_expected_counts_beta = (np.mean(psp_prediction_beta,
                                             axis=0))*duty_hours
        eff_rate_beta = mean_expected_counts_beta/duty_hours
        mean_expected_count_bound = (np.mean(psp_prediction_bound,
                                             axis=0))*duty_hours
        eff_rate_bound = mean_expected_count_bound/duty_hours

        # Plot
        fig = plt.figure(figsize=(4*aspect/zoom, 4/zoom))
        ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2, fig=fig)
        ax2 = plt.subplot2grid((2,6), (0,2), colspan=2, fig=fig)
        ax3 = plt.subplot2grid((2,6), (0,4), colspan=2, fig=fig)
        ax4 = plt.subplot2grid((2,6), (1,1), colspan=2, fig=fig)
        ax5 = plt.subplot2grid((2,6), (1,3), colspan=2, fig=fig)
        axes = np.array([ax1,ax2,ax3,ax4,ax5])

        for a in axes[:]:
            a.set_ylabel("Rate [/h equiv.]")
        for a in axes[:]:
            a.set_xlabel("Time after perihelion [h]")

        # Iterate the groups
        for i,ax in enumerate(axes):  #np.ndenumerate(axes):
            group = i+1
            if group in set(approach_groups):

                ax.set_title(f"Enc. group {group}")

                line_hourdiff = np.zeros(0)
                line_rate = np.zeros(0)
                line_rate_beta = np.zeros(0)
                line_rate_bound = np.zeros(0)

                for approach_date in approach_dates[approach_groups==group]:
                    filtered_indices = np.abs(dates-approach_date
                                              )<dt.timedelta(days=days)
                    datediff = dates[filtered_indices]-approach_date
                    hourdiff = [24*d.days + d.seconds/3600
                                for d in datediff]
                    passage_days = (np.max(post_approach_threshold_passages[
                                            approach_groups==group])
                                    - approach_date)
                    passage_hours = (24*passage_days.days
                                     + passage_days.seconds/3600)
                    ax.scatter(hourdiff,
                               (detecteds[filtered_indices]
                                /duty_hours[filtered_indices]),
                              c=pointcolor,s=0.,zorder=100)
                    ax.errorbar(hourdiff,
                                (detecteds[filtered_indices]
                                 /duty_hours[filtered_indices]),
                                scatter_point_errors[:,filtered_indices],
                               c=pointcolor, lw=0., elinewidth=1,alpha=0.5)

                    line_hourdiff = np.append(line_hourdiff,hourdiff)
                    line_rate = np.append(line_rate,
                                          eff_rate[filtered_indices])
                    line_rate_beta=np.append(line_rate_beta,
                                             eff_rate_beta[filtered_indices])
                    line_rate_bound=np.append(line_rate_bound,
                                              eff_rate_bound[filtered_indices])
                sortmask = line_hourdiff.argsort()

                ax.plot(line_hourdiff[sortmask][1::2],
                        line_rate[sortmask][1::2],
                        c=linecolor,lw=1,zorder=101,label="Total")
                max_y = ax.get_ylim()[1]
                ax.vlines([-passage_hours,passage_hours],
                          0,10000,
                          color="gray")
                if split:
                    ax.plot(line_hourdiff[sortmask][1::2],
                            line_rate_beta[sortmask][1::2],
                            c="red",ls="solid",
                            lw=0.5,zorder=101,label="Beta")
                    ax.plot(line_hourdiff[sortmask][1::2],
                            line_rate_bound[sortmask][1::2],
                            c="blue",ls="solid",
                            lw=0.5,zorder=101,label="Bound")
                    ax.legend(loc=2, fontsize="x-small", frameon=True,
                              facecolor='white',
                              edgecolor='black').set_zorder(200)
                ax.set_ylim(0,max_y)
                ax.set_xlim(-days*24,days*24)

        fig.tight_layout()

        if filename is not None:
            fig.savefig(figures_location+filename+".png",dpi=1200)

        fig.show()


    def ephemerides(self,
                    title=None,
                    aspect = 1.333,
                    detail = False):
        """
        A procedure to plot the ephemerides of SolO and PSP 
        on top of each other. 
    
        Parameters
        ----------
        title : str, optional
            The suptitle of the plot. 
            The default is None, in which case not suptitle is used.
        aspect : float, optional
            The aspect ratio of the plot.
        detail : bool, optional
            Thether to zoom into a comparioson of two most similar aphelia.
    
        Returns
        -------
        None.
    
        """
        if detail:
            fig, ax = plt.subplots(nrows=1,ncols=2,
                                   figsize=(aspect*3, 3))
        else:
            fig, ax = plt.subplots(nrows=2,ncols=1,
                                   figsize=(aspect*3, 3),
                                   sharex=True)

        for dataset, axis in zip([self.solo_input_df,
                                  self.psp_input_df],
                                 [ax[0],ax[1]]):
            # Loading data
            t = [jd2date(jd) for jd in dataset["Julian date"]]
            r = dataset['Radial distance [au]']
            v_rad = dataset["Radial velocity [km/s]"]
            v_azim = dataset["Tangential velocity [km/s]"]

            # Plotting r, v
            axis.plot(t,r,"k")
            axis_sec = axis.twinx()
            axis_sec.plot(t, v_rad, c='crimson', ls="dashed",
                          label="Radial")
            axis_sec.plot(t, v_azim, c='crimson', ls="solid",
                          label="Azimuthal")

            # Plot makeup
            for tl in axis_sec.get_yticklabels():
                tl.set_color('crimson')
            if detail:
                if axis == ax[0]:
                    axis.set_ylabel("Heliocentric \n distance"+r" [$AU$]")
                    axis.set_xlim(dt.datetime(2022,12,1),
                                  dt.datetime(2023,3,1))
                    axis.set_title("SolO, 5th to 6th")
                else:
                    axis_sec.legend()
                    axis_sec.set_ylabel(r"Speed [$km/s$]", color="crimson")
                    axis.set_xlim(dt.datetime(2019,10,1),
                                  dt.datetime(2020,1,1))
                    axis.set_title("PSP, 3rd to 4th")
                axis.xaxis.set_major_locator(mdates.MonthLocator())
                axis.xaxis.set_minor_locator(
                    mdates.DayLocator(bymonthday=(1, 15)))
                axis.set_ylim(0.5,1.1)
                axis_sec.set_ylim(0,50)
                xlabels = axis.get_xticklabels()
                axis.set_xticklabels(xlabels, rotation=40,
                                     ha="right")
            else:
                if axis == ax[0]:
                    axis_sec.legend()
                axis.set_ylabel("Heliocentric \n distance"+r" [$AU$]")
                axis_sec.set_ylabel(r"Speed [$km/s$]", color="crimson")
                axis.xaxis.set_major_locator(mdates.YearLocator())
                axis.xaxis.set_minor_locator(
                    mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
                axis.set_ylim(0.5,1.1)
                axis_sec.set_ylim(0,50)






        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()
        fig.show()




#%%

if __name__ == "__main__":

    r_files = glob.glob(os.path.join("998_generated","inla",
                                     "solo_psp_together_*.RData"))
    r_files["_" in r_files]
    r_filepath = r_files[-1]
    csv_input_solo_path = os.path.join("data_synced","solo_flux_readable.csv")
    csv_input_psp_path = os.path.join("data_synced","psp_flux_readable.csv")

    result = InlaResult(r_filepath, csv_input_solo_path, csv_input_psp_path,
                        careful=False)

    result.summary_prior()
    result.summary_posterior()

    result.plot_prior_posterior_natural(
        title = r_filepath[r_filepath.find("_sample_")+8:-6])

    result.overplot(aspect=2)

    #result.radial_fit_profile()

    result.zoom_psp_maxima(split=True)

    #result.ephemerides()
    #result.ephemerides(detail=True)





