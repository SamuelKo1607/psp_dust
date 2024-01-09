import numpy as np
from scipy import stats
import matplotlib as mpl
import pyreadr

from load_data import Observation
from load_data import load_all_obs
from paths import all_obs_location
from paths import legacy_inla_champion


import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600



def mu(b1, b2, c1, c2, v1, r, vr, vt):
    """
    The legacy detection rate, as in A&A 2023. 

    Parameters
    ----------
    b1 : float
        velocity exponent
    b2 : float
        heliocentric distance exponent
    c1 : float
        multiplicative constant, beta rate
    c2 : float
        constant, background rate
    v1 : float
        the mean dust radial speed
    r : float
        SC heliocentric distance
    vr : float
        SC radial velocity
    vt : float
        SC azimuthal velocity

    Returns
    -------
    rate : float
        The predicted detection rate. The unit is [/h]. 

    """
    rate = ( (((vr-v1)**2+(vt-(12*0.75/r))**2)**0.5)/50 )**(b1)*r**(b2)*c1 + c2
    return rate


def read_legacy_inla_result(filename):
    samples = pyreadr.read_r(filename)
    b1s = np.array(samples["b1"])
    b2s = np.array(samples["b2"])
    c1s = np.array(samples["c1"])
    c2s = np.array(samples["c2"])
    v1s = np.array(samples["v1"])
    return b1s, b2s, c1s, c2s, v1s


def get_detection_errors(counts,
                         duty_hours,
                         prob_coverage = 0.9):
    """
    The function to calculate the errorbars for flux 
    assuming Poisson distribution and taking into account
    the number of detections.

    Parameters
    ----------
    counts : array of float
        Counts per day.
    duty_hours : array of float
        Duty hours per day.
    prob_coverage : float, optional
        The coverage of the errobar interval. The default is 0.9.

    Returns
    -------
    err_plusminus_flux : np.array of float
        The errorbars, lower and upper bound, shape (2, n).

    """

    counts_err_minus = -stats.poisson.ppf(0.5-prob_coverage/2,
                                          mu=counts)+counts
    counts_err_plus  = +stats.poisson.ppf(0.5+prob_coverage/2,
                                          mu=counts)-counts
    err_plusminus_flux = np.array([counts_err_minus,
                                   counts_err_plus]) / (duty_hours/(24))

    return err_plusminus_flux


def get_poisson_sample(rate,
                       duty_hours,
                       sample_size=100):
    """
    A function which gives a sample of rates, assuming Poisson distribution.

    Parameters
    ----------
    rate : float
        The rate as sampled from the INLA posterior and applying using mu().
    duty_hours : float
        The detection time.
    sample_size : int, optional
        The sample size, single rate. The default is 1000.

    Returns
    -------
    rates : TYPE
        DESCRIPTION.

    """
    rates = np.zeros(0)
    for k in range(sample_size):
        rates = np.append(rates,
                          np.random.poisson(lam=rate*duty_hours,
                                            size=sample_size)/duty_hours)
    return rates


def get_predicted_range(r, vr, vt, duty_hours,
                        b1s, b2s, c1s, c2s, v1s,
                        sample_mu=100,
                        sample_poiss=100,
                        prob_coverage=0.9):

    available_samples = len(b1s)
    sample_draw = np.random.choice(np.arange(available_samples),
                                   size = sample_mu)
    mus = [mu(b1s[i], b2s[i], c1s[i], c2s[i], v1s[i],
              r, vr, vt) for i in sample_draw]
    sampled_rates = np.zeros(0)
    for mu in mus:
        some_rates = get_poisson_sample(mu,
                                        duty_hours,
                                        sample_size = sample_poiss)
        sampled_rates = np.append(sampled_rates,some_rates)

    lower_predicted = np.quantile(sampled_rates,(1-prob_coverage)/2)
    upper_predicted = np.quantile(sampled_rates,1-(1-prob_coverage)/2)

    return lower_predicted, upper_predicted


def plot_psp_data_solo_model():

    b1s, b2s, c1s, c2s, v1s = read_legacy_inla_result(legacy_inla_champion)
    psp_obs = load_all_obs(all_obs_location)
    dates = [ob.date for ob in psp_obs]

    for i, date in enumerate(dates):
        # get the scatter point
        detected = psp_obs[i].count_corrected

        # get the scatter point's error
        scatter_point_errors = get_detection_errors(detected,
                                                    psp_obs[i].duty_hours)

        # get the interval of modelled rate
        lower_predicted, upper_predicted = get_predicted_range(
            psp_obs[i].heliocentric_distance,
            psp_obs[i].heliocentric_radial_speed,
            psp_obs[i].heliocentric_tangential_speed,
            psp_obs[i].duty_hours,
            b1s, b2s, c1s, c2s, v1s)

        #plot it



    # as a scatterplot with poission errors and overplotted with red errorbar heads showing the predicted rate
    pass



