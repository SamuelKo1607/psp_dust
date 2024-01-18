import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyreadr
from tqdm.auto import tqdm
import unittest
import datetime as dt
import pickle

from load_data import Observation
from load_data import load_all_obs
from overplot_with_solo_result import read_legacy_inla_result
from paths import all_obs_location
from paths import legacy_inla_champion
from paths import figures_location
from paths import grid_fiting_results


import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600


def mu_vectorized(b1, b2, c1, c2, v1,
                  r, vr, vt,
                  c3=0,
                  shield_compensation=1,
                  bound_r_exponent=-1.3,
                  area_front=6.11,
                  area_side=4.62):
    """
    The legacy detection rate, as in A&A 2023 with improvements: 
        1. shield sensitivity, 2. boud dust contribution. Vectorized in 
        ephemerides.

    Parameters
    ----------
    b1 : float
        velocity exponent
    b2 : float
        heliocentric distance exponent
    c1 : float
        multiplicative constant in beta rate
    c2 : float
        constant, background rate. 
    v1 : float
        the mean dust radial speed
    r : np.array of float
        SC heliocentric distance
    vr : np.array of float
        SC radial velocity
    vt : np.array of float
        SC azimuthal velocity
    c3 : float, optional
        multiplicative constant in bound dust rate. 
        The default is 0, hence as in A&A 2023.
    shield_compensation : float, optional
        Whether to account for a lower shield sensitivity. 
        A float between 0 and 1, where 1 means the same 
        sensitivity as the rest of the spacecraft. 
        The default is 1.
    area_front : float, optional
        Front-side projection area of the spacecraft [m^2]. 
        The default is 6.11, i.e. PSP forntal projection, shield included.
    area_side : float
        Lateral projection area of the spacecraft [m^2]. 
        The default is 4.62, i.e. PSP lateral projection.

    Returns
    -------
    rate : np.array of float
        The predicted detection rate. The unit is [/h]. 

    """
    # beta (c1)
    if c1<=0:
        rate_beta = np.zeros(len(r))
    else:
        v_front_beta = (v1-vr)
        v_side_beta = ((12*0.75/r)-vt)
        rate_beta_raw = (
                            ((v_front_beta**2+v_side_beta**2)**0.5)/50
                        )**(b1)*r**(b2)*c1 + c2
    
        backside_hit = v_front_beta<0
        frontside_hit = v_front_beta>=0

        area_coeff = ( np.abs(v_front_beta)*area_front*shield_compensation +
                           np.abs(v_side_beta)*area_side
                         ) / ( np.abs(v_front_beta)*area_front
                               + np.abs(v_side_beta)*area_side )

        rate_beta = ( rate_beta_raw*backside_hit
                     + rate_beta_raw*area_coeff*frontside_hit)

    # bound (c3)
    if c3<=0:
        rate_bound = np.zeros(len(r))
    else:
        v_front_bound = -vr
        v_side_bound = ((29.8/r)-vt)
        rate_bound_raw = (
                            ((v_front_bound**2+v_side_bound**2)**0.5)/50
                        )**(b1)*r**(bound_r_exponent)*c3

        backside_hit = v_front_bound<0
        frontside_hit = v_side_bound>=0

        area_coeff = ( np.abs(v_front_bound)*area_front*shield_compensation
                       + np.abs(v_side_bound)*area_side
                     ) / ( np.abs(v_front_bound)*area_front
                           + np.abs(v_side_bound)*area_side )

        rate_bound = ( rate_bound_raw*backside_hit
                     + rate_bound_raw*area_coeff*frontside_hit)

    rate = rate_beta + rate_bound
    return rate


class TestMuVectorized(unittest.TestCase):
    def test_mu_vectorized_success(self):
        b1=2.04
        b2=-1.61
        c1=1.96
        c2=1.54
        v1=63.4
        r=np.array([0.5,1])
        vr=np.array([10,-10])
        vt=np.array([5,5])
        actual = mu_vectorized(b1, b2, c1, c2, v1, r, vr, vt)
        expected = np.array([8.796125269920047,5.842202419932526])
        diff = max(actual-expected)
        self.assertEqual(diff, 0)


def get_legacy_post_sample(sample_size=10,
                           result_location=legacy_inla_champion,
                           seed=123):
    """
    Gets a sample from the joint posterior of 
    the legacy INLA fitting for SolO.

    Parameters
    ----------
    sample_size : int, optional
        The length of the sample. The default is 10.
    result_location : str, optional
        the location of the .RData file of choice.
        The default is legacy_inla_champion.
    seed : int, optional
        The seed, for consisentency. The default is 123.

    Returns
    -------
    b1 : np.array of float
        The parameter as in Legacy INLA result.
    b2 : np.array of float
        The parameter as in Legacy INLA result.
    c1 : np.array of float
        The parameter as in Legacy INLA result.
    c2 : np.array of float
        The parameter as in Legacy INLA result.
    v1 : np.array of float
        The parameter as in Legacy INLA result.

    """
    b1s, b2s, c1s, c2s, v1s = read_legacy_inla_result(result_location)
    np.random.seed(seed)
    sample = np.random.choice(np.arange(len(b1s)), size=sample_size)
    b1 = b1s[sample]
    b2 = b2s[sample]
    c1 = c1s[sample]
    c2 = c2s[sample]
    v1 = v1s[sample]
    return b1, b2, c1, c2, v1


def get_poisson_rate(mus,
                     duty_hours,
                     max_detections=3000):
    """
    TBD

    Parameters
    ----------
    mus : np.array of float
        the vector of rates [/h] as computed by mu().
    duty_hours : np.array of float
        the vector of exposures [h], the same length as mus.
    max_detections : int, optional
        The highest independent variabel that is computed. 
        Since we use this to lalculate log likelihood, we need a number
        higher than the highest number of detections in the dataset.
        The default is 3000.

    Raises
    ------
    Exception
        If the input vectors differ in length.

    Returns
    -------
    probty_table : np.array 2D of float
        Rows (1st index) are different mus, 
        columns (2nd index) are different indep. variable k.

    """

    if len(mus)!=len(duty_hours):
        raise Exception("len(mus)!=len(duty_hours):"
                        +f" {len(mus)} vs {len(duty_hours)}")

    probty_table = np.zeros(shape=(len(mus), max_detections),dtype=float)
    for i in range(len(mus)):
        probty_single_mu = stats.poisson.pmf(np.arange(max_detections),
                                             mus[i]*duty_hours[i])
        probty_table[i,:] = probty_single_mu

    return probty_table


def get_poisson_range(mus,
                      duty_hours,
                      prob_coverage=0.9999):
    """
    To make the fitting robus, we need to include only reasonable points. 
    Here we evalueate the range of reasonable points for the given rates.

    Parameters
    ----------
    mus : np.array of float
        the vector of rates [/h] as computed by mu().
    duty_hours : np.array of float
        the vector of exposures [h], the same length as mus.
    prob_coverage : float, optional
        The acceptable range for a value. 1 mieans that all are included,
        0.9 means that only the ones tha fall within 90% of the 
        most likely results are included. The default is 0.9999.

    Raises
    ------
    Exception
        If the input vectors differ in length.

    Returns
    -------
    lower_boundaries : np.array of float
        The lowest acceptable counts, given the rates.
    upper_boundaries : np.array of float
        The highest acceptable counts, given the rates.

    """

    if len(mus)!=len(duty_hours):
        raise Exception("len(mus)!=len(duty_hours):"
                        +f" {len(mus)} vs {len(duty_hours)}")

    lower_boundaries = stats.poisson.ppf(0.5-prob_coverage/2,
                                         mu=mus*duty_hours)
    upper_boundaries = stats.poisson.ppf(0.5+prob_coverage/2,
                                         mu=mus*duty_hours)

    return lower_boundaries, upper_boundaries


def loglik(b1, b2, c1, c2, v1,
           r, vr, vt,
           duty_hours,
           detected,
           shield_compensation=0.3,
           c3=2.5):

    logliks = []
    accepted = []
    for i in range(len(b1)):
        mus = mu_vectorized(b1[i], b2[i], c1[i], c2[i], v1[i], r, vr, vt,
                            shield_compensation=shield_compensation,
                            c3=c3)
    
        lower_reason, upper_reason = get_poisson_range(mus,
                                                       duty_hours,
                                                       prob_coverage=0.99)
        mask_low_enough = detected<upper_reason
        mask_high_enough = detected>lower_reason
        mask = mask_low_enough * mask_high_enough
    
        reasonable_mus = mus[mask]
        reasonable_duty_hours = duty_hours[mask]
        reasonable_detected = detected[mask]

        table = get_poisson_rate(reasonable_mus,reasonable_duty_hours)
        sliced = np.log10(table[np.arange(np.shape(table)[0]),
                                reasonable_detected])
        logliks.append(np.mean(sliced))
        accepted.append(sum(mask))

    return np.mean(logliks), np.mean(accepted)


def main(shield_corrections=np.linspace(0.1,0.5,30),
         c3s=np.linspace(0,5,30),
         plot=False):

    b1, b2, c1, c2, v1 = get_legacy_post_sample()

    psp_obs = [ob for ob in load_all_obs(all_obs_location)
               if ob.duty_hours>0]
    dates = np.array([ob.date for ob in psp_obs])
    r = np.array([ob.heliocentric_distance for ob in psp_obs])
    vr = np.array([ob.heliocentric_radial_speed for ob in psp_obs])
    vt = np.array([ob.heliocentric_tangential_speed for ob in psp_obs])
    duty_hours = np.array([ob.duty_hours for ob in psp_obs])
    detected = np.array([int(ob.count_corrected) for ob in psp_obs])

    shield_corrections = shield_corrections
    c3s = c3s
    logliks = np.zeros((len(shield_corrections),len(c3s)),dtype=float)
    acceptables = np.zeros((len(shield_corrections),len(c3s)),dtype=float)
    for i, sc in enumerate(shield_corrections):
        for j, c3 in enumerate(c3s):
            iloglik, iacceptable = loglik(b1, b2, c1, c2, v1,
                                          r, vr, vt, duty_hours, detected,
                                          shield_compensation=sc,
                                          c3=c3)
            logliks[i,j] = iloglik
            acceptables[i,j] = iacceptable
            print(dt.datetime.now())
            print(logliks[i,j],acceptables[i,j])

    if plot:
        fig, ax = plt.subplots()
        lik = ax.contour(shield_corrections,c3s,
                         logliks.transpose())
        ax.clabel(lik, inline=True, fontsize=10)
        ax.set_title('logliks')
        ax.set_xlabel("shield correction")
        ax.set_ylabel("bound dust flux")
        fig.show()
    
        fig, ax = plt.subplots()
        lik = ax.contour(shield_corrections,c3s,
                         acceptables.transpose())
        ax.clabel(lik, inline=True, fontsize=10)
        ax.set_title('acceptable')
        ax.set_xlabel("shield correction")
        ax.set_ylabel("bound dust flux")
        fig.show()

    data = [shield_corrections,
            c3s,
            logliks,
            acceptables]
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    with open(grid_fiting_results+f"fit_{timestamp}.dat", "wb") as f:
        pickle.dump(data, f)




#%%
if __name__ == "__main__":
    unittest.main()

    main(shield_corrections=np.linspace(0.05,0.25,5),
         c3s=np.linspace(0.5,1.5,5),
         plot=True)

    # with open(grid_fiting_results+filename, "rb") as f:
    #     data = pickle.load(f)
    #     shield_corrections = data[0]
    #     c3s = data[1]
    #     logliks = data[2]
    #     acceptables = data[3]






