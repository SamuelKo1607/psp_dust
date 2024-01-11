import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyreadr
from tqdm.auto import tqdm

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
    """
    A container to read a file of interest and output its contents as arrays.

    Parameters
    ----------
    filename : str
        The file of interest (filepath).

    Returns
    -------
    b1s : np.array of float
        A legacy INLA hyperparameter.
    b2s : TYPE
        A legacy INLA hyperparameter.
    c1s : TYPE
        A legacy INLA hyperparameter.
    c2s : TYPE
        A legacy INLA hyperparameter.
    v1s : TYPE
        A legacy INLA hyperparameter.

    """
    samples = pyreadr.read_r(filename)
    b1s = np.array(samples["b1"]["b1"])
    b2s = np.array(samples["b2"]["b2"])
    c1s = np.array(samples["c1"]["c1"])
    c2s = np.array(samples["c2"]["c2"])
    v1s = np.array(samples["v1"]["v1"])
    return b1s, b2s, c1s, c2s, v1s


def get_detection_errors(counts,
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


def get_poisson_sample(rate,
                       duty_hours,
                       sample_size=100):
    """
    A function which gives a sample of rates, assuming Poisson distribution.

    Parameters
    ----------
    rate : float
        The rate as sampled from the INLA posterior and applying using mu().
        The unit has to be [/h].
    duty_hours : float
        The detection time [h].
    sample_size : int, optional
        The sample size, single rate. The default is 1000.

    Returns
    -------
    rates : np.array of int
        Sampled detection count.

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
    """
    

    Parameters
    ----------
    r : TYPE
        DESCRIPTION.
    vr : TYPE
        DESCRIPTION.
    vt : TYPE
        DESCRIPTION.
    duty_hours : TYPE
        DESCRIPTION.
    b1s : TYPE
        DESCRIPTION.
    b2s : TYPE
        DESCRIPTION.
    c1s : TYPE
        DESCRIPTION.
    c2s : TYPE
        DESCRIPTION.
    v1s : TYPE
        DESCRIPTION.
    sample_mu : TYPE, optional
        DESCRIPTION. The default is 100.
    sample_poiss : TYPE, optional
        DESCRIPTION. The default is 100.
    prob_coverage : TYPE, optional
        DESCRIPTION. The default is 0.9.

    Returns
    -------
    lower_expected_count : TYPE
        DESCRIPTION.
    upper_expected_count : TYPE
        DESCRIPTION.
    mean_expected_count : float
        The mean expected count, assuming duty_hours of observation
        and the hourly rate of mu.

    """
    available_samples = len(b1s)
    sample_draw = np.random.choice(np.arange(available_samples),
                                   size = sample_mu)
    mus = [mu(b1s[i], b2s[i], c1s[i], c2s[i], v1s[i],
              r, vr, vt) for i in sample_draw]
    sampled_counts = np.zeros(0)
    for imu in mus:
        some_counts = get_poisson_sample(imu,
                                         duty_hours,
                                         sample_size = sample_poiss)
        sampled_counts = np.append(sampled_counts,some_counts)

    lower_expected_count = np.quantile(sampled_counts,(1-prob_coverage)/2)
    upper_expected_count = np.quantile(sampled_counts,1-(1-prob_coverage)/2)
    mean_expected_count = np.mean(sampled_counts)

    return lower_expected_count, upper_expected_count, mean_expected_count


def plot_psp_data_solo_model(model_prefact=0.59,
                             aspect=1.5,
                             sample_mu=10,
                             sample_poiss=10,
                             smooth_model=True,
                             min_heliocentric_distance=0.,#25,
                             min_duty_hours=2):

    b1s, b2s, c1s, c2s, v1s = read_legacy_inla_result(legacy_inla_champion)
    psp_obs = load_all_obs(all_obs_location)
    psp_obs = [ob for ob in psp_obs
               if ob.heliocentric_distance > min_heliocentric_distance]
    psp_obs = [ob for ob in psp_obs
               if ob.duty_hours > min_duty_hours]
    dates = np.array([ob.date for ob in psp_obs])

    fig = plt.figure(figsize=(3*aspect, 3))
    gs = fig.add_gridspec((2), hspace=.05)
    ax = gs.subplots(sharex=1)
    ax[0].set_ylim(0,1000)#12000)
    ax[0].set_ylabel("Rate [/24h equiv.]")

    # Calculate and plot the scatter plot
    detecteds = np.array([ob.count_corrected for ob in psp_obs])
    duty_dayss = np.array([ob.duty_hours/(24) for ob in psp_obs])
    ax[0].scatter(dates,detecteds/duty_dayss,
               c="red",s=0.5,zorder=100,label="PSP detections")

    # Calculate and plot  scatter points' errorbars
    scatter_points_errors = get_detection_errors(detecteds)
    ax[0].errorbar(dates, detecteds/duty_dayss,
                scatter_points_errors/duty_dayss,
                c="red", lw=0, elinewidth=0.4,alpha=0.)

    # Evaluate the model
    lower_expected_counts = np.zeros(0)
    upper_expected_counts = np.zeros(0)
    mean_expected_counts = np.zeros(0)
    for i in tqdm(range(len(dates))):
        lower_e_count, upper_e_count, mean_e_count = get_predicted_range(
            r = psp_obs[i].heliocentric_distance,
            vr = psp_obs[i].heliocentric_radial_speed,
            vt = psp_obs[i].heliocentric_tangential_speed,
            duty_hours = psp_obs[i].duty_hours,
            b1s = b1s, b2s = b2s, c1s = c1s, c2s = c2s, v1s = v1s,
            sample_mu = sample_mu,
            sample_poiss = sample_poiss)
        lower_expected_counts = np.append(lower_expected_counts,
                                          lower_e_count)
        upper_expected_counts = np.append(upper_expected_counts,
                                          upper_e_count)
        mean_expected_counts = np.append(mean_expected_counts,
                                         mean_e_count)

    # Plot model line
    if smooth_model:
        mean_expected_counts = np.array(
                               [mu(np.mean(b1s),
                                   np.mean(b2s),
                                   np.mean(c1s),
                                   0,#np.mean(c2s),
                                   np.mean(v1s),
                                   ob.heliocentric_distance,
                                   ob.heliocentric_radial_speed,
                                   ob.heliocentric_tangential_speed)
                                for ob in psp_obs])*24*duty_dayss
    ax[0].plot(dates,model_prefact*mean_expected_counts/duty_dayss,
            c="blue",lw=0.5,zorder=101,label=f"{model_prefact}x SolO model")


    # Plot model errorbars
    ax[0].vlines(dates,
              model_prefact*lower_expected_counts/duty_dayss,
              model_prefact*upper_expected_counts/duty_dayss,
              colors="blue", lw=0.4, alpha=0.)
    ax[0].legend(facecolor='white', framealpha=1,
                 fontsize="x-small").set_zorder(200)

    # Relate the counts and the prediction
    preperi = np.array([ob.heliocentric_radial_speed < 0
                        for ob in psp_obs])
    postperi = np.invert(preperi)
    ax[1].scatter(dates[preperi],
               detecteds[preperi]
               /(model_prefact*mean_expected_counts[preperi]),
               s=0.5,c="firebrick",label="Pre-perihelion")
    ax[1].scatter(dates[postperi],
               detecteds[postperi]
               /(model_prefact*mean_expected_counts[postperi]),
               s=0.5,c="orangered",label="Post-perihelion")
    ax[1].set_ylim(1.01e-2,9.9e1)
    ax[1].set_yscale("log")
    xlo,xhi = ax[1].get_xlim()
    ax[1].hlines(1, xlo, xhi, colors="blue", lw=0.5)
    ax[1].set_xlim(xlo,xhi)
    ax[1].set_ylabel("Detection / model [1]")
    ax[1].legend(facecolor='white', framealpha=1,
                 fontsize="x-small").set_zorder(200)
    fig.show()

#%%
if __name__ == "__main__":
    plot_psp_data_solo_model()
