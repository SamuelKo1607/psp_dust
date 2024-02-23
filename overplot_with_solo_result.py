import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyreadr
import datetime as dt
from tqdm.auto import tqdm

from load_data import Observation
from load_data import load_all_obs
from load_data import encounter_group
from orientation import fetch_orientation
from conversions import date_to_tt2000
from conversions import jd2date
from ephemeris import get_approaches

from paths import all_obs_location
from paths import legacy_inla_champion
from paths import figures_location
from paths import psp_ephemeris_file


import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600



def mu(b1, b2, c1, c2, v1,
       r, vr, vt,
       add_bound=None,
       shield_compensation=None,
       bound_r_exponent=-1.3,
       area_front=6.11,
       area_side=4.62):
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
    add_bound : float, optional
        The additional rate [/h] of bound dust, normalized to the flux at 1AU.
        The default is None, in which case, no bound dust is added.
    r : float
        SC heliocentric distance
    vr : float
        SC radial velocity
    vt : float
        SC azimuthal velocity
    shield_compensation : float, optional
        Whether to account for a lower shield sensitivity. 
        A float between 0 and 1, where 1 means the same 
        sensitivity as the rest of the spacecraft.
    area_front : float, optional
        Front-side projection area of the spacecraft [m^2]. 
        The default is 6.11, i.e. PSP forntal projection, shield included.
    area_side : float
        Lateral projection area of the spacecraft [m^2]. 
        The default is 4.62, i.e. PSP lateral projection.

    Returns
    -------
    rate : float
        The predicted detection rate. The unit is [/h]. 

    """
    # beta
    v_front_beta = (v1-vr)
    v_side_beta = ((12*0.75/r)-vt)
    rate_beta = (
                    ((v_front_beta**2+v_side_beta**2)**0.5)/50
                )**(b1)*r**(b2)*c1 + c2

    if shield_compensation is None or v_front_beta<0:
        pass
    else:
        area_coeff = ( np.abs(v_front_beta)*area_front*shield_compensation +
                       np.abs(v_side_beta)*area_side
                     ) / ( np.abs(v_front_beta)*area_front
                           + np.abs(v_side_beta)*area_side )
        rate_beta *= area_coeff

    # bound
    if add_bound is None:
        rate_bound = 0
    else:
        v_front_bound = -vr
        v_side_bound = ((29.8/r)-vt)
        rate_bound = (
                        ((v_front_bound**2+v_side_bound**2)**0.5)/50
                    )**(b1)*r**(bound_r_exponent)*add_bound
        if shield_compensation is None or v_front_bound<0:
            pass
        else:
            area_coeff = ( np.abs(v_front_bound)*area_front*shield_compensation
                           + np.abs(v_side_bound)*area_side
                         ) / ( np.abs(v_front_bound)*area_front
                               + np.abs(v_side_bound)*area_side )
            rate_bound *= area_coeff

    rate = rate_beta + rate_bound
    return rate


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
                        add_bound=None,
                        shield_compensation=None,
                        sample_mu=100,
                        sample_poiss=100,
                        prob_coverage=0.9):
    """
    A function to asses the lo, mean and high expected counts 
    given the covariates for a specific day. 

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
    add_bound : float, optional
        The additional rate [/h] of bound dust, normalized to the flux at 1AU.
        The default is None, in which case, no bound dust is added.
    shield_compensation : float, optional
        Whether to account for a lower shield sensitivity. 
        A float between 0 and 1, where 1 means the same 
        sensitivity as the rest of the spacecraft.
    sample_mu : TYPE, optional
        DESCRIPTION. The default is 100.
    sample_poiss : TYPE, optional
        DESCRIPTION. The default is 100.
    prob_coverage : TYPE, optional
        DESCRIPTION. The default is 0.9.

    Returns
    -------
    lower_expected_count : float
        The low expected count, assuming duty_hours of observation
        and the hourly rate of mu. Correspond to the quantile of 
        (1-prob_coverage)/2.
    upper_expected_count : float
        The high expected count, assuming duty_hours of observation
        and the hourly rate of mu. Correspond to the quantile of 
        1-(1-prob_coverage)/2.
    mean_expected_count : float
        The mean expected count, assuming duty_hours of observation
        and the hourly rate of mu.

    """
    available_samples = len(b1s)
    sample_draw = np.random.choice(np.arange(available_samples),
                                   size = sample_mu)
    mus = [mu(b1s[i], b2s[i], c1s[i], c2s[i], v1s[i],
              r, vr, vt,
              add_bound = add_bound,
              shield_compensation = shield_compensation)
           for i in sample_draw]
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
                             aspect=1.2,
                             sample_mu=10,
                             sample_poiss=10,
                             add_bound=None,
                             smooth_model=True,
                             add_bg_term=True,
                             shield_compensation=None,
                             min_heliocentric_distance=0.0,
                             min_duty_hours=0.01,
                             prob_coverage=0.99,
                             threshold=0.5,
                             filename=None,
                             title=None):
    """
    A plot which shows how the old SolO model compares to PSP data. 

    Parameters
    ----------
    model_prefact : float, optional
        The multiplicative constant for the model line. The default is 0.59,
        which corresponds to the ratio between PSP's front side projection 
        area and SolO's front side projection area.
    aspect : float, optional
        The aspect ratio of the plot. The default is 1.2.
    sample_mu : int, optional
        Number of samples of the mean rate used to evaluate the fit lines. 
        The default is 10.
    sample_poiss : int, optional
        Number of Poisson rate samples used to evaluate the fit lines. 
        The default is 10.
    add_bound : float, optional
        The additional rate [/h] of bound dust, normalized to the flux at 1AU.
        The default is None, in which case, no bound dust is added.
    smooth_model : bool, optional
        Whether to show the idealized ratio rather than mean of sampled. 
        The default is True.
    add_bg_term : bool, optional
        Whether to include the background term. The default is True.
    shield_compensation : float, optional
        Whether to account for a lower shield sensitivity. 
        A float between 0 and 1, where 1 means the same 
        sensitivity as the rest of the spacecraft.
    min_heliocentric_distance : float, optional
        The min heliocentric distance of the point in order for it to 
        be shown [AU]. The default is 0..
    min_duty_hours : float, optional
        The minimum amount of time [hr] per interval needed for the point 
        to be shown. The default is 2..
    prob_coverage : float, optional
        The threshold for outliers. The default is 0.99.
    threshold : float, optional
        The thresholds to show at each orbit in AU, such as 0.5AU cross. 
        The default is 0.5AU. Not shown in None.
    filename : str, optional
        The filename of the .png to be saved. the default is None, in which
        case, the plot is not saved.

    Returns
    -------
    None.

    """

    b1s, b2s, c1s, c2s, v1s = read_legacy_inla_result(legacy_inla_champion)
    psp_obs = load_all_obs(all_obs_location)
    psp_obs = [ob for ob in psp_obs
               if ob.heliocentric_distance > min_heliocentric_distance]
    psp_obs = [ob for ob in psp_obs
               if ob.duty_hours > min_duty_hours]
    psp_obs = [ob for ob in psp_obs
               if ob.los_deviation < 10]
    dates = np.array([ob.date for ob in psp_obs])

    # gridspec inside gridspec
    fig = plt.figure(figsize=(4*aspect, 4))
    gs0 = mpl.gridspec.GridSpec(2, 1,
                                figure=fig, hspace=.0,  height_ratios=[1, 3])
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(2, 1,
                                               subplot_spec=gs0[1], hspace=.05)
    ax = [fig.add_subplot(gs0[0]),
          fig.add_subplot(gs1[0]),
          fig.add_subplot(gs1[1])]

    ax[1].set_ylabel("Rate [/24h equiv.]",horizontalalignment='center', y=0.9)

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
            add_bound = add_bound,
            sample_mu = sample_mu,
            sample_poiss = sample_poiss)
        lower_expected_counts = np.append(lower_expected_counts,
                                          lower_e_count)
        upper_expected_counts = np.append(upper_expected_counts,
                                          upper_e_count)
        mean_expected_counts = np.append(mean_expected_counts,
                                         mean_e_count)
    mus = np.array([mu(np.mean(b1s),
                       np.mean(b2s),
                       np.mean(c1s),
                       np.mean(c2s)*add_bg_term,
                       np.mean(v1s),
                       ob.heliocentric_distance,
                       ob.heliocentric_radial_speed,
                       ob.heliocentric_tangential_speed,
                       add_bound,
                       shield_compensation=shield_compensation)
                    for ob in psp_obs])*model_prefact

    # Calculate and plot the scatter plot
    detecteds = np.array([ob.count_corrected for ob in psp_obs])
    duty_dayss = np.array([ob.duty_hours/(24) for ob in psp_obs])
    for a in ax[0:2]:
        lower_ok, upper_ok = get_poisson_range(mus,
                                               duty_dayss*24,
                                               prob_coverage=prob_coverage)
        outlier = (detecteds>upper_ok) + (detecteds>upper_ok)
        inlier = (1-outlier).astype(bool)
        a.scatter(dates[inlier],detecteds[inlier]/duty_dayss[inlier],
                  c="red",s=0.5,zorder=100,label="PSP detections")
        a.scatter(dates[outlier],detecteds[outlier]/duty_dayss[outlier],
                  c="limegreen",s=1,zorder=102,
                  label=f"{sum(outlier)} outliers")

    # Calculate and plot scatter points' errorbars
    scatter_points_errors = get_detection_errors(detecteds)
    for a in ax[0:2]:
        a.errorbar(dates, detecteds/duty_dayss,
                   scatter_points_errors/duty_dayss,
                   c="red", lw=0, elinewidth=0.4,alpha=0.)

    # Plot model line
    if smooth_model:
        mean_expected_counts = mus*24*duty_dayss
    eff_rate = mean_expected_counts/duty_dayss
    for a in ax[0:2]:
        a.plot(dates,eff_rate,
               c="blue",lw=0.5,zorder=101,
               label=f"{model_prefact}x SolO model")

    # Plot threshold distance crossings
    if threshold is not None:
        for i,date in enumerate(dates[:-1]):
            if (psp_obs[i].heliocentric_distance > threshold
                and psp_obs[i+1].heliocentric_distance < threshold):
                ax[1].hlines(eff_rate[i],
                             date-dt.timedelta(days=10),
                             date+dt.timedelta(days=30),
                             zorder=102,color="k")

    # Plot model errorbars
    for a in ax[0:2]:
        a.vlines(dates,
                 model_prefact*lower_expected_counts/duty_dayss,
                 model_prefact*upper_expected_counts/duty_dayss,
                 colors="blue", lw=0.4, alpha=0.)



    ax[0].legend(facecolor='white', framealpha=1,
             fontsize="small").set_zorder(200)

    # Relate the counts and the prediction
    preperi = np.array([ob.heliocentric_radial_speed < 0
                        for ob in psp_obs])
    postperi = np.invert(preperi)
    ax[2].scatter(dates[preperi],
               detecteds[preperi]
               /(mean_expected_counts[preperi]),
               s=0.5,c="maroon",label="Pre-perihelion")
    ax[2].scatter(dates[postperi],
               detecteds[postperi]
               /(mean_expected_counts[postperi]),
               s=0.5,c="darkorange",label="Post-perihelion")

    ax[2].set_yscale("log")
    xlo,xhi = ax[2].get_xlim()
    ax[2].hlines(1, xlo, xhi, colors="blue", lw=0.5)
    ax[2].set_ylabel("Detection / model")
    ax[2].legend(facecolor='white', framealpha=1,
                 fontsize="small").set_zorder(200)

    ax[0].spines['bottom'].set_visible(False)
    ax[0].xaxis.tick_top()
    ax[0].xaxis.set_ticklabels([])
    ax[1].hlines(1e3, xlo, xhi, colors="gray", lw=0.5, ls="dashed")
    ax[1].spines['top'].set_visible(False)
    ax[1].xaxis.tick_bottom()
    ax[0].set_ylim(1001,1.05*np.max(mean_expected_counts/duty_dayss))
    ax[1].set_ylim(0,1001)
    ax[1].minorticks_off()
    ax[2].set_ylim(1.01e-2,3.3e2)
    # Plot the r<0.5AU region
    inside_05 = np.array([ob.heliocentric_distance < 0.5 for ob in psp_obs])
    for a in ax:
        a.fill_between(dates, 0, 1e10*inside_05,
                       lw=0, color="gray", alpha=0.2)
        a.set_xlim(xlo,xhi)

    if title is not None:
        fig.suptitle(title)

    if filename is not None:
        fig.savefig(figures_location+filename+".png",dpi=1200)

    fig.show()


def zoom_plot_maxima(max_perihelia=16,
                     add_bg_term=True,
                     shield_compensation=1,
                     add_bound=None,
                     aspect=1.5):

    # Getting the approaches
    approaches = np.linspace(1,16,16,dtype=int)
    approach_dates = np.array([jd2date(a)
                                  for a
                                  in get_approaches(psp_ephemeris_file)
                                  ][:max_perihelia])
    approach_groups = np.array([encounter_group(a)
                                   for a
                                   in approaches])

    # Reading INLA output
    b1s, b2s, c1s, c2s, v1s = read_legacy_inla_result(legacy_inla_champion)
    psp_obs = load_all_obs(all_obs_location)
    psp_obs = [ob for ob in psp_obs
               if ob.duty_hours > 0]
    psp_obs = [ob for ob in psp_obs
               if ob.los_deviation < 10]
    dates = np.array([ob.date for ob in psp_obs])

    # Evaluate the model
    mus = np.array([mu(np.mean(b1s),
                       np.mean(b2s),
                       np.mean(c1s),
                       np.mean(c2s)*add_bg_term,
                       np.mean(v1s),
                       ob.heliocentric_distance,
                       ob.heliocentric_radial_speed,
                       ob.heliocentric_tangential_speed,
                       add_bound,
                       shield_compensation=shield_compensation)
                    for ob in psp_obs])*0.59

    # Calculate the scatter plot
    detecteds = np.array([ob.count_corrected for ob in psp_obs])
    duty_dayss = np.array([ob.duty_hours/(24) for ob in psp_obs])
    lower_ok, upper_ok = get_poisson_range(mus,
                                           duty_dayss*24,
                                           prob_coverage=0.9)
    scatter_points_errors = get_detection_errors(detecteds)

    # Caluclate the model line
    mean_expected_counts = mus*24*duty_dayss
    eff_rate = mean_expected_counts/duty_dayss

    # Iterate the groups
    for group in set(approach_groups):
        fig, ax = plt.subplots(figsize=(4, 4/aspect))
        ax.set_ylabel("Rate [/24h equiv.]")
        ax.set_xlabel("Time after perihelia [h]")
        fig.suptitle(f"Encounter group {group}")

        line_hourdiff = np.zeros(0)
        line_rate = np.zeros(0)

        for approach_date in approach_dates[approach_groups==group]:
            filtered_indices = np.abs(dates-approach_date
                                      )<dt.timedelta(days=7)
            datediff = dates[filtered_indices]-approach_date
            hourdiff = [24*d.days + d.seconds/3600
                        for d in datediff]
            ax.scatter(hourdiff,
                       (detecteds[filtered_indices]
                        /duty_dayss[filtered_indices]),
                      c="orangered",s=0.6,zorder=100,label="PSP detections")
            ax.errorbar(hourdiff,
                        (detecteds[filtered_indices]
                         /duty_dayss[filtered_indices]),
                        (np.array([scatter_points_errors[0,filtered_indices],
                                   scatter_points_errors[1,filtered_indices]
                                   ])
                         /duty_dayss[filtered_indices]),
                       c="orangered", lw=0., elinewidth=0.3,alpha=0.3)
            line_hourdiff = np.append(line_hourdiff,hourdiff)
            line_rate = np.append(line_rate,eff_rate[filtered_indices])
        sortmask = line_hourdiff.argsort()
        ax.plot(line_hourdiff[sortmask],
                line_rate[sortmask],
                c="navy",lw=0.5,zorder=101)
        fig.show()


def plot_psp_overplot_linlin(add_bg_term=True,
                             shield_compensation=1,
                             add_bound=None,
                             ymax=11000,
                             min_heliocentric=0.,
                             min_duty_hours=0,
                             max_los_deviation=45,
                             aspect=2,
                             moldel_lw=0.8,
                             log=False,
                             filename=None,
                             title=None):
    """
    The plotting procedure of PSP data with model lines overplotted 
    for the presentation purposes.

    Parameters
    ----------
    add_bg_term : bool, optional
        Wether to include background dust. The default is True.
    shield_compensation : float, optional
        The front side detection efficiency. The default is 1.
    add_bound : float of None, optional
        The additional rate [/h] of bound dust, normalized to the flux at 1AU.
        The default is None, in which case, no bound dust is added.
    ymax : float, optional
        The upper limit on the Y-axis. The default is 11000.
    min_heliocentric : float, optional
        The threshod for the inclusion of points. The default is 0.
    min_duty_hours : float, optional
        The minimum data amount for the period needed for the point to 
        be included (exclusive). The default is 0.
    max_los_deviation : float, optional
        The maximum deviation in degrees from the LoS for the point 
        to be included (exclusive). The default is 10. 
    aspect : float, optional
        The aspect ratio of the plot. The default is 2.
    moldel_lw : float, optional
        The linewidth of the model line.
    log : bool, optional
        Whether to plot it semilogy.
    filename : str of None, optional
        The name or the file to be saved. Use None to avoid saving.
        The default is None.
    title : str or None, optional
        The suptitle of the plot. Use None to aoid making a suptitle. 
        The default is "PSP: SolO model with bg, shield eff. = 1".

    Returns
    -------
    None.

    """

    b1s, b2s, c1s, c2s, v1s = read_legacy_inla_result(legacy_inla_champion)
    psp_obs = load_all_obs(all_obs_location)
    psp_obs = [ob for ob in psp_obs
               if ob.heliocentric_distance >= min_heliocentric]
    psp_obs = [ob for ob in psp_obs
               if ob.duty_hours > min_duty_hours]
    psp_obs = [ob for ob in psp_obs
               if ob.los_deviation < max_los_deviation]
    dates = np.array([ob.date for ob in psp_obs])

    #Splitting the dates
    gaps = np.where(np.diff(dates) > dt.timedelta(days=7))[0] + 1
    chunks = np.split(np.arange(len(dates)), gaps)

    # Evaluate the model
    mus = np.array([mu(np.mean(b1s),
                       np.mean(b2s),
                       np.mean(c1s),
                       np.mean(c2s)*add_bg_term,
                       np.mean(v1s),
                       ob.heliocentric_distance,
                       ob.heliocentric_radial_speed,
                       ob.heliocentric_tangential_speed,
                       add_bound,
                       shield_compensation=shield_compensation)
                    for ob in psp_obs])*0.59

    # Calculate and plot the scatter plot
    fig, ax = plt.subplots(figsize=(4, 4/aspect))

    detecteds = np.array([ob.count_corrected for ob in psp_obs])
    duty_dayss = np.array([ob.duty_hours/(24) for ob in psp_obs])
    lower_ok, upper_ok = get_poisson_range(mus,
                                           duty_dayss*24,
                                           prob_coverage=0.9)
    ax.scatter(dates,detecteds/duty_dayss,
              c="orangered",s=0.6,zorder=100,label="PSP detections")

    # Calculate and plot scatter points' errorbars
    scatter_points_errors = get_detection_errors(detecteds)
    ax.errorbar(dates, detecteds/duty_dayss,
               scatter_points_errors/duty_dayss,
               c="orangered", lw=0., elinewidth=0.3,alpha=0.3)

    # Plot model line
    mean_expected_counts = mus*24*duty_dayss
    eff_rate = mean_expected_counts/duty_dayss
    for i, chunk in enumerate(chunks):
        if i == 0:
            label = f"{0.59}x SolO model"
        else:
            label = None
        ax.plot(dates[chunk],
                eff_rate[chunk],
                c="navy",lw=moldel_lw,zorder=101,
                label=label)

    ax.set_ylabel("Rate [/24h equiv.]")
    if log:
        ax.set_ylim(10,ymax)
        ax.set_yscale('log')
    else:
        ax.set_ylim(0,ymax)
    ax.legend(loc=2, frameon=True,
              facecolor='white', edgecolor='black').set_zorder(200)
    fig.tight_layout()
    if title is not None:
        fig.suptitle(title)
    if filename is not None:
        fig.savefig(figures_location+filename+".png",dpi=400)
        print(f"saved {filename}.png to {figures_location}")
    fig.show()




#%%
if __name__ == "__main__":

    """
    plot_psp_data_solo_model(add_bg_term=True,shield_compensation=None,
        filename="PSP_SolO_with_bg",
        title="PSP: SolO model with bg")

    plot_psp_data_solo_model(add_bg_term=False,shield_compensation=None,
        filename="PSP_SolO_without_bg",
        title="PSP: SolO model without bg")

    plot_psp_data_solo_model(add_bg_term=False,shield_compensation=0.5,
        filename="PSP_SolO_shield_coeff",
        title="PSP: SolO model without bg, shield eff. = 0.5")

    plot_psp_data_solo_model(add_bg_term=False,shield_compensation=0.3,
        add_bound=2.5,
        filename="PSP_SolO_shield_coeff_bound_found",
        title="PSP: SolO model no bg, shield eff. = 0.3, bound = 2.5")

    plot_psp_data_solo_model(add_bg_term=False,shield_compensation=0.43,
        add_bound=1.23,
        filename="PSP_SolO_shield_coeff_bound_grid_fit",
        title="PSP: SolO model no bg, shield + bound fit grid to all")

    plot_psp_data_solo_model(add_bg_term=False,shield_compensation=0.474,
        add_bound=0.545,
        filename="PSP_SolO_shield_coeff_bound_inla_fit",
        title="PSP: SolO model no bg, shield + bound fit INLA to all")

    plot_psp_data_solo_model(add_bg_term=False,shield_compensation=0.247,
        add_bound=2.71,
        filename="PSP_SolO_shield_coeff_bound_grid_fit_far",
        title="PSP: SolO model no bg, shield + bound fit grid to r>0.5")

    plot_psp_data_solo_model(add_bg_term=False,shield_compensation=0.28,
        add_bound=1.73,
        filename="PSP_SolO_shield_coeff_bound_inla_fit_far",
        title="PSP: SolO model no bg, shield + bound fit INLA to r>0.5")
    """

    zoom_plot_maxima()
    zoom_plot_maxima(add_bg_term=False,
                     shield_compensation=0.3,
                     add_bound=2.5)



    plot_psp_overplot_linlin(moldel_lw=0.3)

    plot_psp_overplot_linlin(ymax=600,min_heliocentric=0.3)

    plot_psp_overplot_linlin(ymax=600,min_heliocentric=0.3,
                             add_bg_term=False)

    plot_psp_overplot_linlin(ymax=600,min_heliocentric=0.3,
                             add_bg_term=False,
                             shield_compensation=0.5)

    plot_psp_overplot_linlin(ymax=600,min_heliocentric=0.3,
                             add_bg_term=False,
                             shield_compensation=0.3,
                             add_bound=2.5)



    plot_psp_overplot_linlin(ymax=5e4,moldel_lw=0.3,log=True)
    plot_psp_overplot_linlin(ymax=5e4,moldel_lw=0.3,log=True,
                             add_bg_term=False)
    plot_psp_overplot_linlin(ymax=5e4,moldel_lw=0.3,log=True,
                             add_bg_term=False,
                             shield_compensation=0.5)
    plot_psp_overplot_linlin(ymax=5e4,moldel_lw=0.3,log=True,
                             add_bg_term=False,
                             shield_compensation=0.3,
                             add_bound=2.5)




