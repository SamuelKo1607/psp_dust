import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
from tqdm.auto import tqdm
from multiprocessing import Pool
from functools import partial
from numba import jit

from conversions import jd2date, date2jd
from conversions import deg2rad
from conversions import GM, AU

from paths import psp_ephemeris_file
from paths import figures_location

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600


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


def rate_samples(sampled,
                 shield,
                 sample_size=100,
                 solo_file=os.path.join("data_synced",
                                        "solo_flux_readable.csv"),
                 psp_file=os.path.join("data_synced",
                                       "psp_flux_readable.csv")):
    """
    Provides sampled mu matrices of shape (sample_size,len(jd)), 
    separately for solo, psp, bound and beta.

    Parameters
    ----------
    sampled :np.array
        shape = (n,6)

    shield : bool
        Whether to use the model which assumes a 
        different shield sensitivity.
    sample_size : int, optional
        The sample size, i.e. the number of rows 
        in the poutput arrays. 
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
    sample = sampled[np.random.choice(sampled.shape[0],
                                      replace=False,
                                      size=sample_size),:]
    if shield:
        raise Exception("shield==True not implemented yet")
    else:
        mu = source_rate

    solo_input_df = pd.read_csv(solo_file)
    psp_input_df = pd.read_csv(psp_file)

    solo_prediction_bound = np.zeros(
        (0,len(solo_input_df["Julian date"])))
    solo_prediction_beta = np.zeros(
        (0,len(solo_input_df["Julian date"])))
    psp_prediction_bound = np.zeros(
        (0,len(psp_input_df["Julian date"])))
    psp_prediction_beta = np.zeros(
        (0,len(psp_input_df["Julian date"])))

    for i,e in enumerate(sample[:,0]):
        solo_prediction_bound = np.vstack((solo_prediction_bound,
            mu(
                solo_input_df["Radial velocity [km/s]"].to_numpy(),
                solo_input_df["Tangential velocity [km/s]"].to_numpy(),
                solo_input_df["Radial distance [au]"].to_numpy(),
                10.34 * np.ones(len(solo_input_df["Julian date"])),
                8.24  * np.ones(len(solo_input_df["Julian date"])),
                0     * np.ones(len(solo_input_df["Julian date"])),
                sample[i,0],
                0,
                sample[i,2],
                sample[i,3],
                sample[i,4],
                sample[i,5]
            )))
        solo_prediction_beta = np.vstack((solo_prediction_beta,
            mu(
                solo_input_df["Radial velocity [km/s]"].to_numpy(),
                solo_input_df["Tangential velocity [km/s]"].to_numpy(),
                solo_input_df["Radial distance [au]"].to_numpy(),
                10.34 * np.ones(len(solo_input_df["Julian date"])),
                8.24  * np.ones(len(solo_input_df["Julian date"])),
                0     * np.ones(len(solo_input_df["Julian date"])),
                0,
                sample[i,1],
                sample[i,2],
                sample[i,3],
                sample[i,4],
                sample[i,5]
            )))

        psp_prediction_bound = np.vstack((psp_prediction_bound,
            mu(
                psp_input_df["Radial velocity [km/s]"].to_numpy(),
                psp_input_df["Tangential velocity [km/s]"].to_numpy(),
                psp_input_df["Radial distance [au]"].to_numpy(),
                psp_input_df["Area front [m^2]"].to_numpy(),
                psp_input_df["Area side [m^2]"].to_numpy(),
                1 * np.ones(len(psp_input_df["Julian date"])),
                sample[i,0],
                0,
                sample[i,2],
                sample[i,3],
                sample[i,4],
                sample[i,5]
            )))
        psp_prediction_beta = np.vstack((psp_prediction_beta,
            mu(
                psp_input_df["Radial velocity [km/s]"].to_numpy(),
                psp_input_df["Tangential velocity [km/s]"].to_numpy(),
                psp_input_df["Radial distance [au]"].to_numpy(),
                psp_input_df["Area front [m^2]"].to_numpy(),
                psp_input_df["Area side [m^2]"].to_numpy(),
                1 * np.ones(len(psp_input_df["Julian date"])),
                0,
                sample[i,1],
                sample[i,2],
                sample[i,3],
                sample[i,4],
                sample[i,5]
            )))

    return (solo_prediction_bound, solo_prediction_beta,
            psp_prediction_bound, psp_prediction_beta)


def load_data(solo_file=os.path.join("data_synced","solo_flux_readable.csv"),
              psp_file=os.path.join("data_synced","psp_flux_readable.csv"),
              which="both",
              timeframe=(2450000.0,2500000.0),
              r_min=0.4):
    """
    Loads anf shapes the observational data. 

    Parameters
    ----------
    solo_file : str, optional
        The path to the Solar Orbiter file. 
        The default is os.path.join("data_synced","solo_flux_readable.csv").
    psp_file : str, optional
        The path to the PSP file. 
        The default is os.path.join("data_synced","psp_flux_readable.csv").
    which : str, optional
        Which SC to perform the fit for. 
        Allowed values: "solo", "psp", "both".
        The default is "both", in which case both are used.
    timeframe : tuple of float
        Min and max jd time.
    r_min : float
        Minimum heliocentric distance [AU].

    Raises
    ------
    Exception
        If the which option got an unknown value.

    Returns
    -------
    data : pandas.DataFrame
        The observational data shaped for MCMC fitting.

    """
    solo_df = pd.read_csv(solo_file)
    solo_df = solo_df[solo_df["Detection time [hours]"]>0]
    solo_df = solo_df[solo_df["Radial distance [au]"]>r_min]
    solo_df = solo_df[solo_df['Julian date']>timeframe[0]]
    solo_df = solo_df[solo_df['Julian date']<timeframe[1]]
    psp_df = pd.read_csv(psp_file)
    psp_df = psp_df[psp_df["Detection time [hours]"]>0]
    psp_df = psp_df[psp_df["Radial distance [au]"]>r_min]
    psp_df = psp_df[psp_df['Julian date']>timeframe[0]]
    psp_df = psp_df[psp_df['Julian date']<timeframe[1]]

    if which=="both":
        v_sc_r = np.append(solo_df["Radial velocity [km/s]"],
                           psp_df["Radial velocity [km/s]"])
        v_sc_t = np.append(solo_df["Tangential velocity [km/s]"],
                           psp_df["Tangential velocity [km/s]"])
        r_sc = np.append(solo_df["Radial distance [au]"],
                         psp_df["Radial distance [au]"])
        area_front = np.append(10.34 * np.ones(len(solo_df.index)),
                               6.11 * np.ones(len(psp_df.index)))
        area_side = np.append(8.24 * np.ones(len(solo_df.index)),
                              4.62 * np.ones(len(psp_df.index)))
        heat_shield = np.append(0 * np.ones(len(solo_df.index)),
                                1 * np.ones(len(psp_df.index)))
        obs_time = np.append(solo_df['Detection time [hours]'],
                             psp_df['Detection time [hours]'])
        measured = np.append(solo_df["Fluxes [/day]"].astype(int),
                             psp_df["Count corrected [/day]"].astype(int))
    elif which=="solo":
        v_sc_r = np.array(solo_df["Radial velocity [km/s]"])
        v_sc_t = np.array(solo_df["Tangential velocity [km/s]"])
        r_sc = np.array(solo_df["Radial distance [au]"])
        area_front = np.array(10.34 * np.ones(len(solo_df.index)))
        area_side = np.array(8.24 * np.ones(len(solo_df.index)))
        heat_shield = np.array(0 * np.ones(len(solo_df.index)))
        obs_time = np.array(solo_df['Detection time [hours]'])
        measured = np.array(solo_df["Fluxes [/day]"].astype(int))
    elif which=="psp":
        v_sc_r = np.array(psp_df["Radial velocity [km/s]"])
        v_sc_t = np.array(psp_df["Tangential velocity [km/s]"])
        r_sc = np.array(psp_df["Radial distance [au]"])
        area_front = np.array(6.11 * np.ones(len(psp_df.index)))
        area_side = np.array(4.62 * np.ones(len(psp_df.index)))
        heat_shield = np.array(1 * np.ones(len(psp_df.index)))
        obs_time = np.array(psp_df['Detection time [hours]'])
        measured = np.array(psp_df["Count corrected [/day]"].astype(int))
    else:
        raise Exception(f"unknown spacecraft: {which}")

    data = pd.DataFrame(data=np.array([v_sc_r,
                                       v_sc_t,
                                       r_sc,
                                       area_front,
                                       area_side,
                                       heat_shield,
                                       obs_time,
                                       measured]).transpose(),
                        index=np.arange(len(v_sc_r),dtype=int),
                        columns=["v_sc_r",
                                 "v_sc_t",
                                 "r_sc",
                                 "area_front",
                                 "area_side",
                                 "heat_shield",
                                 "obs_time",
                                 "measured"])

    return data


def log_prior(theta,
              handout=None):
    """
    A non-normalized prior. In nominal mode, the multivariate prior is 
    evaluated. Alternatively, one of the marginal prior functions is 
    handed out (e.g. for plotting.). 

    Parameters
    ----------
    theta : list of float
        The parameters: l_a, l_b, origin, e_v, beta, eff.

    handout : int or None, optional
        Which prior to provide as a function. If None, 
        the multivariate prior is evaluated. The default is None.

    Returns
    -------
    logpdf : float
        The non-normalized ln(pdf), nominal.

    """

    l_a = theta[0]
    l_b = theta[1]
    origin = theta[2]
    e_v = theta[3]
    beta = theta[4]
    eff = theta[5]

    logpriors = [lambda x : stats.gamma.logpdf(x,a=5,scale=2e-5),
                 lambda x : stats.gamma.logpdf(x,a=5,scale=2e-5),
                 lambda x : stats.gamma.logpdf(x,a=5,scale=1e-1),
                 lambda x : stats.gamma.logpdf(x,a=5,scale=2e-1),
                 lambda x : stats.beta.logpdf(x,3,3),
                 lambda x : stats.beta.logpdf(x,3,3)]

    if handout is None:
        l_a_pdf = logpriors[0]
        l_b_pdf = logpriors[1]
        origin_pdf = logpriors[2]
        e_v_pdf = logpriors[3]
        beta_pdf = logpriors[4]
        eff_pdf = logpriors[5]

        logpdf = (l_a_pdf(l_a)
                  + l_b_pdf(l_b)
                  + origin_pdf(origin)
                  + e_v_pdf(e_v)
                  + beta_pdf(beta)
                  + eff_pdf(eff))
        return logpdf
    else:
        return logpriors[handout]

@jit
def source_rate(v_sc_r, v_sc_t,
                r_sc,
                area_front, area_side,
                heat_shield,
                l_a, l_b,
                origin,
                e_v,
                beta,
                eff,
                e_a_r=-1.3):
    """
    The source rate, assuming the PSP with a different sensitivity.

    Parameters
    ----------
    v_sc_r : np.array of float
        SC radial speed, positive outwards, km/s.
    v_sc_t : np.array of float
        SC azimuthal speed, positive prograde, km/s.
    r_sc : np.array of float
        SC heliocentric distance, >0.
    area_front : np.array of float
        DESCRIPTION.
    area_side : np.array of float
        DESCRIPTION.
    heat_shield : np.array of float
        DESCRIPTION.
    l_a : float
        The amount of bound dust at 1AU, m^-2 s^-1.
    l_b : float
        The amount of beta dust at 1AU, m^-2 s^-1.
    origin : float
        The origin region of beta dust.
    e_v : float
        The exponent on velocity.
    beta : float
        The beta value of beta dust.
    eff : float
        The effectivity of detection on PSP, relative to SolO.
    e_a_r : TYPE, optional
        DESCRIPTION. The default is -1.3.

    Returns
    -------
    hourly_rate : float
        DESCRIPTION.

    """
    b = (beta+1)/2 #theta[4] is natively 0-1, but we want it 0.5-1

    # beta meteoroid contribution
    v_origin = 29.8*origin**(-0.5) #[km/s]
    v_azim_beta = v_origin * origin / r_sc #[km/s]
    v_total_beta = (((v_origin*1000)**2
                    + 2*GM*(1-b)*(1/AU)*(1/r_sc-1/origin))**0.5
                    )/1000 #[km/s]
    v_rad_beta = (v_total_beta**2 - v_azim_beta**2)**0.5
    v_rad_impact = v_rad_beta-v_sc_r
    v_azim_impact = v_azim_beta-v_sc_t
    v_impact_beta = (v_rad_impact**2 + v_azim_impact**2)**0.5

    v_total_beta_1au = (((v_origin*1000)**2
                         + 2*GM*(1-b)*(1/AU)*(1-1/origin))**0.5
                        )/1000 #[km/s]
    v_azim_beta_1au = v_origin * origin #[km/s]
    v_rad_beta_1au = (v_total_beta_1au**2 - v_azim_beta_1au**2)**0.5
    v_factor = (v_impact_beta / v_total_beta_1au) #[1]

    n_1au = l_b / (v_total_beta_1au*1000) #[m^-3]
    n_sc = n_1au * r_sc**(-2) * (v_rad_beta_1au/v_rad_beta) #[m^-3]

    impact_angle = np.arctan( v_azim_impact
                              / v_rad_impact )
    
    area = ( ( area_front  * np.cos(impact_angle)
               + area_side * np.sin(np.abs(impact_angle)) )
            * (1 - ((1-eff) * heat_shield)) ) #[m^2]

    rate_beta = (area
                 * n_sc
                 * (v_impact_beta*1000)
                 * v_factor**(e_v)
                 * 3600) #[/hour]

    #bound dust contribution
    v_azim_bound = 29.8*(r_sc**(-0.5)) #[km/s]

    n_1au = l_a / (v_azim_bound*1000) #[m^-3]
    n_sc = n_1au * (r_sc)**e_a_r #[m^-3]

    v_rad_impact = -v_sc_r #[km/s]
    v_azim_impact = v_azim_bound-v_sc_t #[km/s]
    v_impact_bound = (v_rad_impact**2 + v_azim_impact**2)**0.5 #[km/s]
    v_factor = (v_impact_bound / 29.8) #[1]

    impact_angle = np.arctan( v_azim_impact
                              / v_rad_impact )

    area = ( ( area_front  * np.cos(impact_angle)
               + area_side * np.sin(np.abs(impact_angle)) )
            * (1 - ((1-eff) * heat_shield)) ) #[m^2]

    rate_bound = (area
                  * n_sc
                  * (v_impact_bound*1000)
                  * v_factor**(e_v)
                  * 3600) #[/hour]

    hourly_rate = rate_beta + rate_bound

    return hourly_rate


def log_likelihood(theta,
                   data,
                   shield=False):
    """
    The log-likelihood of data, given theta.

    Parameters
    ----------
    theta : list
        Old theta.
    data : pd.DataFrame
        DESCRIPTION.
    shield : bool, optional
        Whether to assume PSP's heat shield different 
        from the rest of the SC. The default is True.

    Returns
    -------
    loglik : TYPE
        DESCRIPTION.

    """

    l_a = theta[0]
    l_b = theta[1]
    origin = theta[2]
    e_v = theta[3]
    beta = theta[4]
    eff = theta[5]

    if shield:
        raise Exception("shield==True not implemented yet")
    else:
        rate = source_rate(data["v_sc_r"].to_numpy(),
                           data["v_sc_t"].to_numpy(),
                           data["r_sc"].to_numpy(),
                           data["area_front"].to_numpy(),
                           data["area_side"].to_numpy(),
                           data["heat_shield"].to_numpy(),
                           l_a,
                           l_b,
                           origin,
                           e_v,
                           beta,
                           eff) * data["obs_time"]

    logliks = stats.poisson.logpmf(data["measured"],mu=rate)
    loglik = np.sum(logliks)

    return loglik


def proposal(theta,
             scale=1,
             family="normal"):
    """
    Generates a new theta (proposal value for MH) and checks that it is 
    actually within allowed values (prior).

    Parameters
    ----------
    theta : list
        Old theta.
    
    scale : float, optional
        The scale of change, smaller means smaller deviation of the proposal.

    family : str, optional
        The family of the proposal. One of "normal", "uniform". 
        The default is "normal."

    Returns
    -------
    proposed_theta : TYPE
        Proposed theta (allowed by the prior).

    """

    success = False
    while not success:
        if family == "normal":
            rand = np.random.normal(0,1,size=len(theta))
        elif family == "uniform":
            rand = np.random.uniform(-1,1,size=len(theta))
        else:
            raise Exception(f"unknonwn family: {family}")
        proposed_l_a = theta[0] + rand[0] * 2e-5 * scale
        proposed_l_b = theta[1] + rand[1] * 2e-5 * scale
        proposed_origin = theta[2] + rand[2] * 0.01 * scale
        proposed_e_v = theta[3] + rand[3] * 0.1 * scale
        proposed_beta = theta[4] + rand[4] * 0.05 * scale
        proposed_eff = theta[5] + rand[0] * 0.02 * scale
    
        proposed_theta = [proposed_l_a,
                          proposed_l_b,
                          proposed_origin,
                          proposed_e_v,
                          proposed_beta,
                          proposed_eff]
    
        success = np.isfinite(log_prior(proposed_theta))

    return proposed_theta


def step(theta,
         data,
         scale=0.075,
         family="normal",
         shield=False):
    """
    Performs a step, returns either the old or the new theta.

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    npdata : TYPE
        DESCRIPTION.
    scale : float, optional
        The scale of the proposed change.
    shield : bool, optional
        Whether to assume PSP's heat shield different 
        from the rest of the SC. The default is True.


    Returns
    -------
    theta : TYPE
        DESCRIPTION.
    change : bool
        Whether the new theta is actually new, or just the old one.

    """
    old_goodness = log_likelihood(theta, data, shield) + log_prior(theta)

    proposed_theta = proposal(theta,scale=scale,family=family)
    proposed_goodness = (log_likelihood(proposed_theta, data, shield)
                         + log_prior(proposed_theta))

    log_acc_ratio = proposed_goodness - old_goodness
    threshold = np.random.uniform(0,1)

    if np.exp(log_acc_ratio) > threshold:
        change = True
        theta = proposed_theta
    else:
        change = False

    return theta, change


def walk(pbar_row,
         nsteps,
         theta_start,
         data,
         stepscale,
         stepfamily="uniform",
         stepshield=False):
    sampled = np.zeros(shape=(0,6))
    theta = theta_start
    for i in tqdm(range(nsteps),position=pbar_row):
        theta, change = step(theta,
                             data,
                             stepscale,
                             family=stepfamily,
                             shield=stepshield)
        sampled = np.vstack((sampled,np.array(theta)))
    return sampled


def approx_mode(sampled):
    """
    Returns the approx mode for a multivariate sample of len n 
    for m variables, of shape (n,m). The output is an array of len m.

    Parameters
    ----------
    sampled : np.array of float 2D
        The matrix of n samples of m variables of the shape (n,m).

    Returns
    -------
    result : np.array of float 1D
        The approximate mode, length m.

    """
    params = sampled.shape[1]
    result = np.zeros(params)
    for p in range(params):
        std = np.std(sampled[:,p])
        decimals = int(-np.round(np.log10(std/10)))
        rounded = np.around(sampled[:,p],decimals)
        mode = stats.mode(rounded)[0]
        result[p] = mode
    return result


def show_marginals(samples,theta0,mode,loglik,filename=None):
    fig = plt.figure(figsize=(4,0.666*np.shape(samples)[1]))
    gs = fig.add_gridspec(np.shape(samples)[1], hspace=.6)
    ax = gs.subplots()

    for a, p, transform, lim, label, m, prior_domain in zip(
            ax, # a
            range(np.shape(samples)[1]), # p
            [lambda x : x,
             lambda x : x,
             lambda x : x,
             lambda x : 1+x,
             lambda x : (1+x)/2,
             lambda x : x], # transform
            [[0,3e-4],
             [0,3e-4],
             [0,0.4],
             [1,3],
             [0.5,1],
             [0,1]], # lim
            [r"$\lambda_a$",
             r"$\lambda_b$",
             r"$r_o$",
             r"$\epsilon_v$",
             r"$\beta$",
             r"$\eta$"], # label
            mode, # m
            [[0,1e-3],
             [0,1e-3],
             [0,0.5],
             [0,5],
             [0,1],
             [0,1]]): # prior_domain
        # Histogram posterior
        a.hist(transform(samples[:,p]),
               bins=int(np.sqrt(np.shape(samples)[0]/10)),density=True,
               histtype="step",color="tab:red",zorder=5)
        # Initial value and MAP
        ylim = a.get_ylim()
        a.vlines(transform(m),ylim[0],ylim[1],color="tab:red")
        a.vlines(transform(theta0[p]),ylim[0],ylim[1],color="tab:purple")
        a.text(.02, .9, str(m),
               ha='left', va='top', transform=a.transAxes, color="tab:red")
        # Functions prior
        logprior = log_prior(np.zeros(6),handout=p)
        domain = np.linspace(prior_domain[0],prior_domain[1],num=1000)
        prior_value = np.exp(logprior(domain))
        prior_value /= np.max(prior_value)
        domain_transformed = transform(domain)
        a.plot(domain_transformed,ylim[1]*prior_value,c="tab:purple")
        # Make-up
        a.set_ylim(ylim)
        a.set_xlim(lim)
        a.set_ylabel(label)

    fig.suptitle(f"{np.shape(samples)[0]} samples,"
                 +f" loglik = {loglik} \n"
                 +r"$\theta_{0}=$"+f" {theta0}", color="tab:purple")
    fig.tight_layout()
    if filename is not None:
        plt.savefig(os.path.join(figures_location,
                                 "mcmc_source",
                                 filename+'.png'),
                                 format='png', dpi=600)
        plt.close()
    else:
        fig.show()


def show_rate(samples,
              shield,
              filename=None,
              aspect=1.333,
              solo_file=os.path.join("data_synced",
                                     "solo_flux_readable.csv"),
              psp_file=os.path.join("data_synced",
                                    "psp_flux_readable.csv")):

    solo_input_df = pd.read_csv(solo_file)
    psp_input_df = pd.read_csv(psp_file)
    
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(aspect*3, 3),
                           sharex=True)

    for dataset, axis, flux_name in zip([solo_input_df,
                                         psp_input_df],
                                        [ax[0],ax[1]],
                                        ["Fluxes [/day]",
                                         "Count corrected [/day]"]):
        for i,df in enumerate([
                dataset[dataset['Radial distance [au]'] <= 0.4],
                dataset[dataset['Radial distance [au]'] > 0.4]]):
            point_errors = get_detection_errors(df[flux_name])
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
     psp_prediction_beta) = rate_samples(samples,shield,
                                         solo_file=solo_file,
                                         psp_file=psp_file)
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
                    in solo_input_df["Julian date"]],
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
                    in psp_input_df["Julian date"]],
                    np.mean(prediction, axis = 0),
                    color=color, label=label, lw=lw)

    for a in ax:
        a.set_ylim(bottom=0)
        a.set_ylim(top=30)
        a.set_ylabel(r"Detection rate $[/h]$")
        a.set
    ax[0].legend()

    fig.tight_layout()
    if filename is not None:
        plt.savefig(os.path.join(figures_location,
                                 "mcmc_source",
                                 filename+"_rate.png"),
                                 format='png', dpi=600)
        plt.close()
    else:
        fig.show()


def main(goal_length=1e5,
         burnin=10000,
         burnin_changes=100,
         theta0=[1e-04, 1e-04, 0.1, 1., 0.6, 0.5],
         shield=False,
         stepscale=0.075,
         cores=8,
         family="uniform",
         mute=False,
         filename=None,
         data=None):
    if data is None:
        data = load_data()
    theta = theta0
    changes = 0
    sampled = np.zeros(shape=(0,6))
    sampled = np.vstack((sampled,np.array(theta)))
    if not mute:
        print(f"Burn-in of {burnin} start")
    with tqdm(total=burnin) as pbar:
        while np.shape(sampled)[0]<burnin or changes<burnin_changes:
            theta, change = step(theta,data,stepscale,family,shield)
            if change:
                changes += change
            if not mute:
                pbar.update(1)
            sampled = np.vstack((sampled,np.array(theta)))
    if not mute:
        act_burnin = np.shape(sampled)[0]
        print(f"Burn-in of {act_burnin} done, continuing with {cores} cores")
        print(f"Burn-in acc. rate = {changes/act_burnin}")
        start = dt.datetime.now()
    with Pool(cores) as p:
        sprint = partial(walk,
                         nsteps=int(goal_length/cores)+1,
                         theta_start=theta,
                         data=data,
                         stepscale=stepscale,
                         stepfamily=family,
                         stepshield=shield)
        poolresults = p.map(sprint,np.arange(cores))
    for result in poolresults:
        sampled = np.vstack((sampled,result))
    mode = approx_mode(sampled[burnin:,:])
    loglik = log_likelihood(mode, data, shield)
    if not mute:
        diff = dt.datetime.now() - start
        seconds = diff.seconds + 24*3600*diff.days
        print(f"{sampled.shape[0]} samples produced in {seconds} seconds")
        print(f"mode: {mode}, loglik: {loglik}")
        show_marginals(sampled[burnin:,:],theta0,mode,loglik,filename=filename)
        show_rate(sampled[burnin:,:],shield=shield,filename=filename)
        print(filename+" done")
    return sampled[burnin:,:], mode




#%%
if __name__ == "__main__":
    pass

    sampled = main(goal_length=1e5,
                   burnin=10000,
                   theta0 = [1e-4,
                             1e-4,
                             0.2,
                             1.,
                             0.2,
                             0.5],
                    filename = "first_try")















