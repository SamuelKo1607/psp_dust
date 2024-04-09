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

from conversions import jd2date, date2jd
from conversions import deg2rad

from paths import psp_ephemeris_file
from paths import figures_location

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600


def load_data(solo_file=os.path.join("data_synced","solo_flux_readable.csv"),
              psp_file=os.path.join("data_synced","psp_flux_readable.csv"),
              which="both"):
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
    solo_df = solo_df[solo_df["Radial distance [au]"]>0.4]
    psp_df = pd.read_csv(psp_file)
    psp_df = psp_df[psp_df["Detection time [hours]"]>0]
    psp_df = psp_df[psp_df["Radial distance [au]"]>0.4]

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


def log_prior(theta):
    """
    A non-normalized prior.

    Parameters
    ----------
    theta : list of float
        The parameters: l_a, l_b, v_b_r, e_v, e_b_r, shield_miss_rate.

    Returns
    -------
    logpdf : float
        The non-normalized ln(pdf). 

    """

    l_a = theta[0]
    l_b = theta[1]
    v_b_r = theta[2]
    e_v = theta[3]
    e_b_r = theta[4]
    shield_miss_rate = theta[5]

    l_a_pdf = lambda x : stats.gamma.logpdf(x,a=5,scale=2e-5)
    l_b_pdf = lambda x : stats.gamma.logpdf(x,a=5,scale=2e-5)
    v_b_r_pdf = lambda x : stats.norm.logpdf(x,loc=50,scale=5)
    e_v_pdf = lambda x : stats.gamma.logpdf(x,a=5,scale=2e-1)
    e_b_r_pdf = lambda x : stats.beta.logpdf(x,4,4)
    shield_miss_rate_pdf = lambda x : stats.beta.logpdf(x,4,4)

    logpdf = (l_a_pdf(l_a)
              + l_b_pdf(l_b)
              + v_b_r_pdf(v_b_r)
              + e_v_pdf(e_v)
              + e_b_r_pdf(e_b_r)
              + shield_miss_rate_pdf(shield_miss_rate))

    return logpdf


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
    """
    The rate, assuming the heat shield of PSP having different sensitivity.

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
    v_b_r : float
        The beta veloctiy at 1AU, km/s.
    e_v : float
        The exponent on velocity.
    e_b_r : float
        The exponent on radial distance (beta).
    shield_miss_rate : float
        The miss rate on the heat shield of PSP.
    e_a_r : TYPE, optional
        DESCRIPTION. The default is -1.3.
    v_b_a : TYPE, optional
        DESCRIPTION. The default is 9.
    v_earth_a : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    hourly_rate : float
        DESCRIPTION.

    """
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


def homogeneous_psp_rate(v_sc_r, v_sc_t,
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
    """
    The rate, assuming the whole PSP has a different sensitivity. 
    Thw shield_miss_rate takes on the role of 
    the sensitivity relative to SolO.

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
    v_b_r : float
        The beta veloctiy at 1AU, km/s.
    e_v : float
        The exponent on velocity.
    e_b_r : float
        The exponent on radial distance (beta).
    shield_miss_rate : float
        The miss rate on the heat shield of PSP.
    e_a_r : TYPE, optional
        DESCRIPTION. The default is -1.3.
    v_b_a : TYPE, optional
        DESCRIPTION. The default is 9.
    v_earth_a : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    hourly_rate : float
        DESCRIPTION.

    """
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

    area = ( ( area_front  * np.cos(impact_angle)
               + area_side * np.sin(np.abs(impact_angle)) )
            * (1 - shield_miss_rate * heat_shield) )
    
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
    
    area = ( ( area_front  * np.cos(impact_angle)
               + area_side * np.sin(np.abs(impact_angle)) )
            * (1 - shield_miss_rate * heat_shield) )
    
    L_a = l_a * area * (v_factor)**(e_v+1) * (r_factor)**e_a_r
    
    #normalization to hourly rate, while L_i are in s^-1
    hourly_rate = 3600 * ( L_b + L_a )

    return hourly_rate


def log_likelihood(theta,
                   data,
                   shield=True):
    """
    The log-likelihood of data, given theta.

    Parameters
    ----------
    theta : list
        Old theta.
    data : TYPE
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
    v_b_r = theta[2]
    e_v = theta[3]
    e_b_r = theta[4]
    shield_miss_rate = theta[5]

    if shield:
        rate = usual_rate(data["v_sc_r"],
                          data["v_sc_t"],
                          data["r_sc"],
                          data["area_front"],
                          data["area_side"],
                          data["heat_shield"],
                          l_a,
                          l_b,
                          v_b_r,
                          e_v,
                          e_b_r,
                          shield_miss_rate) * data["obs_time"]
    else:
        rate = homogeneous_psp_rate(data["v_sc_r"],
                                    data["v_sc_t"],
                                    data["r_sc"],
                                    data["area_front"],
                                    data["area_side"],
                                    data["heat_shield"],
                                    l_a,
                                    l_b,
                                    v_b_r,
                                    e_v,
                                    e_b_r,
                                    shield_miss_rate) * data["obs_time"]


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
        proposed_v_b_r = theta[2] + rand[2] * 10 * scale
        proposed_e_v = theta[3] + rand[3] * 0.1 * scale
        proposed_e_b_r = theta[4] + rand[4] * 0.05 * scale
        proposed_shield_miss_rate = theta[5] + rand[0] * 0.02 * scale
    
        proposed_theta = [proposed_l_a,
                          proposed_l_b,
                          proposed_v_b_r,
                          proposed_e_v,
                          proposed_e_b_r,
                          proposed_shield_miss_rate]
    
        success = np.isfinite(log_prior(proposed_theta))

    return proposed_theta


def step(theta,
         data,
         scale=0.075,
         family="normal",
         shield=True):
    """
    Performs a step, returns either the old or the new theta.

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    data : TYPE
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
         stepfamily="normal",
         stepshield=True):
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
    params = sampled.shape[1]
    result = np.zeros(params)
    for p in range(params):
        std = np.std(sampled[:,p])
        decimals = int(-np.round(np.log10(std/10)))
        print(decimals)
        rounded = np.around(sampled[:,p],decimals)
        mode = stats.mode(rounded)[0]
        result[p] = mode
    return result


def show(samples,theta0,mode,filename=None):
    fig = plt.figure(figsize=(4,0.666*np.shape(samples)[1]))
    gs = fig.add_gridspec(np.shape(samples)[1], hspace=.6)
    ax = gs.subplots()

    for a, p, transform, lim, label, m in zip(
            ax,
            range(np.shape(samples)[1]),
            [lambda x : x,
             lambda x : x,
             lambda x : x,
             lambda x : 1+x,
             lambda x : -x-1.5,
             lambda x : x],
            [[0,3e-4],
             [0,3e-4],
             [20,80],
             [1,3],
             [-2.5,-1.5],
             [0,1],],
            [r"$\lambda_a$",
             r"$\lambda_b$",
             r"$v_{b,r}$",
             r"$\epsilon_v$",
             r"$\epsilon_{b,r}$",
             r"$\alpha_{shield}$"],
            mode):
        a.hist(transform(samples[:,p]),
               bins=int(np.sqrt(np.shape(samples)[0]/10)))
        ylim = a.get_ylim()
        a.vlines(transform(m),ylim[0],ylim[1],color="red")
        a.set_ylim(ylim)
        a.set_xlim(lim)
        a.text(.02, .9, str(m),
               ha='left', va='top', transform=a.transAxes, color="red")
        a.set_ylabel(label)

    fig.suptitle(f"""{np.shape(samples)[0]} samples \n
                 theta0 = {theta0} \n""")
    fig.tight_layout()
    if filename is not None:
        plt.savefig(os.path.join(figures_location,"mcmc",filename+'.png'),
                                 format='png', dpi=600)
        plt.close()
    else:
        fig.show()


def main(goal_length=1e5,
         burnin=10000,
         burnin_changes=100,
         theta0=[7.79e-05, 5.88e-05, 62.4, 1.6, 0.075, 0.742],
         shield=True,
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
    if not mute:
        diff = dt.datetime.now() - start
        seconds = diff.seconds + 24*3600*diff.days
        print(f"{sampled.shape[0]} samples produced in {seconds} seconds")
        print(f"mode: {mode}")
        show(sampled[burnin:,:],theta0,mode,filename=filename)
    return sampled[burnin:,:], mode




#%%
if __name__ == "__main__":

    sampled, mode = main(
        goal_length = 1e6,
        burnin = 20000,
        shield = True,
        theta0 = [7.79e-05,
                  5.88e-05,
                  62.4,
                  1.6,
                  0.075,
                  0.742],
        filename = "shield_good_start")

    sampled, mode = main(
        goal_length = 1e6,
        burnin = 20000,
        shield = False,
        theta0 = [7.79e-05,
                  5.88e-05,
                  62.4,
                  1.6,
                  0.075,
                  0.742],
        filename = "no_shield_good_start")

    sampled = main(goal_length = 1e6,
                    burnin = 20000,
                    theta0 = [7.79e-05,
                              5.88e-05,
                              62.4,
                              1.6,
                              0.075,
                              0.742],
                    filename = "solo_only",
                    data = load_data(which="solo"))

    sampled = main(goal_length = 1e6,
                    burnin = 20000,
                    theta0 = [7.79e-05,
                              5.88e-05,
                              62.4,
                              1.6,
                              0.075,
                              0.742],
                    filename = "psp_only",
                    data = load_data(which="psp"))














