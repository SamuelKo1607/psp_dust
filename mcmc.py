import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
from tqdm.auto import tqdm

from conversions import jd2date, date2jd
from conversions import deg2rad

from paths import psp_ephemeris_file
from paths import figures_location

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600


def load_data(solo_file=os.path.join("data_synced","solo_flux_readable.csv"),
              psp_file=os.path.join("data_synced","psp_flux_readable.csv")):

    solo_df = pd.read_csv(solo_file)
    solo_df = solo_df[solo_df["Detection time [hours]"]>0]
    solo_df = solo_df[solo_df["Radial distance [au]"]>0.4]
    psp_df = pd.read_csv(psp_file)
    psp_df = psp_df[psp_df["Detection time [hours]"]>0]
    psp_df = psp_df[psp_df["Radial distance [au]"]>0.4]

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
    

    Parameters
    ----------
    v_sc_r : TYPE
        DESCRIPTION.
    v_sc_t : TYPE
        DESCRIPTION.
    r_sc : TYPE
        DESCRIPTION.
    area_front : TYPE
        DESCRIPTION.
    area_side : TYPE
        DESCRIPTION.
    heat_shield : TYPE
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
    hourly_rate : TYPE
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


def log_likelihood(theta,
                   data):

    l_a = theta[0]
    l_b = theta[1]
    v_b_r = theta[2]
    e_v = theta[3]
    e_b_r = theta[4]
    shield_miss_rate = theta[5]

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

    logliks = stats.poisson.logpmf(data["measured"],mu=rate)
    loglik = np.sum(logliks)

    return loglik




def proposal(theta,scale=1):
    """
    Generates a new theta (proposal value for MH) and checks that it is 
    actually within allowed values (prior).

    Parameters
    ----------
    theta : TYPE
        Old theta.
    
    scale : float
        The scale of change, smaller means smaller deviation of the proposal.

    Returns
    -------
    proposed_theta : TYPE
        Proposed theta (allowed by the prior).

    """

    success = False
    while not success:
        proposed_l_a = theta[0] + np.random.uniform(-1,1) * 2e-5 * scale
        proposed_l_b = theta[1] + np.random.uniform(-1,1) * 2e-5 * scale
        proposed_v_b_r = theta[2] + np.random.uniform(-1,1) * 10 * scale
        proposed_e_v = theta[3] + np.random.uniform(-1,1) * 0.1 * scale
        proposed_e_b_r = theta[4] + np.random.uniform(-1,1) * 0.05 * scale
        proposed_shield_miss_rate = (theta[5]
                                     + np.random.uniform(-1,1) * 0.02 * scale)
    
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
         scale=0.075):
    """
    Performs a step, returns either the old or the new theta.

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    theta : TYPE
        DESCRIPTION.
    change : bool
        Whether the new theta is actually new, or just the old one.

    """
    old_goodness = log_likelihood(theta, data) + log_prior(theta)

    proposed_theta = proposal(theta,scale=scale)
    proposed_goodness = (log_likelihood(proposed_theta, data)
                         + log_prior(proposed_theta))

    log_acc_ratio = proposed_goodness - old_goodness
    threshold = np.random.uniform(0,1)

    if np.exp(log_acc_ratio) > threshold:
        change = True
        theta = proposed_theta
    else:
        change = False

    return theta, change


def show(samples,theta0,filename=None):

    #fig, ax = plt.subplots(nrows=np.shape(samples)[1],ncols=1,
    #                       figsize=(3, 0.75*np.shape(samples)[1]))

    fig = plt.figure(figsize=(4,0.666*np.shape(samples)[1]))
    gs = fig.add_gridspec(np.shape(samples)[1], hspace=.6)
    ax = gs.subplots()

    for a, p, transform, lim, label in zip(ax,
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
                           r"$\alpha_{shield}$"]):
        a.hist(transform(samples[:,p]),
               bins=int(np.sqrt(np.shape(samples)[0]/10)))
        a.set_xlim(lim)
        a.set_ylabel(label)

    fig.suptitle(f"""{np.shape(samples)[0]} samples \n
                 theta0 = {theta0} \n""")
    fig.tight_layout()
    if filename is not None:
        plt.savefig(os.path.join(figures_location,filename+'.png'),
                                 format='png', dpi=600)
    fig.show()


def main(goal_length=10000,
         goal_changes=100,
         theta0=[7.79e-05, 5.88e-05, 62.4, 1.6, 0.075, 0.742],
         mute=False,
         stepscale=0.075,
         burnin=1000):
    data = load_data()
    theta = theta0
    changes = 0
    goal_tot = int(goal_length+burnin)
    sampled = np.zeros(shape=(0,6))
    sampled = np.vstack((sampled,np.array(theta)))

    with tqdm(total=goal_tot) as pbar:
        while np.shape(sampled)[0]<goal_tot or changes<goal_changes:
            theta, change = step(theta,data,scale=stepscale)
            if change:
                changes += change
            if not mute:
                pbar.update(1)
            sampled = np.vstack((sampled,np.array(theta)))

    if not mute:
        print(f"goal length: {int(goal_length)} (+{int(burnin)} burn-in)")
        print(f"acc. rate = {changes/goal_tot}")
        print(dt.datetime.now())
        show(sampled[burnin:,:],theta0)
    return sampled[burnin:,:]

#%%
if __name__ == "__main__":

    sampled = main(goal_length = 1e5,
                   theta0 = [7.79e-05,
                             5.88e-05,
                             62.4,
                             1.6,
                             0.075,
                             0.742])

    sampled = main(goal_length = 1e5,
                   theta0 = [1e-04,
                             1e-04,
                             50,
                             1,
                             0.5,
                             0.5])















