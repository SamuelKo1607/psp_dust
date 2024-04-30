import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
from numba import jit
from tqdm.auto import tqdm

from eccentricity_core import bound_flux_vectorized
from eccentricity_core import r_smearing
from ephemeris import get_approaches
from ephemeris import load_ephemeris
from load_data import encounter_group
from conversions import jd2date
from overplot_with_solo_result import get_detection_errors
from conversions import GM, AU

from paths import psp_ephemeris_file
from paths import figures_location

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600



def plot_maxima_zoom(data,
                     perihelia,
                     flux,
                     e,
                     max_perihelia=16,
                     aspect=2,
                     zoom=1.2,
                     days=7,
                     pointcolor="darkorange",
                     linecolor="black",
                     filename=None):

    # Getting the approaches
    approaches = np.linspace(1,16,
                             16,
                             dtype=int)

    approach_dates = np.array([jd2date(a)
                                  for a
                                  in perihelia])
    approach_groups = np.array([encounter_group(a)
                                   for a
                                   in approaches])

    dates = np.array([jd2date(jd) for jd in data["jd"]])
    post_approach_threshold_passages = np.array([
        np.min(dates[(dates>approach_date)
                     *(data["r_sc"]>0.4)])
        for approach_date in approach_dates])

    # Calculate the scatter plot
    point_errors = get_detection_errors(data["measured"])
    duty_hours = data["obs_time"]
    detecteds = data["measured"]
    scatter_point_errors = np.vstack((point_errors[0]
                                         / data["obs_time"],
                                      point_errors[1]
                                         / data["obs_time"]))

    # Caluclate the model line
    eff_rate_bound = flux

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
                line_rate = np.append(line_rate,flux[filtered_indices])
            sortmask = line_hourdiff.argsort()

            ax.plot(line_hourdiff[sortmask],
                    line_rate[sortmask],
                    c=linecolor,lw=1,zorder=101,label=f"e = {e}")
            max_y = ax.get_ylim()[1]
            ax.vlines([-passage_hours,passage_hours],
                      0,10000,
                      color="gray")

            ax.set_ylim(0,max_y)
            ax.set_xlim(-days*24,days*24)

    fig.tight_layout()

    if filename is not None:
        fig.savefig(figures_location+filename+".png",dpi=1200)

    fig.show()


def plot_perihelion(flux_front,
                    flux_side,
                    r,
                    v_r,
                    v_phi,
                    xmin,
                    xmax,
                    perihel,
                    ex,
                    loc):

    x = (np.arange(xmin,xmax)-(xmin+xmax)/2)/3
    flux = flux_front+flux_side
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x,flux_front[xmin:xmax],label="rad")
    ax1.plot(x,flux_side[xmin:xmax],label="azim")
    ax1.plot(x,flux[xmin:xmax],label="tot")
    ax1.vlines(0,0,1.1*np.max(flux[xmin:xmax]),color="tab:red")
    ax1.legend()
    ax1.set_xlabel("Time since perihelion [d]")
    ax1.set_ylabel("Dust detection rate [/s]")
    ax2.set_ylabel("SC speed [km/s]")
    ax2.hlines(0,min(x),max(x),color="grey",ls="dashed")
    ax2.plot(x,v_r[xmin:xmax],color="black",label=r"$v_{rad}$")
    ax2.plot(x,v_phi[xmin:xmax],color="grey",label=r"$v_{azim}$")
    ax2.legend(loc=0)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(min(x),max(x))
    fig.suptitle(f"peri: {perihel}; ecc: {ex}")
    fig.tight_layout()
    if loc is not None:
        plt.savefig(loc+f"{perihel}th_peri_{ex}"+".png",dpi=1200)
    plt.show()


def density_scaling(gamma=-1.3,
                    ex=0.001,
                    r_min=0.04,
                    r_max=1.1,
                    size=50000,
                    mu=GM,
                    loc=figures_location):
    """
    Analyzes if the slope of heliocentric distance distribution 
    has changed or not.

    Parameters
    ----------
    gamma : float, optional
        The slope. The default is -1.3.
    ex : float, optional
        Eccentricity. The default is 0.001.
    r_min : float, optional
        min perihelion. The default is 0.04.
    r_max : float, optional
        max perihelion. The default is 1.1.
    size : int, optional
        number of sampled orbits. The default is 50000.
    mu : float, optional
        gravitational parameter. The default is GM.
    loc : str, optional
        Figure target directory. The default is figures_location.

    Returns
    -------
    None.

    """

    r_peri_proposed = np.random.uniform(r_min/3,r_max,size)
    thresholds = ((r_peri_proposed)**(gamma+1))/max(r_peri_proposed**(gamma+1))
    r_peri = r_peri_proposed[thresholds > np.random.uniform(0,1,size)]
    spreaded = np.zeros(0)
    all_sampled = []
    for r in tqdm(r_peri):
        samples = r_smearing(r,ex,size=1000,burnin=200)
        all_sampled.append(samples)
    spreaded = np.reshape(all_sampled,newshape=(1000*len(r_peri)))

    bins = np.linspace(r_min,r_max,int((size/10)**0.5))
    bincenters = (bins[1:]+bins[:-1])/2

    hist_orig = np.histogram(r_peri,bins,weights=r_peri**(-gamma-1))
    hist_mod = np.histogram(spreaded,bins,weights=spreaded**(-gamma-1))

    fig,ax = plt.subplots()
    ax.hlines(1,r_min,r_max,"grey")
    ax.step(bincenters, hist_orig[0]/np.mean(hist_orig[0]),
            where='mid', label="starting")
    ax.step(bincenters, hist_mod[0]/np.mean(hist_mod[0]),
            where='mid', label="smeared")
    ax.legend(fontsize="small")
    ax.text(0.1,1.15,rf"e = {ex}, $\gamma$ = {gamma}")
    ax.set_xlabel("Heliocentric distance [AU]")
    ax.set_ylabel(r"$\gamma$-compensated pdf [arb.u.]")
    ax.set_ylim(0.8,1.2)
    ax.set_xlim(r_min,r_max)
    ax.set_aspect(1.5)
    fig.tight_layout()
    if loc is not None:
        plt.savefig(loc+f"spred_{gamma}_{ex}"+".png",dpi=1200)
    plt.show()


def load_ephem_data(ephemeris_file,
                    r_min=0,
                    r_max=0.5,
                    decimation=4):
    """
    Reads the ephemerides, returns distance, speeds while disregarding 
    the "z" component.

    Parameters
    ----------
    ephemeris_file : str
        Location of the ephemerides file.
    r_min : float, optional
        The minimum (exclusive) heliocentric distance. The default is 0.
    r_max : float, optional
        The maximum (exclusive) heliocentric distance. The default is 0.5.
    decimation : int, optional
        How much to decimate the data. The default is 4 = every 4th is kept.

    Returns
    -------
    data : pd.df
        All the data which "main" needs.

    """
    (jd,
     hae,
     hae_v,
     hae_phi,
     radial_v,
     tangential_v,
     hae_theta,
     v_phi,
     v_theta) = load_ephemeris(os.path.join("data_synced",
                                         "psp_ephemeris_noheader.txt"))#psp_ephemeris_file)

    x = hae[:,0]/(AU/1000) #AU
    y = hae[:,1]/(AU/1000)
    z = hae[:,2]/(AU/1000)

    vx = hae_v[:,0] #km/s
    vy = hae_v[:,1]
    vz = hae_v[:,2]

    # for i in [3,4]:
    #     per = perihelia[i]
    #     indices = indices = np.abs(jd-per)<5
    #     plt.plot(r_sc[indices],v_tot[indices])

    v_sc_r = np.zeros(len(x))
    v_sc_t = np.zeros(len(x))
    for i in range(len(x)):
        unit_radial = hae[i,0:2]/np.linalg.norm(hae[i,0:2])
        v_sc_r[i] = np.inner(unit_radial,hae_v[i,0:2])
        v_sc_t[i] = np.linalg.norm(hae_v[i,0:2]-radial_v[i]*unit_radial)
    r_sc = np.sqrt(x**2+y**2)
    area_front = np.ones(len(x))*6.11
    area_side = np.ones(len(x))*4.62

    data = pd.DataFrame(data=np.array([v_sc_r,
                                       v_sc_t,
                                       r_sc,
                                       area_front,
                                       area_side,
                                       jd]).transpose(),
                        index=np.arange(len(r_sc),dtype=int),
                        columns=["v_sc_r",
                                 "v_sc_t",
                                 "r_sc",
                                 "area_front",
                                 "area_side",
                                 "jd"])

    data = data[data["r_sc"]>r_min]
    data = data[data["r_sc"]<r_max]
    data = data[data.index % decimation == 0]

    return data


def main(data,
         ex=0.01,
         gamma=-1.3,
         loc=figures_location):

    perihelia = get_approaches(psp_ephemeris_file)[:16]

    r_vector = data["r_sc"].to_numpy()
    v_r_vector = data["v_sc_r"].to_numpy()
    v_phi_vector = (data["v_sc_t"].to_numpy())
    S_front_vector = data["area_front"].to_numpy()
    S_side_vector = data["area_side"].to_numpy()
    jd = data["jd"].to_numpy()

    flux_front = bound_flux_vectorized(
        r_vector = r_vector,
        v_r_vector = v_r_vector,
        v_phi_vector = v_phi_vector,
        S_front_vector = S_front_vector,
        S_side_vector = S_side_vector*0,
        ex = ex,
        beta = 0,
        gamma = gamma,
        n = 7e-9)
    flux_side = bound_flux_vectorized(
        r_vector = r_vector,
        v_r_vector = v_r_vector,
        v_phi_vector = v_phi_vector,
        S_front_vector = S_front_vector*0,
        S_side_vector = S_side_vector,
        ex = ex,
        beta = 0,
        gamma = gamma,
        n = 7e-9)

    for peri in [1,4,6,8,10]:
        peri_index = np.argmin(np.abs(jd-perihelia[peri]))
        bounds = (peri_index-40,peri_index+40)
        plot_perihelion(flux_front,flux_side,
                        r_vector,v_r_vector,v_phi_vector,
                        bounds[0],bounds[1],
                        peri,
                        ex,loc)

    #plot_maxima_zoom(data,perihelia,flux,e,filename=f"eccentricity_{e}")




#%%
if __name__ == "__main__":

    loc = os.path.join(figures_location,"eccentricity","")
    data = load_ephem_data(os.path.join("data_synced",
                                        "psp_ephemeris_2_noheader.txt"))
    for ex in [0.001,0.1,0.2,0.3,0.4]:
        density_scaling(ex=ex,size=500000,loc=loc)
        #main(data=data,ex=ex,loc=loc)



