import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
from numba import jit

from eccentricity_core import bound_flux_vectorized
from mcmc import load_data
from ephemeris import get_approaches
from load_data import encounter_group
from conversions import jd2date
from overplot_with_solo_result import get_detection_errors

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
                    xmin,
                    xmax,
                    perihel,
                    ex,
                    loc):

    flux = flux_front+flux_side
    plt.plot(flux_front[xmin:xmax],label="rad")
    plt.plot(flux_side[xmin:xmax],label="azim")
    plt.plot(flux[xmin:xmax],label="tot")
    plt.legend()
    plt.ylim(bottom=0)
    plt.suptitle(f"peri: {perihel}; ecc: {ex}")
    plt.savefig(loc+f"{perihel}th_peri_{ex}"+".png",dpi=1200)
    plt.show()


def density_scaling(gamma=-1.3,
                    ex=0,
                    r_min=0.04,
                    r_max=1.1,
                    size=1000000):

    r_peri_proposed = np.random.uniform(r_min,r_max,size)
    thresholds = ((r_peri_proposed)**(2-gamma))/(r_max**(2-gamma))
    r_peri = r_peri_proposed[thresholds > np.random.uniform(0,1,size)]



def main(ex=0.01,loc=figures_location):
    data = load_data(which="psp",r_min=0,r_max=0.5)
    perihelia = get_approaches(psp_ephemeris_file)[:16]

    flux_front = bound_flux_vectorized(
        r_vector = data["r_sc"].to_numpy(),
        v_r_vector = data["v_sc_r"].to_numpy(),
        v_phi_vector = data["v_sc_t"].to_numpy(),
        S_front_vector = data["area_front"].to_numpy(),
        S_side_vector = data["area_side"].to_numpy()*0,
        ex = ex,
        beta = 0,
        gamma = -1.3,
        n = 1e-8)
    flux_side = bound_flux_vectorized(
        r_vector = data["r_sc"].to_numpy(),
        v_r_vector = data["v_sc_r"].to_numpy(),
        v_phi_vector = data["v_sc_t"].to_numpy(),
        S_front_vector = data["area_front"].to_numpy()*0,
        S_side_vector = data["area_side"].to_numpy(),
        ex = ex,
        beta = 0,
        gamma = -1.3,
        n = 1e-8)

    plot_perihelion(flux_front,flux_side,0,100,1,ex,loc)
    plot_perihelion(flux_front,flux_side,300,400,4,ex,loc)
    plot_perihelion(flux_front,flux_side,520,600,6,ex,loc)
    plot_perihelion(flux_front,flux_side,700,800,8,ex,loc)
    plot_perihelion(flux_front,flux_side,900,950,10,ex,loc)

    #plot_maxima_zoom(data,perihelia,flux,e,filename=f"eccentricity_{e}")




#%%
if __name__ == "__main__":

    loc = os.path.join(figures_location,"eccentricity")
    for ex in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]:
        main(ex=ex,loc=loc)



