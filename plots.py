import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from load_data import Observation
from load_data import load_list
from overplot_with_solo_result import get_poisson_range

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi']= 600

from paths import figures_location
from paths import all_obs_location


def plot_simple_flux(all_obs,
                     aspect=1.9,
                     zoom=0.8,
                     colorful=True):

    dates = np.array([o.date for o in all_obs if o.rate_ucc>0])
    rates = np.array([o.rate_ucc for o in all_obs if o.rate_ucc>0])
    duties = np.array([o.duty_hours for o in all_obs if o.rate_ucc>0])
    if colorful:
        colors = [f"C{o.encounter_group}" for o in all_obs if o.rate_ucc>0]
    else:
        colors = ["teal" for o in all_obs if o.rate_ucc>0]

    lowers,uppers = get_poisson_range(3600*rates,duties,0.9)

    fig,ax = plt.subplots(figsize=(2*aspect/zoom, 2/zoom))
    ax.scatter(dates,rates*86400,
               s=0.8/zoom,edgecolors="none",color=colors)
    ax.vlines(dates,lowers/(3600*duties)*86400,uppers/(3600*duties)*86400,
              lw=0.4/zoom,color=colors,alpha=0.4)
    ax.set_ylabel("Impact rate (corrected) [/day]")
    ax.set_ylim(bottom=0)
    if aspect > 1.8:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5,
                                                                7, 9, 11)))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlim(left = min(dates), right = max(dates))
    ax.tick_params(axis='x',which="minor",bottom=True,top=True)
    xlabels = ax.get_xticklabels()
    ax.set_xticklabels(xlabels, rotation=60,
                       ha="right", rotation_mode='anchor')
    ax.tick_params(labelsize="medium")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.tight_layout()
    fig.show()
    plt.savefig(figures_location+'flux_corrected.png', format='png', dpi=600)


def main():
    all_obs = load_list("all_obs.pkl", all_obs_location)

    plot_simple_flux(all_obs)

    return all_obs


#%%
if __name__ == "__main__":
    all_obs = main()