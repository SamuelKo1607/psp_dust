import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from load_data import Observation
from load_data import load_list
from overplot_with_solo_result import get_poisson_range

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi']= 600

from paths import figures_location
from paths import all_obs_location


def plot_simple_flux(all_obs):

    dates = np.array([o.date for o in all_obs if o.rate_ucc>0])
    rates = np.array([o.rate_ucc for o in all_obs if o.rate_ucc>0])
    duties = np.array([o.duty_hours for o in all_obs if o.rate_ucc>0])
    colors = [f"C{o.encounter_group}" for o in all_obs if o.rate_ucc>0]

    lowers,uppers = get_poisson_range(3600*rates,duties,0.9)

    fig,ax = plt.subplots()
    ax.scatter(dates,rates,
               s=0.8,edgecolors="none",color=colors)
    ax.vlines(dates,lowers/(3600*duties),uppers/(3600*duties),
              lw=0.4,color=colors,alpha=0.4)
    ax.set_ylabel("Impact rate [/s]")
    ax.set_ylim(bottom=0)
    fig.show()
    plt.savefig(figures_location+'flux_corrected.png', format='png', dpi=600)


def main():
    all_obs = load_list("all_obs.pkl", all_obs_location)

    plot_simple_flux(all_obs)

    return all_obs


#%%
if __name__ == "__main__":
    all_obs = main()