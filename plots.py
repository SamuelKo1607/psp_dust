import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from load_data import Observation
from load_data import load_list

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi']= 600

from paths import figures_location
from paths import all_obs_location


def plot_simple_flux(all_obs):

    dates = [o.date for o in all_obs]
    rates = [o.rate_ucc for o in all_obs]
    colors = [f"C{o.encounter_group}" for o in all_obs]

    fig,ax = plt.subplots()
    ax.scatter(dates,rates,
               s=0.3,edgecolors="none",color=colors)
    ax.set_ylabel("Impact rate [/s]")
    fig.show()
    plt.savefig(figures_location+'flux_corrected.png', format='png', dpi=600)


def main():
    all_obs = load_list("all_obs.pkl", all_obs_location)

    plot_simple_flux(all_obs)

    return all_obs


#%%
if __name__ == "__main__":
    all_obs = main()