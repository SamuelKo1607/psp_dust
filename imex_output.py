import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import datetime as dt
from conversions import jd2date
from conversions import date2jd
from ephemeris import fetch_heliocentric
from paths import solo_ephemeris_file
from paths import psp_ephemeris_file
mpl.rcParams['figure.dpi']= 600
import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')



def load_orbital_elements(file):
    df = pd.read_fwf(file, header=None, engine='python',
                     names=['computedjd',
                            'mass_kg',
                            'a_km',
                            'e',
                            'i',
                            'Omega',
                            'omega',
                            'M_deg'])
    return df


def load_so_summary(file):
    df = pd.read_csv(file, header=None, sep="           ", engine='python',
                     names=['spkid',
                            'flux0_/m2d',
                            'flux1_/m2d',
                            'flux2_/m2d',
                            'flux3_/m2d',
                            'flux4_/m2d',
                            'flux5_/m2d',
                            'flux6_/m2d',
                            'flux7_/m2d',
                            'meanspeed0_km/s',
                            'meanspeed1_km/s',
                            'meanspeed2_km/s',
                            'meanspeed3_km/s',
                            'meanspeed4_km/s',
                            'meanspeed5_km/s',
                            'meanspeed6_km/s',
                            'meanspeed7_km/s',
                            'ephemepoch'])
    df['mass0_kg'] = 1e-8
    df['mass1_kg'] = 1.64e-8
    df['mass2_kg'] = 4.39e-8
    df['mass3_kg'] = 1.93e-7
    df['mass4_kg'] = 1.39e-6
    df['mass5_kg'] = 1.64e-5
    df['mass6_kg'] = 3.16e-4
    df['mass7_kg'] = 1e-2
    df['ephemjd']  = df['ephemepoch']/(3600*24)+2451544.5000000
    df['flux_tot_/m2d'] = sum([df[f'flux{i}_/m2d'] for i in range(8)])
    return df


#%%
if __name__ == "__main__":

    f_hel_r, *rest = fetch_heliocentric(psp_ephemeris_file,cache_psp=False)
    orbelts_file = os.path.join("data_synced","so_orbelts_mass.txt")
    so_summary_file = os.path.join("data_synced","so_summaryall.txt")

    orbelts_df = load_orbital_elements(orbelts_file)
    so_summary_df = load_so_summary(so_summary_file)

    # Eccentricity
    f, ax = plt.subplots()
    e = orbelts_df["e"][orbelts_df["e"]<=1]
    ax.hist(e,bins=100,density=True,label="all",alpha=0.3)
    e = orbelts_df["e"][(orbelts_df["e"]<=1)*(orbelts_df["mass_kg"]<=1e-8)]
    ax.hist(e,bins=100,density=True,label=r"$m<1^{-7}$",alpha=0.3)
    ax.set_xlabel("eccentricity")
    ax.text(.5, .95, r"$\mu(e)=\,$"+f"{np.mean(e):.2f}",
             ha='left', va='top', transform=ax.transAxes)
    ax.legend(loc=2)
    plt.show()

    # Inclination
    f, ax = plt.subplots()
    i = orbelts_df["i"][orbelts_df["e"]<=1]
    ax.hist(-np.abs(i-90)+90,bins=100,density=True,
            label="all",alpha=0.3)
    i = orbelts_df["i"][(orbelts_df["e"]<=1)*(orbelts_df["mass_kg"]<=1e-8)]
    ax.hist(-np.abs(i-90)+90,bins=100,density=True,
            label=r"$m<1^{-7}$",alpha=0.3)
    ax.set_xlabel("inclination")
    ax.text(.5,.95,r"$\mu(i)=\,$"+f"{np.mean(-np.abs(i-90)+90):.2f}",
             ha='left',va='top',transform=ax.transAxes)
    plt.text(.5,.85,r"$\%retro=\,$"+f"{100*np.sum(i>90)/np.sum(i>=0):.2f}",
             ha='left',va='top',transform=ax.transAxes)
    ax.legend(loc=2)
    plt.show()

    # SolO dust flux
    f, ax = plt.subplots()
    ax2 = ax.twinx()
    jd = so_summary_df["ephemjd"]
    r = f_hel_r(jd)
    date = [jd2date(j) for j in jd]
    flux = so_summary_df["flux_tot_/m2d"]
    ax.semilogy(date,flux,label="all")
    flux = so_summary_df["flux0_/m2d"]
    ax.semilogy(date,flux,label="bin0")
    ax2.plot(date,r,"k")
    ax2.set_ylim(0,1.5)
    ax.set_ylabel(r"flux [$m^{-2} s^{-1}$]")
    ax.set_xlim(dt.datetime(2021,7,10),dt.datetime(2021,8,20))
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[10,20,30]))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(),
                  rotation=45, ha='right')
    ax.legend()
    plt.show()

    # SolO flux vs r
    f, ax = plt.subplots()
    jd = so_summary_df["ephemjd"]
    r = f_hel_r(jd)
    date = [jd2date(j) for j in jd]
    flux = sum([so_summary_df[f"flux{i}_/m2d"] for i in range(8)][:1])
    ax.scatter(r[jd<date2jd(dt.datetime(2024,1,1))],
               flux[jd<date2jd(dt.datetime(2024,1,1))],
               label="bin0")
    ax.set_yscale("log")
    ax.set_xlabel("heliocentric distance [AU]")
    ax.set_ylabel(r"flux [$m^{-2} s^{-1}$]")
    plt.show()








