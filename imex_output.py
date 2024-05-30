import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 600
import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')



def load_orbital_elements(file):
    df = pd.read_fwf(file, header=None, engine='python',
                     names=['computedjd',
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
    return df


#%%
if __name__ == "__main__":

    orbelts_file = os.path.join("data_synced","so_IMEX_orbelts.txt")
    so_summary_file = os.path.join("data_synced","data_so_summaryall.txt")

    orbelts_df = load_orbital_elements(orbelts_file)
    so_summary_df = load_so_summary(so_summary_file)

    f, ax = plt.subplots()
    e = orbelts_df["e"][orbelts_df["e"]<=1]
    plt.hist(e,bins=100)
    plt.xlabel("eccentricity")
    plt.text(.5, .95, r"$\mu(e)=\,$"+f"{np.mean(e):.2f}",
             ha='left', va='top', transform=ax.transAxes)
    plt.show()

    i = orbelts_df["i"][orbelts_df["e"]<=1]
    f, ax = plt.subplots()
    plt.hist(-np.abs(i-90)+90,bins=100)
    plt.xlabel("inclination")
    plt.text(.5,.95,r"$\mu(i)=\,$"+f"{np.mean(-np.abs(i-90)+90):.2f}",
             ha='left',va='top',transform=ax.transAxes)
    plt.text(.5,.85,r"$\%retro=\,$"+f"{100*np.sum(i>90)/np.sum(i>=0):.2f}",
             ha='left',va='top',transform=ax.transAxes)
    plt.show()

    









