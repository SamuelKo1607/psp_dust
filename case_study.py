import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from numba import jit

from conversions import jd2date

def load_data(solo_file=os.path.join("data_synced","solo_flux_readable.csv"),
              psp_file=os.path.join("data_synced","psp_flux_readable.csv")):
    """
    Loads the observational data. 
    
    Parameters
    ----------
    solo_file : str, optional
        The path to the Solar Orbiter file. 
        The default is os.path.join("data_synced","solo_flux_readable.csv").
    psp_file : str, optional
        The path to the PSP file. 
        The default is os.path.join("data_synced","psp_flux_readable.csv").
    
    Returns
    -------
    solo_df : pandas.DataFrame
        The observational data from SolO.
    psp_df : pandas.DataFrame
        The observational data from SolO.
    
    """
    solo_df = pd.read_csv(solo_file)
    solo_df = solo_df[solo_df["Detection time [hours]"]>0]
    solo_df.insert(len(solo_df.columns),"Area front [m^2]",
                   10.34 * np.ones(len(solo_df.index)),
                   allow_duplicates=True)
    solo_df.insert(len(solo_df.columns),"Area side [m^2]",
                   8.24 * np.ones(len(solo_df.index)),
                   allow_duplicates=True)
    psp_df = pd.read_csv(psp_file)
    psp_df = psp_df[psp_df["Detection time [hours]"]>0]

    return solo_df, psp_df


def find_best_match(solo_df, psp_df):

    # 1st dimension is as in solo jd, 2nd dimension as in psp jd
    solo_r = np.tile(solo_df['Radial distance [au]'].to_numpy(),
                     (len(psp_df.index),1)).transpose()
    solo_vr = np.tile(solo_df['Radial velocity [km/s]'].to_numpy(),
                      (len(psp_df.index),1)).transpose()
    solo_vt = np.tile(solo_df['Tangential velocity [km/s]'].to_numpy(),
                      (len(psp_df.index),1)).transpose()
    psp_r = np.tile(psp_df['Radial distance [au]'].to_numpy(),
                    (len(solo_df.index),1))
    psp_vr = np.tile(psp_df['Radial velocity [km/s]'].to_numpy(),
                    (len(solo_df.index),1))
    psp_vt = np.tile(psp_df['Tangential velocity [km/s]'].to_numpy(),
                    (len(solo_df.index),1))

    badness = ((np.abs(solo_vr-psp_vr)/5)
                + (np.abs(solo_vt-psp_vt)/5)
                + (np.abs(solo_r-psp_r)/0.1))

    best = np.argwhere(badness==np.min(badness))[0]

    return (solo_df['Julian date'].to_numpy()[best[0]],
            psp_df['Julian date'].to_numpy()[best[1]])






#%%
if __name__ == "__main__":

    solo_df, psp_df = load_data()
    solo_jd, psp_jd = find_best_match(solo_df, psp_df)
    print(solo_jd,jd2date(solo_jd))
    print(psp_jd,jd2date(psp_jd))