import numpy as np
import cdflib

from load_data import Observation
from load_data import load_all_obs
from load_data import list_cdf

from paths import all_obs_location
from paths import figures_location
from paths import l3_dust_location

AU_per_RS = 0.00465047

def normalize(v):
    """
    A simple function to normalize a given vector. 

    Parameters
    ----------
    v : np.array (1D)
        The vector to be normalized.

    Returns
    -------
    normalized : np.array (1D)
        The normlaized vector.

    """
    norm = np.linalg.norm(v)
    if norm == 0: 
       normalized = v
    normalized = v / norm
    return normalized


def project(projected,target):
    """
    Projection of the projected vector onto the target.

    Parameters
    ----------
    projected : np.array (1D)
        The vector to be projected.
    target : np.array (1D)
        The vector to be projected on.

    Returns
    -------
    projection : np.array (1D)
        The projection of the "projected" onto the "target".

    """
    projection = (np.dot(projected, target)) * target
    return projection


psp_obs = [ob for ob in load_all_obs(all_obs_location)
           if ob.duty_hours>0]


for file in list_cdf(l3_dust_location):
    cdf_file = cdflib.CDF(file)
    cdf_short_name = str(cdf_file.cdf_info().CDF)[
                         str(cdf_file.cdf_info().CDF).find("psp_fld_l3_")
                         :-4]

    YYYYMMDD = file[-16:-8]

    epochs = cdf_file.varget("psp_fld_l3_dust_V2_rate_epoch")
    rate_seg_starts = epochs - (epochs[1]-epochs[0])//2
    rate_seg_ends   = epochs + (epochs[1]-epochs[0])//2

    event_epochs = cdf_file.varget("psp_fld_l3_dust_V2_event_epoch")
    count_observed = np.zeros(len(epochs),dtype=int)

    for i,epoch in enumerate(epochs):
        count_observed[i] = len(event_epochs[
                                     (event_epochs > rate_seg_starts[i])
                                    *(event_epochs < rate_seg_ends[i]  )
                                             ])

    sc_x = cdf_file.varget("psp_fld_l3_dust_V2_rate_ej2000_x")*AU_per_RS
    sc_y = cdf_file.varget("psp_fld_l3_dust_V2_rate_ej2000_y")*AU_per_RS
    sc_z = cdf_file.varget("psp_fld_l3_dust_V2_rate_ej2000_z")*AU_per_RS
    pointing_z = cdf_file.varget("psp_fld_l3_dust_ej2000_pointing_sc_z_vector")

    azimuth = np.zeros(len(pointing_z))
    elevation = np.zeros(len(pointing_z))

    for i,pointing in enumerate(pointing_z):
        segment = np.argmax(np.cumsum(count_observed)>i)
        x = sc_x[segment] #AU
        y = sc_y[segment] #AU
        z = sc_z[segment] #AU
        r_normlaized = normalize(np.array([x,y,z]))
        pointing_projected_to_r = project(pointing,r_normlaized)
        pointing_residue = pointing - pointing_projected_to_r






