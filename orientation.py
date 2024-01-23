import numpy as np
import cdflib
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from load_data import Observation
from load_data import load_all_obs
from load_data import list_cdf

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600

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


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def main(plot=True,
         save=False,
         limit_first_n_days=9999):
    """
    

    Parameters
    ----------
    plot : bool, optional
        Whether to show the plot. 
        The default is True.
    save : bool, optional
        Whether to save the plot. Dummy if plot==0. 
        The default is False.
    limit_first_n_days : int, optional
        How many days we want to process, up to the number of days available. 
        The default is 9999.

    Returns
    -------
    azimuth : np.array of float, 1D
        The unit vector projection to the line perpendicular to the LoS,
        prograte (azimuthal) component.
    elevation : np.array of float, 1D
        The unit vector projection to the line perpendicular to the LoS,
        HAE Z (solar north elevation) component.
    deviation : np.array of float, 1D
        The deviation of the shield normal form the LoS to the Sun, degrees.

    """
    psp_obs = [ob for ob in load_all_obs(all_obs_location)
               if ob.duty_hours>0]
    
    azimuth = np.zeros(0)
    elevation = np.zeros(0)
    deviation = np.zeros(0)
    iazimuth = np.zeros(0)
    ielevation = np.zeros(0)
    ideviation = np.zeros(0)
    
    cdfs = list_cdf(l3_dust_location)
    for file in tqdm(cdfs[:np.min([limit_first_n_days,len(cdfs)-1])]):
        azimuth = np.append(azimuth,iazimuth)
        elevation = np.append(elevation,ielevation)
        deviation = np.append(deviation,ideviation)

        iazimuth = np.zeros(0)
        ielevation = np.zeros(0)
        ideviation = np.zeros(0)

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
    
        for i,pointing in enumerate(pointing_z):
            segment = np.argmax(np.cumsum(count_observed)>i)
            x = sc_x[segment] #AU
            y = sc_y[segment] #AU
            z = sc_z[segment] #AU
            r_normlaized = normalize(np.array([x,y,z]))
            pointing_projected_to_r = project(pointing,
                                              r_normlaized)
            pointing_in_the_plane = pointing - pointing_projected_to_r
            #get a new base for the perpendicular plane
            plane_base_z = np.array([0,0,1])-project(np.array([0,0,1]),
                                                     r_normlaized)
            plane_base_x = -normalize(np.cross(r_normlaized,plane_base_z))
            #decompose the pointing vector in the new plane
            plane_x = np.dot(plane_base_x,pointing_in_the_plane)
            plane_z = np.dot(plane_base_z,pointing_in_the_plane)
    
            # projction onto the normal plane
            iazimuth = np.append(iazimuth,plane_x)
            ielevation = np.append(ielevation,plane_z)
    
            #deviation angle
            ideviation = np.append(ideviation,
                                  np.rad2deg(angle_between(-r_normlaized,
                                                           pointing)))
    azimuth = azimuth[~np.isnan(azimuth)]
    elevation = elevation[~np.isnan(elevation)]
    deviation = deviation[~np.isnan(deviation)]

    if plot:
        fig, ax = plt.subplots()
        heatmap, xedges, yedges = np.histogram2d(azimuth, elevation, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(heatmap.T, extent=extent, origin='lower')
        fig.colorbar(im, ax=ax, label=r"$count$")
        ax.set_xlabel("Prograde unit projection [1]")
        ax.set_ylabel("HAE Z unit projection [1]")
        if save:
            fig.savefig(figures_location+"orientation_projections_lin"+".png",
                        dpi=400)
        fig.show()

        fig, ax = plt.subplots()
        heatmap, xedges, yedges = np.histogram2d(azimuth, elevation, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(np.log10(heatmap).T, extent=extent, origin='lower')
        fig.colorbar(im, ax=ax, label=r"$log_{10}(count)$")
        ax.set_xlabel("Prograde unit projection [1]")
        ax.set_ylabel("HAE Z unit projection [1]")
        if save:
            fig.savefig(figures_location+"orientation_projections_log"+".png",
                        dpi=400)
        fig.show()
        
        fig, ax = plt.subplots()
        ax.hist(deviation,bins=100)
        ylim = ax.get_ylim()[1]
        ax.vlines(np.quantile(deviation,0.8),0,ylim,color="red")
        ax.text(np.quantile(deviation,0.8)+1,ylim/2,r"$80\%$",
                rotation=90,va='center',color="red")
        ylim = ax.set_ylim(0,ylim)
        ax.set_xlabel("Deviation angle [deg]")
        ax.set_ylabel("Count [1]")
        if save:
            fig.savefig(figures_location+"orientation_deviations"+".png",
                        dpi=400)
        fig.show()

    return azimuth, elevation, deviation

#%%
if __name__ == "__main__":
    azimuth, elevation, deviation = main(plot=True,
                                         save=True,
                                         limit_first_n_days=9999)



