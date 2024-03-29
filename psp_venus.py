import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats
from scipy import interpolate

from load_data import Observation

from ephemeris import get_approaches
from ephemeris import get_phase_angle
from ephemeris import load_hae
from load_data import load_all_obs
from load_data import encounter_group
from conversions import date2jd
from conversions import YYYYMMDD2jd
from conversions import jd2date

from paths import psp_ephemeris_file
from paths import venus_ephemeris_file
from paths import mercury_ephemeris_file
from paths import jupiter_ephemeris_file

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600


def find_nearest(array, value):
    """
    Finds the index of the nearest value in the array.

    Parameters
    ----------
    array : np.array
        The array of interest. Not tested with other than 1D.
    value : float
        The value of interest.

    Returns
    -------
    idx : int
        The index.

    """
    array = np.asarray(array)
    idx = np.nan_to_num(np.abs(array - value),nan=1000).argmin()
    return idx


def angular_declination_from_body(HAE,
                                  body_ephem=venus_ephemeris_file):
    """
    Caluculated the angluar declination from a planetary plane.

    Parameters
    ----------
    HAE : np.array of float
        The vector (len=3) to be projected. Alternatively, an array of (n,3).
        to be projected.
    body_ephem : str, optional
        The ephemeris file for a body of interest, e.g. Venus or Mercury. 
        The default is venus_ephemeris_file.

    Returns
    -------
    None.

    """

    # fix the input vector
    if len(np.shape(HAE))==2:
        pass
    elif len(np.shape(HAE))==1:
        HAE = HAE.reshape(1,3)
    else:
        raise Exception(f"wrong HAE shape: {np.shape(HAE)}")

    body_jd, body_hae = load_hae(body_ephem)
    body_orbits = (body_jd
                   - body_jd[0])/224.7 #should work for Venus, Mercury
    quarter_index = find_nearest(body_orbits,0.25)

    # the body-of-interest plane
    center = np.array([0,0,0])
    vect_1 = body_hae[0]
    vect_2 = body_hae[quarter_index]
    # normal vector, pointing north
    body_plane_norm = ( np.cross(vect_1,vect_2)
                       / np.linalg.norm(np.cross(vect_1,vect_2)) )

    # row-wise normalization
    rownorm = np.linalg.norm(HAE, axis=1, keepdims=True)
    HAE = HAE / rownorm
    # tile the plane norm to be compatible with HAE
    plane_norm_tiled = np.tile(body_plane_norm,
                               np.shape(HAE)[0]).reshape(np.shape(HAE)[0],3)
    # dot product of unit vectors is the cos
    rowwise_dot = np.sum(HAE*plane_norm_tiled, axis=1)
    # row-wise declination angle from the normal (north), radians
    theta = np.arccos(np.clip(rowwise_dot, -1.0, 1.0))
    # declination from the plane, degrees. Positive means north.
    declination = -1*((theta*180/np.pi)-90)

    return declination


def groups_to_colors(groups):
    """
    Transalte the orbital groups to colors.

    Parameters
    ----------
    groups : np.array of int
        The orbital group.

    Returns
    -------
    colors : str
        The colors according to the dictionary.

    """
    thisdict = {1 : u"#E15554",
                2 : u"#E1BC29",
                3 : u"#3BB273",
                4 : u"#4D9DE0",
                5 : u"#7768AE"}
    colors = [thisdict[g] for g in groups]
    return colors


def get_counting_errors(counts,
                        prob_coverage = 0.9):
    """
    The function to calculate the errorbars for flux 
    assuming Poisson distribution and taking into account
    the number of detections.

    Parameters
    ----------
    counts : array of float
        Counts per day.
    prob_coverage : float, optional
        The coverage of the errobar interval. The default is 0.9.

    Returns
    -------
    err_plusminus_flux : np.array of float
        The errorbars, lower and upper bound, shape (2, n). 
        The unit is [1] (count).

    """

    counts_err_minus = -stats.poisson.ppf(0.5-prob_coverage/2,
                                          mu=counts)+counts
    counts_err_plus  = +stats.poisson.ppf(0.5+prob_coverage/2,
                                          mu=counts)-counts
    err_plusminus_counts = np.array([counts_err_minus,
                                   counts_err_plus])

    return err_plusminus_counts


def psp_approaches(max_approach_date = "20230630",
                   find_max = True,
                   averaging = 0):
    """
    Evaluates the max flux near each perihelion and returns various
    useful pieces of information about thee perihelia.

    Parameters
    ----------
    max_approach_date : str, optional
        The max date of interest, YYYYMMDD. 
        The default is "20230630".
    find_max : bool, optional
        Whether to look for the local maximum of the flux ner the perihelion,
        which would usually be 1-3 days before the perihelion and which would 
        correspond to the max of beta. If False, then the flux at perihelion 
        would be used, which often corresponds to the dip.
        The default is True.
    averaging : int, optional
        How many points before and after are to be taken into account. 
        If 0, then only the max point is analyzed. If 1, then 1 before 
        and 1 after are analyzed in addition, leading to 3 analyzed points
        in total. The default is 0.

    Raises
    ------
    Exception
        If not matched well enough. Indicates data quality issue.

    Returns
    -------
    approach_jds : array of float
        Julian dates of the approaches.
    groups : array of int
        The approach group number.
    peak_counts : array of float
        The peak counts.
    peak_duty_hours : array of float
        The duty hours corresponding to the counts.

    """
    max_approach_jd = YYYYMMDD2jd(max_approach_date)
    approach_jds = get_approaches(psp_ephemeris_file)
    approach_jds = approach_jds[approach_jds<=max_approach_jd]

    groups = np.array([encounter_group(i+1) #due to the counting convention
                       for i,jd
                       in enumerate(approach_jds)])

    psp_obs = load_all_obs()
    psp_jds = [obs.jd_center for obs in psp_obs]
    peak_counts = np.zeros(0)
    peak_duty_hours = np.zeros(0)
    for jd in approach_jds:
        nearest_index = find_nearest(psp_jds, jd)
        if find_max: #looking for the local maximum
            flux_subset = [obs.count_corrected / obs.duty_hours
                           for obs in psp_obs[nearest_index-20:
                                              nearest_index+20+1]]
            jd_subset = [obs.jd_center
                         for obs in psp_obs[nearest_index-20:
                                            nearest_index+20+1]]
            jd_of_loc_max = jd_subset[find_nearest(flux_subset,
                                                   np.nanmax(flux_subset))]
            nearest_index = find_nearest(psp_jds, jd_of_loc_max)
        else:
            pass
        if np.abs(psp_obs[nearest_index].jd_center - jd)>0.5 and not find_max:
            print(np.abs(psp_obs[nearest_index].jd_center - jd))
            raise Exception("jd not matched closely enough")
        else:
            jds = [obs.jd_center
                   for obs in psp_obs[nearest_index-averaging:
                                      nearest_index+averaging+1]]
            if np.max(np.diff(np.append(jds,jds[-1])))>0.5:
                raise Exception(f"poor data availability near {jd}")
            fluxes = [obs.count_corrected
                      for obs in psp_obs[nearest_index-averaging:
                                         nearest_index+averaging+1]]
            duty_hours = [obs.duty_hours
                          for obs in psp_obs[nearest_index-averaging:
                                             nearest_index+averaging+1]]
            peak_counts = np.append(peak_counts,np.sum(fluxes))
            peak_duty_hours = np.append(peak_duty_hours,np.sum(duty_hours))
    return approach_jds, groups, peak_counts, peak_duty_hours


def flux_at_peak(*kwargs):
    """
    Shows the reported peak flux as a function of time for each peak of 
    PSP detected flux.

    Parameters
    ----------
    *kwargs : various
        Passed into psp_approaches.

    Returns
    -------
    None.

    """
    (approach_jds,
     groups,
     peak_counts,
     peak_duty_hours) = psp_approaches(*kwargs)

    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(4, 3))
    point_errors = get_counting_errors(peak_counts)
    dates = [jd2date(jd) for jd in approach_jds]
    peak_rates = peak_counts / peak_duty_hours
    ax.scatter(dates, peak_rates, c=groups_to_colors(groups), s=8, zorder=100)
    ax.errorbar(dates,
                peak_rates,
                np.vstack((point_errors[0] / peak_duty_hours,
                           point_errors[1] / peak_duty_hours)),
                lw=0, elinewidth=0.4, c="grey")
    ax.set_ylabel(r"Peak flux [$/h]$")

#     df = pd.DataFrame(dict(dates=dates, y=y, label=labels))

# groups = df.groupby('label')

# # Plot
# fig, ax = plt.subplots()
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
# for name, group in groups:
#     ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
# ax.legend()

    fig.show()


def compare_flux_phase(ephem_file,
                       title=None,
                       *kwarg):
    """
    Shows the perihel flux as a function of 
    phase angle difference for a body of choice.

    Parameters
    ----------
    ephem_file : str
        Epehemris file for the body of choice (e.g. Venus).
    title : str, optional
        The suptitle for the plot, if any. The default is None.
    *kwarg : various
        Passed into psp_approaches.

    Returns
    -------
    None.

    """
    (approach_jds,
     groups,
     peak_counts,
     peak_duty_hours) = psp_approaches(*kwarg)

    phase_f_psp = get_phase_angle(psp_ephemeris_file)
    phases_psp = phase_f_psp(approach_jds)
    phase_f_body = get_phase_angle(ephem_file)
    phases_body = phase_f_body(approach_jds)

    phase_diff = phases_psp-phases_body
    phase_diff += (phase_diff<0)*360 #to be between 0 and 360 degrees

    err_plusminus_counts = get_counting_errors(peak_counts)
    group_means = np.zeros(0)
    for i,count in enumerate(peak_counts):
        mean_count = np.mean(peak_counts[groups==groups[i]])
        group_means = np.append(group_means,mean_count)
    relative_peak_counts = peak_counts / group_means
    min_max_relative_peak_count = (np.vstack((-err_plusminus_counts[0,:],
                                              err_plusminus_counts[1,:]))
                                   + peak_counts) / group_means

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(phase_diff/180*np.pi,relative_peak_counts,
               c=groups_to_colors(groups), s=8, zorder=100)
    ax.vlines(phase_diff/180*np.pi,
              min_max_relative_peak_count[0,:],
              min_max_relative_peak_count[1,:],
              color="grey")
    ax.set_rmax(1.5)
    ax.set_rticks([1])
    ax.set_rlabel_position(30)
    if title is not None:
        ax.set_title(title, va='bottom')
    fig.show()


def compare_flux_angle(reference_body_ephem,
                       title=None,
                       *kwarg):
    """
    Compares the relative observed rate to the declination from a body,
    that is for example Venus or Mercury.

    Parameters
    ----------
    reference_body_ephem : str
        The epehemeris file fro the reference body, e.g. Venus.
    title : str, optional
        Suptitle of the plot, if any. The default is None.
    *kwarg : various
        Passed down to psp_approaches.

    Returns
    -------
    None.

    """
    psp_jd, psp_hae = load_hae(psp_ephemeris_file)
    declination = angular_declination_from_body(
        psp_hae,
        body_ephem=reference_body_ephem)
    f_declination = interpolate.interp1d(psp_jd,declination,
                                         fill_value="extrapolate",kind=3)
    (approach_jds,
     groups,
     peak_counts,
     peak_duty_hours) = psp_approaches(*kwarg)

    #evaluate declination at approach jds
    approach_declinations = f_declination(approach_jds)

    err_plusminus_counts = get_counting_errors(peak_counts)
    group_means = np.zeros(0)
    for i,count in enumerate(peak_counts):
        mean_count = np.mean(peak_counts[groups==groups[i]])
        group_means = np.append(group_means,mean_count)
    relative_peak_counts = peak_counts / group_means
    min_max_relative_peak_count = (np.vstack((-err_plusminus_counts[0,:],
                                              err_plusminus_counts[1,:]))
                                   + peak_counts) / group_means

    fig, ax = plt.subplots()
    ax.scatter(approach_declinations,relative_peak_counts,
               c=groups_to_colors(groups), s=8, zorder=100)
    ax.vlines(approach_declinations,
              min_max_relative_peak_count[0,:],
              min_max_relative_peak_count[1,:],
              color="grey")
    ax.set_xlabel(r"Declination from the reference body plane [$deg$]")
    ax.set_ylabel(r"Observed rate / group average [$1$]")
    if title is not None:
        fig.suptitle(title)
    fig.show()




#%%

flux_at_peak()

for ephem, title in zip([venus_ephemeris_file,
                         mercury_ephemeris_file,
                         jupiter_ephemeris_file],
                        ["Venus","Mercury","Jupiter"]):
    compare_flux_phase(ephem, title=title)
    compare_flux_angle(ephem, title=title)
