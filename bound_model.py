import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
from tqdm.auto import tqdm

from ephemeris import get_approaches
from load_data import load_all_obs

from paths import psp_ephemeris_file

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600

GM=1.3271244002e20 #gravitational paramter of the Sun
AU = 149597870700 #astronomical unit, m




def possible_peri(r,
                  e,
                  grid=100):
    """
    A function to generate a grid of possible perihelia from which the 
    dust present at "r" might have originated, given "e".
    
    The grid is generated in a way that the points cover the range 
    in the most effective way, eg. interval from 1 to 2 would be sampled 
    with 5 points as 1.1, 1.3, 1.5, 1.7, 1.9 and weach of these represents 
    a range of 0.2.

    Parameters
    ----------
    r : float
        The heliocentric distance [AU].
    e : float
        The eccentricity.
    grid : float, optional
        How many points d owe need. The default is 100.

    Returns
    -------
    grid_peri : np.array of float
        The grid points in [AU].

    """
    max_peri = r
    min_peri = r*(1-e)/(1+e)
    grid_peri = np.linspace(min_peri,max_peri,grid*2+1)[1::2]
    return grid_peri


def perihelion_speed(perihelia,
                     e,
                     beta):
    """
    A function to find the perihelion speed of an object 
    with a given beta (F_rp/F_g) and eccentricity.

    Parameters
    ----------
    perihelia : float or np.array of float
        The perihalion distance [AU].
    e : float
        Eccentricity (between 0 and 1).
    beta : float
        Beta factor, i.e. gravity "reducer" due 
        to the radiation pressure force.

    Returns
    -------
    speed : float or np.array of float
        The preihelion speed [km/s]. 
        The shape is inherited from "perielia".

    """
    speed = np.sqrt(GM*(1-beta)*(1+e)/(perihelia*AU))
    return speed/1000


def instantaneous_speed(r,
                        perihelia,
                        e,
                        beta):
    """
    The function to get the immedaite radial and azimuthal speeds.

    Parameters
    ----------
    r : float
        The heliocentric distance [AU].
    perihelia : np.array of float
        The grid of perihelia where the dust may originate from [AU].
    e : float
        Eccentricity (between 0 and 1).
    beta : float
        Beta factor, i.e. gravity "reducer" due 
        to the radiation pressure force.

    Returns
    -------
    total_speed : np.array of float 
        The total speed of all the grains which originate in 
        perihelia, hence has the shape of perihelia variable [km/s].
    radial_speed : np.array of float
        The radial speed of all the grains which originate in 
        perihelia, hence has the shape of perihelia variable [km/s].
    azimuthal_speed : np.array of float
        The azimuthal speed of all the grains which originate in 
        perihelia, hence has the shape of perihelia variable [km/s].

    """
    # speed at the perihelion
    peri_speed = perihelion_speed(perihelia,e,beta)
    # Vis-viva
    total_speed = np.sqrt(GM*(1-beta)*(
        (2/(r*AU)) - ((1-e)/(perihelia*AU))    ))/1000
    # Angular momentum conservation
    azimuthal_speed = peri_speed * perihelia / r
    # Energy conservation, or Pythagoras theorem, as you like it
    radial_speed = np.sqrt(total_speed**2 - azimuthal_speed**2)

    return total_speed, radial_speed, azimuthal_speed


def bound_flux(r,
               v_rad,
               v_azim,
               S,
               C,
               e,
               beta,
               gamma=1.3):

    # bins of the original perihelion
    peri = possible_peri(r,e)
    # the original perihelion speed
    v_peri = perihelion_speed(peri,e,beta)
    # the immediate dust speed, decomposed
    v_tot_dust, v_rad_dust, v_azim_dust = instantaneous_speed(r,peri,v_peri,
                                                              e,beta)
    # weight the amount of dust by:
    # 1. velocity relative to periheioln veocity
    velocity_factor = v_peri/v_tot_dust
    # 2. density at perihelion relative to 1AU perihelion density
    distance_factor = peri**(-gamma)
    # 3. density of perihelion grid
    grid_density_factor = np.average(np.diff(peri))
    # all the scalings for the density
    dust_amount = velocity_factor * distance_factor * grid_density_factor
    # relativa radial speed (we will do the +- options later)
    relative_v_rad_single_leg = v_rad - v_rad_dust
    # relative azimuthal speed
    relative_v_azim_single_leg = v_azim - v_azim_dust
    # double the perihelia array, we want to include outbound and inbound
    peri_both_legs = np.append(peri,
                               peri)
    relative_v_azim = np.append(relative_v_azim_single_leg,
                                relative_v_azim_single_leg)
    relative_v_rad = np.append(relative_v_rad_single_leg,
                               -relative_v_rad_single_leg)

    # to be continued


