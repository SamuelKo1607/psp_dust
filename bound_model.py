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
               S_front=6.11,
               S_side=4.62,
               speed_exponent=2.04,
               distance_exponent=-1.3,
               miss_rate_front=0.,
               C=5,
               e=0,
               beta=0):
    """
    

    Parameters
    ----------
    r : float
        Heliocentric distance of PSP [AU].
    v_rad : float
        Readial speed of PSP [km/s].
    v_azim : float
        Azimuthal speed of PSP [km/s].
    S_front : float, optional
        Surface area (front) of PSP [m^2], optional. 
        The default is 6.11.
    S_side : float, optional
        Surface area (lateral) of PSP [m^2], optional. 
        The default is 4.62.
    speed_exponent : float, optional
        The exponent on the dependence on speed
    distance_exponent : float, optional
        The dependence of the dust density on the heliocentric distance. 
        The default is -1.3.
    miss_rate_front : float, optional
        The probability that a grain is missed if it hits the front side.
        The default is 0.
    C : float, optional
        The amount of bound dust at 1AU [/m^2 /s] as detected 
        by a stationary object, optional. 
        The default is 5.
    e : float, optional
        Eccentricity of the bound dust, optional. 
        The default is 0.
    beta : float, optional
        The beta value fo the bound dust, optional. 
        The default is 0.


    Returns
    -------
    TBD flux as detected in that moment

    """

    # bins of the original perihelion
    peri = possible_peri(r,e)
    # the original perihelion speed
    v_peri = perihelion_speed(peri,e,beta)
    # the immediate dust speed, decomposed
    v_tot_dust, v_rad_dust, v_azim_dust = instantaneous_speed(r,peri,e,beta)

    # weight the amount of dust by:
    # 1. velocity relative to periheioln veocity inverse
    velocity_factor = v_peri/v_tot_dust
    # 2. density at perihelion relative to 1AU perihelion density
    distance_factor = peri**(distance_exponent)
    # 3. density of the perihelion grid -
    #        - how many particles does this one represent (volume)
    grid_density_factor = np.average(np.diff(peri)) * peri
    # all the scalings for the density
    dust_factors = velocity_factor * distance_factor * grid_density_factor
    # double the dust factors, we want to include inbound and outbound
    dust_factors = np.append(dust_factors/2,
                             dust_factors/2)
    # relative radial speed (hit on front = negative rel speed)
    relative_v_rad = np.append(v_rad - v_rad_dust,
                               v_rad + v_rad_dust)
    # relative azimuthal speed
    relative_v_azim = np.append(v_azim - v_azim_dust,
                                v_azim - v_azim_dust)
    # total relative speed
    relative_v = ((relative_v_rad**2 + relative_v_azim**2)**0.5)
    # raw amount present to be detected
    amount = ( C/len(peri) *
              dust_factors *
              (relative_v/30)**(speed_exponent) )
    # front area not accounting for the fornt side miss rate
    eff_area_front = S_front * relative_v_rad / relative_v
    # introducing the miss rate and making it all positive
    eff_area_front = ( (eff_area_front>0)*eff_area_front -
                       (eff_area_front<0)*eff_area_front*(1-miss_rate_front) )
    # lateral area
    eff_area_side = np.abs( S_side * relative_v_azim / relative_v )

    # divide by len(peri) and not twice that, since /2 is used in dust_factors
    rate = np.sum( amount / np.append(grid_density_factor,
                                      grid_density_factor) *
                  ( eff_area_front + eff_area_side ) )

    return rate


print(bound_flux(1,0,0))