import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600

AU = 149597870.7 #astronomical unit, km
GM = 1.327e20 #gravity of Sun, m3 s-2


def orbital_velocity(r,e=0):
    """
    The function for the orbital velocity for either perihelion or aphelion.

    Parameters
    ----------
    r : float
        The distance from the Sun [AU]. Is either perihelion or aphelion.
    e : float, optional
        Eccentricity, positive if we need perihelion, negative for apheion. 
        The default is 0.

    Returns
    -------
    v : float
        the speed in km/s
    """
    a = r/(1-e)
    v = np.sqrt(GM*(2/(r*AU*1000) - 1/(a*AU*1000)))/1000
    return v

def beta_velocity(a,
                  r,
                  e=0,
                  beta=0,
                  azimuthal=True,
                  radial=True):
    """
    The dust speed at different heliocentric distance, given it was released 
    from a given initial orbit.

    Parameters
    ----------
    a : float
        initial orbit in AU.
    r : np.array of float
        The heliocentric dostance in AU.
    e : float, optional
        initial eccentricity. The default is 0.
    beta : float, optional
        The beta parameter. The default is 0.
    azimuthal : bool, optional
        whether to include the azimuthal component. 
        The default is True.
    radial : TYPE, optional
        whether to include the radial component. 
        The default is True.

    Returns
    -------
    np.array of float
        The resulting velocity profile [km/s].
    """

    v_initial = orbital_velocity(a,e=e)
    
    v_total = np.sqrt( (v_initial*1000)**2 
                   + 2*GM*(1-beta)*(1/(r*AU*1000)-1/(a*AU*1000)) )/1000
    
    v_azimuthal = v_initial * a / r
    
    v_radial = np.sqrt(v_total**2 - v_azimuthal**2)
    
    if azimuthal and radial:
        return v_total
    elif radial:
        return v_radial
    elif azimuthal:
        return v_azimuthal
    else:
        return np.zeros(len(v_total))


v_beta_velocity = np.vectorize(beta_velocity)




#%%

a = 0.05
r = np.linspace(0.05,1,100)
beta = 0.4
e = 0.3


v_tot = beta_velocity(a,r,e=e,beta=beta)
v_azm = beta_velocity(a,r,e=e,beta=beta,radial=False)
v_rad = beta_velocity(a,r,e=e,beta=beta,azimuthal=False)
fig,ax = plt.subplots()
ax.plot(r,v_tot,label="total")
ax.plot(r,v_azm,label="aziuthal")
ax.plot(r,v_rad,label="radial")
ax.set_ylim(0,1.05*np.max(v_tot))
ax.legend()
ax.set_xlabel("Heliocentric distance [AU]")
ax.set_ylabel("Dust speed [km/s]")
plt.show()



