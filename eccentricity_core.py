import numpy as np
from numba import jit

from conversions import GM
from conversions import AU

@jit
def azimuthal_flux(r_si,v_phi_si,
                   ex,mu,gamma):
    """
    The azimuthal component of the bound dust flux.

    Parameters
    ----------
    r : float
        SC heliocentric distance [m].
    v_phi : float
        SC heliocentric azimuthal speed [m/s].
    e : float
        Dust eccentricity.
    mu : float
        Effective gravitational parameter, 
        acocunting for beta value.
    gamma : float
        Bound dust radial spatial density exponent. 

    Returns
    -------
    j_tot : float
        The total azimuthal flux, stupid unit, see bound_flux.

    """
    def indefinite(x):
        a = (x**(2*gamma+2))/(2*gamma+2)
        b = v_phi_si*(x**(2*gamma+1))/(2*gamma+1)
        return a-b

    j_plus = ( indefinite(np.max(np.array([v_phi_si,
                                  (mu*(1+ex)/r_si)**0.5])))
              - indefinite(np.max(np.array([v_phi_si,
                                   (mu*(1-ex)/r_si)**0.5]))) )

    j_minus = ( indefinite(np.min(np.array([v_phi_si,
                                   (mu*(1+ex)/r_si)**0.5])))
               - indefinite(np.min(np.array([v_phi_si,
                                    (mu*(1-ex)/r_si)**0.5]))) )

    j_tot = j_plus - j_minus
    return j_tot

@jit
def radial_flux(r_si,v_r_si,
                ex,mu,gamma,
                size=100000):
    """
    The radial component of the bound dust flux.

    Parameters
    ----------
    r_si : float
        SC heliocentric distance [m].
    v_r_si : float
        SC heliocentric radial speed [m/s].
    e : float
        Dust eccentricity.
    mu : float
        Effective gravitational parameter, 
        acocunting for beta value.
    gamma : float
        Bound dust radial spatial density exponent. 
    size : int, optional
        The number of MC integration points.
        The default is 100000.

    Returns
    -------
    j_tot : float
        The total radial flux, stupid unit, see bound_flux.

    """
    lo = (mu*(1-ex)/r_si)**0.5
    hi = (mu*(1+ex)/r_si)**0.5

    V = hi - lo
    x = np.random.uniform(lo,hi,size)
    good_chunk = ( ((ex**2-1)*mu**2)
                   +(2*mu*(x**2)*r_si)
                   -((x**4)*(r_si**2))
                  )**0.5/(x*r_si)

    j_plus_pre = V * np.average(
         +1*(x**(2*gamma))*( - good_chunk - v_r_si )
         * ( ( - good_chunk - v_r_si)>0 ) )
    j_plus_post = V * np.average(
         +1*(x**(2*gamma))*( + good_chunk - v_r_si )
         * ( ( + good_chunk - v_r_si)>0 ) )
    j_minus_pre = V * np.average(
         -1*(x**(2*gamma))*( - good_chunk - v_r_si )
         * ( ( + good_chunk + v_r_si)>0 ) )
    j_minus_post = V * np.average(
         -1*(x**(2*gamma))*( + good_chunk - v_r_si )
         * ( ( - good_chunk + v_r_si)>0 ) )

    j_tot = 0.5 * (j_plus_pre + j_plus_post + j_minus_pre + j_minus_post)
    return j_tot

@jit
def bound_flux(r,v_r,v_phi,
               S_front,
               S_side,
               ex=1e-2,
               beta=0,
               gamma=-1.3,
               A=1):
    """
    The wrapper for the total bound flux observed, 
    given the dust parameters and the sc state.

    Parameters
    ----------
    r : float
        SC heliocentric distance [AU].
    v_r : float
        SC heliocentric radial speed [km/s].
    v_phi : float
        SC heliocentric azimuthal speed [km/s].
    S_front : float
        SC front-side cross section [m^2].
    S_side : float
        SC lateral cross section [m^2].
    e : float, optional
        Dust eccentricity. The default is 0.
    beta : float, optional
        Dust beta parameter. The default is 0.
    gamma : float, optional
        Bound dust radial spatial density exponent. 
        The default is -1.3.
    A : float, optional
        Multiplicative constant. 
        The default is 1.

    Returns
    -------
    total_flux : float
        The total bound dust flux, [s^-1].
        
    """

    r_si = r * AU #[m]
    v_r_si = v_r * 1000 #[m/s]
    v_phi_si = v_phi * 1000 #[m/s]

    mu = (1-beta)*GM
    prefactor = (((r_si**2)/(mu*(1+ex)))**gamma)

    total_flux = A / AU**gamma * prefactor * (
                  S_side * azimuthal_flux(r_si,v_phi_si,ex,mu,gamma)
                + S_front * radial_flux(r_si,v_r_si,ex,mu,gamma) )

    return total_flux

@jit
def bound_flux_vectorized(r_vector,v_r_vector,v_phi_vector,
                          S_front_vector,
                          S_side_vector,
                          ex=1e-2,
                          beta=0,
                          gamma=-1.3,
                          A=1):
    """
    A vectorizer for bound_flux function.

    Parameters
    ----------
    r : np.array of float
        SC heliocentric distance [AU].
    v_r : np.array of float
        SC heliocentric radial speed [km/s].
    v_phi : np.array of float
        SC heliocentric azimuthal speed [km/s].
    S_front : np.array of float
        SC front-side cross section [m^2].
    S_side : np.array of float
        SC lateral cross section [m^2].
    e : float, optional
        Dust eccentricity. The default is 0.
    beta : float, optional
        Dust beta parameter. The default is 0.
    gamma : float, optional
        Bound dust radial spatial density exponent. 
        The default is -1.3.
    A : float, optional
        Multiplicative constant. 
        The default is 1.

    Returns
    -------
    flux_vector : np.array of float
        The bound dust vector, as encountered along the ephemeris.
    
    """

    flux_vector = np.zeros(0)
    for r,v_r,v_phi,S_front,S_side in zip(r_vector,
                                          v_r_vector,
                                          v_phi_vector,
                                          S_front_vector,
                                          S_side_vector):
        flux_vector = np.append(flux_vector,
                                bound_flux(r,
                                           v_r,
                                           v_phi,
                                           S_front,
                                           S_side,
                                           ex,beta,gamma,A))
    return flux_vector





"""
i = 150

r = r_vector[i]
v_r = v_r_vector[i]
v_phi = v_phi_vector[i]
S_front = S_front_vector[i]
S_side = S_side_vector[i]

bound_flux(r,
           v_r,
           v_phi,
           S_front,
           S_side,
           e,beta,gamma,A)

"""








