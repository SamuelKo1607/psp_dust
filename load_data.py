import os
import glob
import cdflib
import pickle

from fetch_data import psp_dust_load
from conversions import tt2000_to_date

from paths import l3_dust_location



class Observation:
    """
    Aggreaget results of the measurement day, plus SolO and RPW status.
    """
    def __init__(self,
                 date,
                 impact_times,
                 non_impact_times,
                 duty_hours,
                 sampling_rate,
                 heliocentric_distance,
                 spacecraft_speed,
                 heliocentric_radial_speed,
                 velocity_phase,
                 velocity_inclination):
        self.date = date
        self.YYYYMMDD = date.strftime('%Y%m%d')
        self.impact_count = len(impact_times)
        self.impact_times = impact_times
        self.non_impact_times = non_impact_times
        self.duty_hours = duty_hours
        self.sampling_rate = sampling_rate
        self.heliocentric_distance = heliocentric_distance
        self.spacecraft_speed = spacecraft_speed
        self.heliocentric_radial_speed = heliocentric_radial_speed
        self.heliocentric_tangential_speed = ( spacecraft_speed**2
                                               - heliocentric_radial_speed**2
                                             )**0.5
        self.velocity_phase = velocity_phase
        self.velocity_inclination = velocity_inclination
        self.velocity_HAE_x = ( spacecraft_speed
                                * np.sin(np.deg2rad(90-velocity_inclination))
                                * np.cos(np.deg2rad(velocity_phase)) )
        self.velocity_HAE_y = ( spacecraft_speed
                                * np.sin(np.deg2rad(90-velocity_inclination))
                                * np.sin(np.deg2rad(velocity_phase)) )
        self.velocity_HAE_z = ( spacecraft_speed
                                * np.cos(np.deg2rad(90-velocity_inclination)) )
        self.produced = dt.datetime.now()


    def info(self):
        print(self.YYYYMMDD+
              " \n impact count: "+ str(self.impact_count)+
              " \n duty hours: " + str(self.duty_hours))


def save_list(data,
              name,
              location=""):
    """
    A simple function to save a given list to a specific location using pickle.

    Parameters
    ----------
    data : list
        The data to be saved. In our context: mostly a list of objects.
    
    name : str
        The name of the file to be written. May be an absolute path, 
        if the location is not used.
        
    location : str, optional
        The relative path to the data folder. May not be used if the name
        contains the folder as well. Default is therefore empty.

    Returns
    -------
    none
    """

    location = os.path.join(os.path.normpath( location ), '')
    os.makedirs(os.path.dirname(location+name), exist_ok=True)
    with open(location+name, "wb") as f:  
        pickle.dump(data, f)
        
        
def load_list(name,
              location):
    """
    A simple function to load a saved list from a specific location 
    using pickle.

    Parameters
    ----------    
    name : str
        The name of the file to load. 
        
    location : str, optional
        The relative path to the data folder.

    Returns
    -------
    data : list
        The data to be loaded. In our context: mostly a list of objects.
    """

    if location != None:
        location = os.path.join(os.path.normpath( location ), '')
        with open(location+name, "rb") as f:
            data = pickle.load(f)
        return data
    else:
        with open(name, "rb") as f:
            data = pickle.load(f)
        return data


def list_cdf(location=l3_dust_location):
    """
    The function to list all the l3 dust datafiles.

    Parameters
    ----------
    location : str, optional
        The data directory. Default is paths.l3_dust_location.

    Returns
    -------
    days : list of str 
        The available datafiles.
    """

    files = glob.glob(os.path.join(l3_dust_location,"psp_fld_*.cdf"))
    return files


def build_obs_from_cdf(cdf_file):


    epochs = cdf_file.varget("psp_fld_l3_dust_V2_rate_epoch")
    YYYYMMDD = [str(cdf_file.cdf_info().CDF)[-16:-8]]*len(epochs)
    dates = tt2000_to_date(epochs)
    encounter = cdf_file.varget("psp_fld_l3_dust_V2_rate_encounter")
    rate_corrcted = cdf_file.varget("psp_fld_l3_dust_V2_rate_ucc")

    observation = Observation()

    return observation


def main():

    for file in list_cdf(l3_dust_location):
        cdf_file = cdflib.CDF(file)
        short_name = str(cdf_file.cdf_info().CDF)[
                             str(cdf_file.cdf_info().CDF).find("psp_fld_l3_")
                             :-4]
        try:
            observation = build_obs_from_cdf(cdf_file)
        except Exception as err:
            print(f"{short_name}{err}")
        else:
            save_list(observation,
                      short_name+"_obs.pkl",
                      os.path.join("998_generated","observations",""))
    return


#%%
if __name__ == "__main__":
    main()

