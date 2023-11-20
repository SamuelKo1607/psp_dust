import os
import glob
import cdflib
import pickle
import datetime as dt

from fetch_data import psp_dust_load
from conversions import tt2000_to_date

from paths import l3_dust_location



class Observation:
    """
    Aggregate results of the measurement period.
    """
    def __init__(self,
                 date,
                 epoch_center,
                 epochs_on_day,
                 encounter,
                 rate_corrected,
                 inbound,
                 ej2000):
        self.date = date
        self.YYYYMMDD = date.strftime('%Y%m%d')
        self.epoch_center = epoch_center
        self.epochs_on_day = epochs_on_day
        self.encounter = encounter
        self.rate_corrected = rate_corrected
        self.inbound = inbound
        self.ej2000 = ej2000
        self.produced = dt.datetime.now()


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
    """
    A function to build a list of observations extracted from one cdf file.

    Parameters
    ----------
    cdf_file : cdflib.cdfread.CDF
        The PSP L3 dust cdf file to extract data from. 

    Returns
    -------
    observations : list of Observation object
        One entry for each observation, typically once per 8 hours.

    """

    epochs = cdf_file.varget("psp_fld_l3_dust_V2_rate_epoch")
    YYYYMMDD = [str(cdf_file.cdf_info().CDF)[-16:-8]]*len(epochs)
    dates = tt2000_to_date(epochs)
    encounter = cdf_file.varget("psp_fld_l3_dust_V2_rate_encounter")
    rate_corrected = cdf_file.varget("psp_fld_l3_dust_V2_rate_ucc")
    inbound = cdf_file.varget("psp_fld_l3_dust_V2_rate_inoutbound")
    ej2000_x = cdf_file.varget("psp_fld_l3_dust_V2_rate_ej2000_x")
    ej2000_y = cdf_file.varget("psp_fld_l3_dust_V2_rate_ej2000_y")
    ej2000_z = cdf_file.varget("psp_fld_l3_dust_V2_rate_ej2000_z")

    observations = []
    for i,epoch in enumerate(epochs):
        observations.append(Observation(dates[i],
                                        epoch,
                                        len(epochs),
                                        encounter[i],
                                        rate_corrected[i],
                                        inbound[i],
                                        [ej2000_x[i],
                                         ej2000_y[i],
                                         ej2000_z[i]]
                                        ))

    return observations


def main(dust_location=l3_dust_location,
         target_directory=os.path.join("998_generated","observations",""),
         save=True):
    """
    A function to aggregate all the observations as per PSP L3 dust. A folder
    is browsed nad a list of Observation type is created and optionally saved.

    Parameters
    ----------
    dust_location : str, optional
        The path to the data. The default is l3_dust_location.
    target_directory : str, optional
        The path where to put the final list. 
        The default is os.path.join("998_generated","observations","").
    save : bool, optional
        Whether to save the data. The default is True.

    Returns
    -------
    observation : list of Observation
        The agregated data.

    """
    observations = []
    for file in list_cdf(dust_location):
        cdf_file = cdflib.CDF(file)
        short_name = str(cdf_file.cdf_info().CDF)[
                             str(cdf_file.cdf_info().CDF).find("psp_fld_l3_")
                             :-4]
        try:
            observation = build_obs_from_cdf(cdf_file)
        except Exception as err:
            print(f"{short_name}{err}")
        else:
            observations.extend(observation)



    if save:
        save_list(observations,
                  "all_obs.pkl",
                  target_directory)

    return observations


#%%
if __name__ == "__main__":
    main()

