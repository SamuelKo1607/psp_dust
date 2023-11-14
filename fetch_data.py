import requests
import os
import datetime as dt
import cdflib
import numpy as np

from conversions import date2YYYYMMDD

from paths import l3_dust_location


def build_url(YYYYMMDD,
              version="v01"):
    """
    A function to prepare the URL where the data file is found. 
    The URL goes like:  https://research.ssl.berkeley.edu/data/psp/data/
                        sci/fields/l3/dust/2019/08/
                        psp_fld_l3_dust_20190806_v01.cdf

    Parameters
    ----------
    YYYYMMDD : str
        The date of interest.
    version : str, optional
        The suffix indicating the version. The default is "v01", which is 
        sufficient as of 11/2023.

    Returns
    -------
    url : str
        The URL to the datafile for the day.

    filename : str
        Just the filename.

    """
    YYYY = YYYYMMDD[:4]
    MM = YYYYMMDD[4:6]
    url = ("https://research.ssl.berkeley.edu/data/psp/data/sci/fields/l3/"
        + f"dust/{YYYY}/{MM}/psp_fld_l3_dust_{YYYYMMDD}_{version}.cdf")
    filename = url[url.find("psp_fld_l3_"):]
    return url, filename


def psp_dust_download(YYYYMMDD,
                      target_folder=l3_dust_location):
    """
    The function that downloads the .cdf L3 dust file for the requested date.

    Parameters
    ----------
    YYYYMMDD : str
        The date of interest.
    target_folder : str, optional
        The target folder for the download. 
        The default is l3_dust_location.

    Returns
    -------
    target : str
        The filepath to the downloaded file.

    Raises
    ------
    Exception
        In case the download failed. 

    """
    url, filename = build_url(YYYYMMDD)
    target = os.path.join(l3_dust_location,filename)

    a = dt.datetime.now()
    r = requests.get(url, allow_redirects=True)
    if not str(r)=="<Response [404]>":
        open(target, 'wb').write(r.content)
        print(str(round(os.path.getsize(target)/(1024**2),ndigits=2))
              +" MiB dowloaded in "+str(dt.datetime.now()-a))
    else:
        print(filename+" N/A; <Response [404]>")
        raise Exception(f"Download unseccsessful @ {YYYYMMDD}")
    return target


def psp_dust_fetch(YYYYMMDD,
                   target_folder=l3_dust_location):
    """
    The function to hand in a file for the requests date. If not present,
    then calls psp_dust_download. Either returns the target file that 
    can be reached or rasies an Exception through psp_dust_download.

    Parameters
    ----------
    YYYYMMDD : str
        The date of interest.
    target_folder : str, optional
        The target folder for the download. 
        The default is l3_dust_location.

    Returns
    -------
    target : str
        The filepath to the downloaded file that can be reached.

    """

    url, filename = build_url(YYYYMMDD)
    target = os.path.join(l3_dust_location,filename)
    try:
        f = open(target)
    except:
        target = psp_dust_download(YYYYMMDD,target_folder)
    else:
        f.close()
    return target


def psp_dust_load(YYYYMMDD,
                  target_folder=l3_dust_location):
    """
    A wrapped to load the correct psp dust file as a cdf file using cdflib.

    Parameters
    ----------
    YYYYMMDD : str
        The date of interest.
    target_folder : str, optional
        The target folder for the download. 
        The default is l3_dust_location.

    Returns
    -------
    cdf_file : cdflib.cdfread.CDF
        The cdf datafile of interest.

    """
    target = psp_dust_fetch(YYYYMMDD,l3_dust_location)
    cdf_file = cdflib.CDF(target)
    return cdf_file


def get_list_of_days(date_min = dt.date(2018,10,2),
                     date_max = dt.date(2023,12,31)):

    days = np.arange(date_min,
                     date_max,
                     step=dt.timedelta(days=1)
                     ).astype(dt.datetime)
    YYYYMMDDs = [date2YYYYMMDD(day) for day in days]
    return YYYYMMDDs


#%%
if __name__ == "__main__":
    for YYYYMMDD in get_list_of_days():
        try:
            psp_dust_fetch(YYYYMMDD)
        except:
            pass








