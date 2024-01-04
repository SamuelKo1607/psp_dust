import os

l3_dust_location = os.path.join("..","data","psp","fields","l3_dust","")
exposure_location = os.path.join("..","data","psp","fields","exposure","")
psp_ephemeris_file = os.path.join("data_synced","psp_ephemeris_noheader.txt")
figures_location = os.path.join("998_generated","figures","")
all_obs_location = os.path.join("998_generated","observations","")
inla_results = os.path.join("998_generated","inla","")
readable_data = os.path.join("data_synced","psp_flux_readable.csv")

psp_model_location = os.path.join("data_synced","parkersolarprobe.stl")

if __name__ == "__main__":
    print("data paths in your system:")
    print(l3_dust_location)
    print("---------------------------------------")
    print("checking ephemeris files:")
    for file in [psp_ephemeris_file]:
        try:
            with open(file, 'r') as f:
                print(file+" OK in place")
        except:
            print(file+" not OK")