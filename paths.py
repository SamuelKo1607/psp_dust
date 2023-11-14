import os

l3_dust_location = os.path.join("..","data","psp","fields","l3_dust","")
psp_ephemeris_file = os.path.join("data_synced","psp_ephemeris_noheader.txt")

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