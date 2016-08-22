import numpy as np

def hiptohd(hip_number):
    '''
    Input an HIP identifier, get back the corresponding HD identifier.
    Result: HD number (integer)
    '''
    hipcat, hdcat = np.genfromtxt('data/hip_main.dat', delimiter='|', usecols=(1,71), dtype="i8", unpack=True)
    ind = np.where(hipcat == int(hip_number))[0]
    if ind.size == 0:
        print "HIP identifier does not exist!"
        return None
    return hdcat[ind][0]
    
def hdtohip(hd_number):
    '''
    Input an HD identifier, get back the corresponding HIP identifier.
    Result: HD number (integer)
    '''
    hipcat, hdcat = np.genfromtxt('data/hip_main.dat', delimiter='|', usecols=(1,71), dtype="i8", unpack=True)
    ind = np.where(hdcat == int(hd_number))[0]
    if ind.size == 0:
        print "HD identifier is not in the HIP catalog!"
        return None
    return hipcat[ind][0]