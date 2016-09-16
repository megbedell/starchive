import numpy as np

'''
usage demo:
conv = identifiers.Converter()
hip_number = 30037
hd_number = conv.hiptohd(hip_number)
print 'HIP{0} is HD{1}'.format(hip_number, hd_number)
'''

class Converter:
    def __init__(self):
        hipcat, hdcat = np.genfromtxt('data/hip_main.dat', delimiter='|', usecols=(1,71), dtype="i8", unpack=True)
        self.dict_hiptohd = dict(zip(hipcat, hdcat))
        self.dict_hdtohip = dict(zip(hdcat, hipcat))
        
    def hiptohd(self, hip_number):
        '''
        Input an HIP identifier, get back the corresponding HD identifier.
        Result: HD number (integer)
        '''
        hd_number = self.dict_hiptohd[hip_number]
        if hd_number < 0:
            print "HIP identifier does not exist!"
            return None
        else:
            return hd_number
            
    def hdtohip(self, hd_number):
        '''
        Input an HD identifier, get back the corresponding HIP identifier.
        Result: HIP number (integer)
        '''
        hip_number = self.dict_hdtohip[hd_number]
        if hip_number < 0:
            print "HD identifier is not in the HIP catalog!"
            return None
        else:
            return hip_number
