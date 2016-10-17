import numpy as np
import csv
import matplotlib.pyplot as plt
from math import copysign

class Catalog:
    def __init__(self):
        self.name = []
        self.ra = []
        self.dec = []
        self.n_exp = []
        self.snr = []
    
    def add_obj(self, name, ra, dec, snr):
        # create a new object
        self.name.append(name)
        self.ra.append(ra)
        self.dec.append(dec)
        self.n_exp.append(1)
        self.snr.append(snr)
        
    def add_obs(self, ind, snr):
        # add another observation to the existing object with index ind
        self.n_exp[ind] += 1
        self.snr[ind] = np.sqrt(self.snr[ind]**2 + snr**2)
        
    def add_param_col(self):
        # AFTER catalog has been populated, get ready to add in stellar parameters
        self.teff = np.zeros_like(self.ra)
        #self.teff_err = np.zeros_like(self.ra)
        self.logg = np.zeros_like(self.ra)
        #self.logg_err = np.zeros_like(self.ra)
        self.feh = np.zeros_like(self.ra)
        #self.feh_err = np.zeros_like(self.ra)
        
    def add_param(self, ind, param):
        # add stellar parameters (dict format assumed) to the existing object with index ind_self
        #for attr in ['teff','logg','feh']:
        #    saved = getattr(self, attr)
        #    saved[ind_self]
        #    setattr(self, 'teff')
        self.teff[ind] = param['teff']
        self.logg[ind] = param['logg']
        self.feh[ind] = param['feh']

def is_close(ra1, ra2, dec1, dec2, radius=1.0/120.0):
    # default radius: 30 arcsec (units degrees)
    delra = np.abs(ra1 - ra2)
    deldec = np.abs(dec1 - dec2)
    if (deldec <= radius and delra <= radius/np.cos(dec1 * np.pi/180.0)):
        return True
    return False

if __name__ == "__main__":

    #lamost = np.loadtxt('/Users/mbedell/Documents/Research/Stars/LAMOST/dr2_stellar.csv', delimiter='|', skiprows=1, \
    #            usecols=(1,34,35,36,37,38,39,40,41), \
    #            dtype={'names': ('Designation', 'RA', 'Dec', 'teff', 'teff_err', 'logg', 'logg_err', 'feh', 'feh_err'), \
    #            'formats': ('S30', 'f16', 'f16', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')})
                
    spocs_param = np.genfromtxt('/Users/mbedell/Documents/Research/Stars/SPOCS/table8.dat', delimiter=',', usecols=(0,2,3,4), \
             dtype={'names': ('ID', 'teff', 'logg', 'feh'), 'formats': ('<i8', '<f8', '<f8', '<f8')}, skip_header=1)
    spocs_pos = np.genfromtxt('/Users/mbedell/Documents/Research/Stars/SPOCS/table9.dat', delimiter=',', usecols=(0,2,3,4,5,6,7), \
             dtype={'names': ('ID', 'RAh', 'RAm', 'RAs', 'DEd', 'DEm', 'DEs'), 'formats': ('<i8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8')}, skip_header=1)
    spocs_RA = (spocs_pos['RAh']+spocs_pos['RAm']/60.0+spocs_pos['RAs']/3600.0)/24.0*360.0 
    Dec_sign = [copysign(1.0,i) for i in spocs_pos['DEd']]  # this works with -0h objects too
    spocs_Dec = Dec_sign * (np.abs(spocs_pos['DEd'])+spocs_pos['DEm']/60.0+spocs_pos['DEs']/3600.0)
    
    sousa = np.genfromtxt('/Users/mbedell/Documents/Research/Stars/HARPS_GTO/Sousa2008.tsv', delimiter='|', skip_header=72, \
                usecols=(0,1,3,5,7), dtype={'names': ('RA', 'Dec', 'teff', 'logg', 'feh'), 'formats': ('<f8', '<f8', '<f8', '<f8', '<f8')})

    ambre = np.genfromtxt('/Users/mbedell/Documents/Research/Stars/AMBRE/ambre_feros.csv', delimiter=',', skip_header=1, \
                usecols=(3,4,17,20,23), dtype={'names': ('RA', 'Dec', 'teff', 'logg', 'feh'), 'formats': ('<f8', '<f8', '<f8', '<f8', '<f8')})
    
    gaia = np.genfromtxt('/Users/mbedell/Documents/Research/Stars/Gaia/gaia_eso.csv', delimiter=',', skip_header=1, \
                usecols=(5,6,9,11,13), dtype={'names': ('RA', 'Dec', 'teff', 'logg', 'feh'), 'formats': ('<f8', '<f8', '<f8', '<f8', '<f8')})
        

    # READ IN THE HARPS CATALOG:
    HARPScat = Catalog()
    HARPScat.name = np.genfromtxt('data/HARPS_all.csv', delimiter=',', skip_header=1, usecols=(0), dtype='|S30')
    HARPScat.ra, HARPScat.dec, HARPScat.n_exp, HARPScat.snr = np.genfromtxt('data/HARPS_all.csv', \
                delimiter=',', skip_header=1, usecols=(1,2,3,4), unpack=True) # workaround bc dtype + unpack has a bug in np.genfromtxt

    # CROSS-CHECK WITH SPECTRAL PARAMETERS:
    HARPScat.add_param_col()
    gaia_count = 0
    ambre_count = 0
    spocs_count = 0
    sousa_count = 0
    for i,(ra,dec) in enumerate(zip(HARPScat.ra, HARPScat.dec)):
        # find Gaia-ESO match:
        gaia_ind = (np.sqrt(((ra - gaia['RA'])/np.cos(dec * np.pi/180.0))**2 + (dec - gaia['Dec'])**2)).argmin()
        if is_close(ra, gaia['RA'][gaia_ind], dec, gaia['Dec'][gaia_ind]):
            HARPScat.add_param(i, gaia[gaia_ind]) 
            gaia_count += 1
        # find AMBRE match:
        ambre_ind = (np.sqrt(((ra - ambre['RA'])/np.cos(dec * np.pi/180.0))**2 + (dec - ambre['Dec'])**2)).argmin()
        if is_close(ra, ambre['RA'][ambre_ind], dec, ambre['Dec'][ambre_ind]):
            HARPScat.add_param(i, ambre[ambre_ind])
            ambre_count += 1
        # find SPOCS match:
        spocs_ind = (np.sqrt(((ra - spocs_RA)/np.cos(dec * np.pi/180.0))**2 + (dec - spocs_Dec)**2)).argmin()
        if is_close(ra, spocs_RA[spocs_ind], dec, spocs_Dec[spocs_ind]):
            HARPScat.add_param(i, spocs_param[spocs_ind])
            spocs_count += 1
        # find Sousa match:
        sousa_ind = (np.sqrt(((ra - sousa['RA'])/np.cos(dec * np.pi/180.0))**2 + (dec - sousa['Dec'])**2)).argmin()
        if is_close(ra, sousa['RA'][sousa_ind], dec, sousa['Dec'][sousa_ind]):
            HARPScat.add_param(i, sousa[sousa_ind])
            sousa_count += 1
    
    print "+ Gaia-ESO: {0} objects found".format(gaia_count)   
    print "+ AMBRE: {0} objects found".format(ambre_count)   
    print "+ SPOCS: {0} objects found".format(spocs_count)       
    print "+ Sousa: {0} objects found".format(sousa_count) 
    
    print "net unique objects found: {0}".format(np.sum(HARPScat.logg > 0.0))      
            
    # write out the catalog with parameters:
    save_cat =  np.transpose(np.asarray([HARPScat.name, HARPScat.ra, HARPScat.dec, HARPScat.n_exp, HARPScat.snr, \
            HARPScat.teff, HARPScat.logg, HARPScat.feh]))
    np.savetxt('HARPScat_param.csv', save_cat, \
            delimiter=',', fmt='%s', header='Name, RA, Dec, N_exp, SNR, Teff, logg, [M/H]')

    # make an H-R diagram:
    t_data = np.float64(HARPScat.teff[HARPScat.logg > 0.0])
    g_data = np.float64(HARPScat.logg[HARPScat.logg > 0.0])
    plt.hist2d(t_data, g_data, bins=[np.arange(4600, 6700, 100),np.arange(3.2,5.2, 0.1)], cmap=plt.get_cmap('BuGn'))
    plt.xlim([6600,4600])
    plt.ylim([5.1,3.2])
    cbar = plt.colorbar()
    cbar.set_label('# of stars', rotation=90)
    plt.ylabel('log(g)')
    plt.xlabel(r'T$_{eff}$')
    plt.savefig('harps_hr.png')
        