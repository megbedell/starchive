import numpy as np

def is_close(ra1, ra2, dec1, dec2, radius=1.0/120.0):
    # default radius: 30 arcsec (units degrees)
    delra = np.abs(ra1 - ra2)
    deldec = np.abs(dec1 - dec2)
    if (deldec <= radius and delra <= radius/np.cos(dec1 * np.pi/180.0)):
        return True
    return False

def abs_mag(rel_mag, plx):
    # takes relative magnitude and parallax (unit mas); returns absolute magnitude
    abs_mag = rel_mag + 5.0*(np.log10(plx/1000.)+1.)
    return abs_mag
