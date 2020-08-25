import numpy as np
import pyGM.pyGM_funcs as pgf
import scipy.interpolate as interp
import rasterio

"""k = 55555
x = 2 * np.random.rand(k, 1) - 1
y = 2 * np.random.rand(k, 1) - 1
d = np.sqrt(x**2 + y**2)

n = d[d < 1]
pi_approx = 4*len(n)/len(d)
print(pi_approx)
print(pi_approx - np.pi)
print((pi_approx-np.pi)/np.pi)"""

path = '../bayesian-thickness/data/pre-surge/'

void_dem = path + 'lk_dem_2007_clipped.tif'
filler_dem = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\data\kluane\Surface_DEM_Little_Kluane.tif'

#pgf.patch_raster(void_dem, filler_dem, path + 'lk_dem_2007_patched.tif', no_data_values=-9999)

a = [0.5, 0.2, 0.3]

print()