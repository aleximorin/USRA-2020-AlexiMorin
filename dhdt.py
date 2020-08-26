import rasterio
from rasterio.mask import geometry_mask
from pyGM.pyGM_funcs import *
import geopandas as gpd
import matplotlib.pyplot as plt


def compute_index(dh, z, meta):

    dA = -1 * meta['transform'][0] * meta['transform'][4]
    rho = 950  # Whatever number it should be
    g = 9.81
    im = dh*z
    index = np.mean(im[~np.isnan(im)])
    return im


if __name__ == '__main__':

    r"""arctic_dem_path = r'..\data\kluane\arcticdem_lk.tif'
    surface_dem_path = r'../data/kluane/Surface_DEM_Little_Kluane.tif'
    
    patch_raster(arctic_dem_path, surface_dem_path, output_ras=None, no_data_values=0)"""

    # Define path parameters
    dh_path = '../data/kluane/dh_2018-2007_within150.tif'  # dh/dt data
    outline_path = '../data/kluane/glacier_outlines/little_kluane.shp'  # the outline of the lk glacier
    dem_path = '../data/kluane/arcticDEM_43_08_merge_UTM7N.tif'  # the big big surface dem
    cropped_dem_path = '../data/kluane/arcticdem_lk_filled.tif'  # the smaller surface dem

    # Opens the various data

    # the smaller area surface dem
    with rasterio.open(cropped_dem_path, 'r') as cropped_dem_ras:
        cropped_dem_im = cropped_dem_ras.read()
        cropped_dem_meta = cropped_dem_ras.meta.copy()
        cropped_dem_im[cropped_dem_im <= 0] = np.nan

    # dh/dt, also crops it to the
    with rasterio.open(dh_path, 'r') as dh_ras:
        dh_im = dh_ras.read()
        dh_meta = dh_ras.meta.copy()
        cropped_dh_im, cropped_dh_meta = resize_ras_to_target(dh_ras, cropped_dem_meta)

    # the glacier shapefile
    outline = gpd.read_file(outline_path)

    # Smaller, lk only dataset and filled
    cropped_mask = geometry_mask(outline.geometry, cropped_dem_im[0].shape, cropped_dem_meta['transform'])
    cropped_dem_im[:, cropped_mask] = np.nan
    lk_index_im = compute_index(cropped_dh_im, cropped_dem_im, cropped_dem_meta)
    lk_index = np.nanmean(lk_index_im)

    # the bigger surface dem, crops it and makes it the same extent as of the dh/dt data raster
    with rasterio.open(dem_path, 'r') as dem_ras:
        dem_im = dem_ras.read()
        resized_dem, resized_meta = resize_ras_to_target(dem_ras, dh_meta)

    # Only averaging dh/dt data
    mask = geometry_mask(outline.geometry, dh_im[0].shape, dh_meta['transform'])
    total_mu_dh = np.nanmean(dh_im)
    partial_mu_dh = np.nanmean(dh_im[:, mask])
    lk_mu_dh = np.nanmean(dh_im[:, ~mask])

    # Computing total indices for dh/dt data
    total_index_im = compute_index(dh_im, resized_dem, dh_meta)
    total_index = np.nanmean(total_index_im)
    partial_index = np.nanmean(total_index_im[:, mask])
    lk_var_index = np.nanmean(total_index_im[:, ~mask])

