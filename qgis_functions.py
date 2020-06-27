import numpy as np
import pandas as pd
import gdal
import geopandas as gpd
import shapely.geometry as geom
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio import features, Affine
from rasterio.mask import mask, geometry_mask
import skimage.transform as st
import matplotlib.pyplot as plt
import os

# Plenty of more or less documented functions, from useful to not really.
# Most of the functions actually used are directly imported in the compare_model_v2.py file


def df_dict_to_excel(df_dict, path, header=True, index=True):
    # Outputs a dictionary of DataFrame to an excel file with the DataFrame's key as the sheet name
    # https://stackoverflow.com/questions/51696940/how-to-write-a-dict-of-dataframes-to-one-excel-file-in-pandas-key-is-sheet-name
    writer = pd.ExcelWriter(path, engine='xlsxwriter')

    for tab_name, df in df_dict.items():
        df.to_excel(writer, sheet_name=tab_name, header=header, index=index)

    saved = False
    extension = path[path.rfind('.'):]
    xl_file = path[:-len(extension)]
    i = 1
    while not saved:
        try:
            writer.save()
            saved = True
        except PermissionError:
            print(f'{path} not accesible.')
            print(f'Saving at {xl_file}')


def ddmm_to_decimal(latitude, longitude):
    # Converts latitude as ddmm.mmmm and longitude as dddmm.mmmm to decimal degrees

    longitude = np.abs(longitude)

    lat_degree = (np.floor(latitude / 100)).astype(int)
    long_degree = (np.floor(longitude) / 100).astype(int)

    lat_mm_mmm = latitude % 100
    long_mm_mmm = longitude % 100

    converted_lat = lat_degree + lat_mm_mmm / 60
    converted_long = long_degree + long_mm_mmm / 60

    return converted_lat, -converted_long


def edit_csv_long_lat(csv_path, csv_out, lat_field, long_field, delimiter=','):
    file = pd.read_csv(csv_path, delimiter=delimiter)
    file[lat_field], file[long_field] = ddmm_to_decimal(file[lat_field], file[long_field])
    file.to_csv(csv_out, index=False)
    return file


def configures_itmix_txt_to_csv(txt_file):  # Obsolete
    with open(txt_file, 'r') as txt:
        lines = txt.readlines()
        proj_string = lines[2]
        proj = proj_string[proj_string.find('"') + 1:proj_string.rfind('"')]
    csv = pd.read_csv(txt_file, skiprows=13, delim_whitespace=True)
    csv.to_csv(txt_file.replace('.txt', '.csv'), index=False)


def interpolates_csv_to_raster(path, csv, projection, extension='.csv'):  # Is cool but not what i want
    #  With respect to this code: https://stackoverflow.com/questions/53854688/csv-to-raster-python
    csv_path = path + '\\' + csv
    vrt_fn = csv_path.replace(extension, '.vrt')
    layer = csv.replace(extension, '')
    out_tif = csv_path.replace(extension, '.tif')

    with open(csv_path, 'r') as csv:
        x, y, z = csv.readline()[:-1].split(',')  # [:-1]'s only purpose is to remove the \n

    with open(vrt_fn, 'w') as fn_vrt:
        fn_vrt.write('<OGRVRTDataSource>\n')
        fn_vrt.write(f'\t<OGRVRTLayer name="{layer[1:]}">\n')
        fn_vrt.write(f'\t\t<SrcDataSource>CSV:{csv_path}</SrcDataSource>\n')
        fn_vrt.write('\t\t<GeometryType>wkbPoint</GeometryType>\n')
        fn_vrt.write(f'\t\t<GeometryField encoding="PointFromColumns" x="{x}" y="{y}" z="{z}"/>\n')
        fn_vrt.write('\t</OGRVRTLayer>\n')
        fn_vrt.write('</OGRVRTDataSource>\n')

    output = gdal.Grid(out_tif, vrt_fn)
    return output


def resize_ras_to_target(orig_ras, meta, output_ras):

    # Gets the data from the target raster and
    # creates a box from it's extent to resize the original raster to
    out_z, out_h, out_w = meta['count'], meta['height'], meta['width']
    extent = rasterio.transform.array_bounds(out_h, out_w, meta['transform'])
    bbox = geom.box(extent[0], extent[1], extent[2], extent[3])
    gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=meta['crs'])

    # Crops the original raster
    cropped, t = mask(orig_ras, shapes=gdf.geometry, crop=True, nodata=np.NaN)
    cropped = cropped.astype(meta['dtype'])

    out_im = np.zeros(shape=(out_z, out_h, out_w)).astype(meta['dtype'])
    out_im[out_im == 0] = np.NaN
    out_im, t = reproject(cropped, destination=out_im, src_transform=t, dst_transform=meta['transform'],
                          src_crs=orig_ras.crs, dst_crs=orig_ras.crs)

    if output_ras.endswith('.tif'):
        meta['driver'] = 'GTiff'

    # Writes the new raster with the target raster's metadata
    out = rasterio.open(output_ras, 'w+', **meta)
    out.write(out_im)
    return out


def crop_raster_to_geometry(raster, geometry):
    #im, t = mask(raster, geometry, crop=True, nodata=np.NaN, all_touched=False)
    #im[im <= -9999] = np.NaN
    im = raster.read()
    mask = geometry_mask(geometry, im[0].shape, raster.transform, all_touched=False)
    im[:, mask] = np.NaN
    return im


def crop_image_to_geometry(im, metadata, geometry):

    h, w = im.shape[-2], im.shape[-1]
    mask = geometry_mask(geometry, (h, w), metadata['transform'])
    im[:, mask] = np.NaN
    return im


def points_to_raster(points_shp, z, meta):

    # Gets the x and y values of the points
    x, y = points_shp.geometry.x, points_shp.geometry.y

    # Gets the important information from the metadata and gives them an actual meaning
    c, w, h = meta['count'], meta['width'], meta['height']
    dx, dy = meta['transform'][0], meta['transform'][4]
    x0, y0, = meta['transform'][2], meta['transform'][5]

    # Creates the bins from the DEM's extent
    xs, ys = np.arange(x0, x0 + (w+1)*dx, dx), np.flip(np.arange(y0, y0 + (h+1)*dy, dy))

    # Number of points per cell
    n, _, _ = np.histogram2d(y, x, bins=(ys, xs))
    n[n == 0] = np.NaN

    # Sum of values per cell and compute the mean value
    im, y_edges, x_edges = np.histogram2d(y, x, bins=(ys, xs), weights=z, normed=False)
    im /= n

    # Flip the array so it fits with the map's format
    im = np.flip(im.reshape((c, h, w)), axis=1)

    # Computes the new raster's extents
    y_index = (~np.isnan(im)).sum(2) != 0
    x_index = (~np.isnan(im)).sum(1) != 0

    xmin = x_edges[x_index.argmax()]
    ymin = y_edges[np.flip(y_index).argmax()]

    xmax = x_edges[-np.flip(x_index).argmax()-1]
    ymax = y_edges[-y_index.argmax()-1]

    bounds = np.array([xmin, ymin, xmax, ymax])

    return im, bounds


if __name__ == '__main__':
    r""" lat = 'Latitude_ddmm.mmmm'
    long = 'Longitude_dddmm.mmmm'

    path = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\data\job\Meager_radar_picked_9Nov2018.csv'
    out_path = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\data\job\gpr_data.csv'
    df = edit_csv_long_lat(path, path.replace('.csv', '_decimal.csv'), lat, long, ';')
    gdf = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=[Point(xy) for xy in zip(df[long], df[lat])])
    gdf = gdf.to_crs('epsg:32610')
    df = pd.DataFrame()
    df['x'] = gdf.geometry.x
    df['y'] = gdf.geometry.y
    df['h'] = gdf['ice_thickness_m']
    df.to_csv(out_path, index=False)"""

    r"""path = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\data\kluane' + r'\north_glacier_gpr_points.shp'
    gdf = gpd.read_file(path)
    df = pd.DataFrame()
    df['x'] = gdf.geometry.x
    df['y'] = gdf.geometry.y
    df['h'] = gdf['field_3']
    df.to_csv(r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\data\kluane\specific_points_north_glacier.csv',
              index=False, header=False)
"""
    r"""    dem = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\data\job\surface_DEM_RGI60-02.01654.tif'
    orig = 'C:/Users/Adalia Rose/Desktop/2scool4cool/e2020/data/job/Lidar/meager_full_bedem1.tif'"""

    r"""ori = '../data/Thickness_Alaska/RGI60-01.16835_thickness.tif'
    tar = '../data/itmix/01_ITMIX_input_data/NorthGlacier/02_surface_NorthGlacier_2007_UTM07.asc'
    shp = '../data/itmix/01_ITMIX_input_data/NorthGlacier/shapefiles/01_margin_NorthGlacier_2007_UTM07.shp'
    shp = gpd.read_file(shp)
    model = rasterio.open(ori)
    dem = rasterio.open(tar, 'r+')
    meta = dem.meta.copy()"""
   # im = crop_raster_to_geometry(dem, shp.geometry)
   # out = resize_ras_to_target(model, meta, 'resized_model.tif')

    r"""dem = rasterio.open('../data/job/resized_lidar_job.tif', 'r')
    meta = dem.meta.copy()
    df = pd.read_csv('../data/job/gpr_data.csv')
    points = gpd.GeoDataFrame(df, crs=meta['crs'], geometry=[geom.Point(xy) for xy in zip(df.iloc[:, 0], df.iloc[:, 1])])
    im, bounds = points_to_raster(points, points.iloc[:, 2], meta)
    extent = rasterio.transform.array_bounds(dem.height, dem.width, dem.transform)
    extent = [extent[i] for i in [0, 2, 1, 3]]
    plt.imshow(im[0], extent=extent)
    plt.scatter(points.geometry.x, points.geometry.y, c=points.iloc[:, 2])
    c_xmin, c_ymin, c_xmax, c_ymax = bounds
    plt.xlim(c_xmin, c_xmax)
    plt.ylim(c_ymin, c_ymax)
    plt.show()"""