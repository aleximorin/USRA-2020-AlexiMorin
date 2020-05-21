import numpy as np
import pandas as pd
import gdal

"""transform, width, height = rasterio.warp.calculate_default_transform(dem_ras.crs, projection,
                                                                                dem_ras.width, dem_ras.height,
                                                                                *dem_ras.bounds)
           data = dem_ras.read()
           z = np.zeros_like(data)
           for i, band in enumerate(data, 1):
               rasterio.warp.reproject(band, z[i], src_transform=dem_ras.transform,
                                       src_crs=dem_ras.crs, dst_transform=transform,
                                       dst_crs=projection, resamplig=rasterio.warp.Resampling.nearest)

           mask = rasterio.features.rasterize([geometry, 0], data.shape, transform=transform,
                                              fill=1, all_touched=True, dtype=np.uint8)
           masked_data = np.ma.array(data=z, mask=mask.astype(bool))

           dem_im, dem_out = rasterio.mask.mask(dem_ras, geometry, crop=True)
           meta = dem_ras.meta
           meta.update({'driver': 'GTiff',
                        'height': dem_im.shape[1],
                        'width': dem_im.shape[2],
                        'transform': dem_out})"""

def ddmm_to_decimal(latitude, longitude):
    # Converts latitude as ddmm.mmmm and longitude as dddmm.mmmm to decimal degrees

    longitude = np.abs(longitude)

    lat_degree = (np.floor(latitude / 100)).astype(int)
    long_degree = (np.floor(longitude) / 100).astype(int)

    lat_mm_mmm = latitude % 100
    long_mm_mmm = longitude % 100

    converted_lat = lat_degree + lat_mm_mmm/60
    converted_long = long_degree + long_mm_mmm/60

    return converted_lat, -converted_long

def edit_csv_long_lat(csv_path, csv_out, lat_field, long_field, delimiter = ','):
    file = pd.read_csv(csv_path, delimiter=delimiter)
    file[lat_field], file[long_field] = ddmm_to_decimal(file[lat_field], file[long_field])
    file.to_csv(csv_out, index=False)

def configures_itmix_txt_to_csv(txt_file):  #Obsolete
    with open(txt_file, 'r') as txt:
        lines = txt.readlines()
        proj_string = lines[2]
        proj = proj_string[proj_string.find('"') + 1:proj_string.rfind('"')]
    csv = pd.read_csv(txt_file, skiprows=13, delim_whitespace=True)
    csv.to_csv(txt_file.replace('.txt', '.csv'), index=False)

def interpolates_csv_to_raster(path, csv, projection, extension = '.csv'):  # Is cool but not what i want
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
lat = 'Latitude_ddmm.mmmm'
long = 'Longitude_dddmm.mmmm'

path = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\data\kluane\surging_glacier\2019_radar\lk_radar_points.csv'
edit_csv_long_lat(path, path.replace('.csv', '_decimal.csv'), lat, long, ';')
