import os
from comparing_models import *
import pandas as pd
import rasterio
import numpy as np
import traceback
import fiona
from shapely.geometry import shape, Polygon, mapping

if __name__ == '__main__':
    # xs, ys = compute_every_hypsometry()
    # csv_path = compare_every_curve()
    # csv_ez_read('hypsometry_compared_r2.csv')

    # Columns for the error DataFrame
    cols = 'Mean error (%), Std (%), Median error (%), ' \
           'Maximum error (%), Minimum error (%), Mean absolute error (%)'.split(', ')

    # Data needed for the error DataFrame
    results_path = 'hypsometry_compared_r2.csv'
    results = pd.read_csv(results_path, delimiter=',', index_col=0)
    error_dataFrame = {}  # Creates a dictionary of DataFrames

    # Initializing the paths
    itmix_path = r'..\data\itmix'
    shp_path = itmix_path + '\\01_ITMIX_input_data\\'
    DEM_path = itmix_path + '\\01_ITMIX_input_data\\'
    bedrock_path = itmix_path + '\\02_ITMIX_results'
    true_bed_path = itmix_path + '\\03_ITMIX_measured_thickness'
    true_bed_raster_path = 'true_bed_tifs'
    error_raster_path = 'error_tifs'
    img_folder = 'comparison_images'

    # Some test cases
    """glacier = 'Brewster'
    burns_csv_to_raster(f'{true_bed_path}\\03_RES_{glacier}.txt',
                        f'{shp_path}\\{glacier}\\shapefiles\\01_margin_{glacier}_1997_UTM-59.shp',
                        f'{DEM_path}\\{glacier}\\02_surface_{glacier}_1997_UTM-59.asc', output_fn=None, crop=True)"""

    glacier = 'Columbia'
    model = 'Farinotti'
    model_path = bedrock_path + f'\\{model}_{glacier}_bedrock.asc'
    dem_ras = rasterio.open(f'{DEM_path}\\{glacier}\\02_surface_{glacier}_2007_UTM06.asc')
    #burns_csv_to_raster(f'{true_bed_path}\\03_RES_{glacier}.txt',
    #                    f'{shp_path}\\{glacier}\\shapefiles\\01_margin_{glacier}_2007_UTM06.shp',
    #                   f'{DEM_path}\\{glacier}\\02_surface_{glacier}_2007_UTM06.asc', output_fn=None, crop=True)
    #compare_glacier_to_model(glacier, model, f'{shp_path}\\{glacier}\\shapefiles\\01_margin_{glacier}_2007_UTM06.shp', model_path,
    #                                             rasterio.open(f'dem_tifs\\{glacier}_true_bed_elev.tif'), dem_ras, img_folder)


    # If cropping is wanted
    crop = True

    # Creates the various folders needed
    try:
        os.makedirs(true_bed_raster_path)
    except Exception as e:
        pass
    try:
        os.makedirs(error_raster_path)
    except Exception as e:
        pass
    try:
        os.makedirs(img_folder)
    except Exception as e:
        pass

    #for glacier in results.index:
    for glacier in ['Washmawapta']:

        print(f'\nProcessing {glacier}\n')

        # Resets the path names
        shp_path = itmix_path + '\\01_ITMIX_input_data\\'
        DEM_path = itmix_path + '\\01_ITMIX_input_data\\'
        bedrock_path = itmix_path + '\\02_ITMIX_results'
        true_bed_path = itmix_path + '\\03_ITMIX_measured_thickness'

        # Finds the corresponding models of each glaciers
        models = [i.split('_')[0] for i in os.listdir(bedrock_path) if i.find(glacier) > 0]

        # Each DataFrame consists of the various error data as columns and the models as index
        error_dataFrame[glacier] = pd.DataFrame(columns=cols, index=models)

        # Finds the corresponding DEM path
        DEM_path += f'\\{glacier}\\' + [file for file in os.listdir(f'{shp_path}\\{glacier}')
                     if file.endswith('.asc') and file.find('surface') >= 0][0]

        # Finds the corresponding shapefile path
        shp_path += f'\\{glacier}\\shapefiles\\' + \
                    [file for file in os.listdir(f'{shp_path}\\{glacier}\\shapefiles') if file.endswith('.shp')][0]

        # Finds the corresponding true_bed_path
        true_bed_path += f'\\03_RES_{glacier}.txt'
        raster_path = true_bed_raster_path + f'\\{glacier}_true_bed_elev.tif'

        # Opens the shapefile and verifies it's a polygon
        # Help from: https://gis.stackexchange.com/questions/340938/convert-lines-to-polygons-with-shapely-polygonize
        with fiona.open(shp_path, 'r') as shp:
            projection = shp.crs
            geometry = [feature['geometry'] for feature in shp]
            if geometry[0]['type'].find('Polygon') < 0:
                # geometry = [mapping(shape(x).convex_hull) for x in geometry] Gives some weird bugs with multilines
                geometry[0]['type'] = 'Polygon'



        # Opens the true bed raster if it already exists
        if any(glacier in file for file in os.listdir(true_bed_raster_path)):
            true_bed_raster = rasterio.open(raster_path, 'r+')

        # If not, burns it and use it
        else:
            print(f'True bed raster unavailable. Trying to rasterize...\n')
            try:
                burns_csv_to_raster(true_bed_path, projection, geometry, DEM_path, output_fn=raster_path, crop=crop)
                true_bed_raster = rasterio.open(raster_path)
            except Exception as e:
                # Academy's thickness data is non existent in the files
                print(f'Problem with {glacier}')
                print(traceback.format_exc())
                continue

        # Opens the DEM
        dem_raster = rasterio.open(DEM_path, 'r')

        # Iterates over every model available for the glacier
        for model in models:
            print(f'Comparing {glacier} data with {model}')
            # Finds the corresponding bedrock model
            bedrock_path = itmix_path + '\\02_ITMIX_results'
            bedrock_path += f'\\{model}_{glacier}_bedrock.asc'
            try:
                error = compare_glacier_to_model(glacier, model, geometry, bedrock_path,
                                                 true_bed_raster, dem_raster, img_folder)
                table = error_to_table(error)
                error_dataFrame[glacier].loc[model] = table

            except Exception as e:
                print(f'Problem with {glacier} and {model}')
                print(traceback.format_exc())

        # Closes the data
        true_bed_raster.close()
        dem_raster.close()

    output_xl = 'itmix_results.xlsx'
    df_dict_to_excel(error_dataFrame, output_xl)
