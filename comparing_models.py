import numpy as np
import pandas as pd
import os
from skimage import io
from PIL import Image
from PIL.TiffTags import TAGS
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import rasterio
from rasterio import features
import rasterio.mask
import gdal
import os
from osgeo import ogr, osr
import fiona
import geopandas as gpd
from shapely.geometry import shape, Point
import xlsxwriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from descartes import PolygonPatch


def hypsometric_curve(raster, nbins, normalized=True, show_figures=False,
                      fig_name=None):  # Plots the hypsometric curve. Returns relative hypsometric data

    if raster.find('.asc') > 0:
        txt_file = open(raster)
        no_data = float(txt_file.readlines()[5].split(' ')[-1])  # Finds the no data value
        txt_file.close()
        z = np.loadtxt(raster, skiprows=6)  # Loads the file as an array
        z[z == no_data] = np.NaN  # Eliminates the no data values, could be improved but eh whatever

    else:  # raster.find('.tif') > 0: # Cases where its a .tif
        img = Image.open(raster)
        z = np.array(img)
        z[z < 0] = np.NaN  # Eliminates negative values

    z = z[np.logical_not(np.isnan(z))]  # Drops nan values
    z = z.flatten()  # Does hypsometric curves take into account the area of each pixel with respect to its slope?
    total_pixels = len(z)

    if normalized:  # Normalizes the elevation values to percentages if need be
        amplitude = np.max(z) - np.min(z)
        z = (z - np.min(z)) / amplitude

    elevations = [np.max(z) * (100 - i) / 100 for i in range(0, 100, int(100 / nbins))]  # Computes the bins
    hypso = np.zeros(shape=(nbins, 1))  # Initializes the hypsometry array
    hypso_histogram = np.zeros_like(hypso)
    for i in range(nbins):  # Computes the hypsometric curve
        nbr = np.sum(z > elevations[i])  # Finds the number of pixels with elevation greater than the ith percentile
        hypso[i] = nbr / total_pixels  # Relative area
        if i == 0:
            hypso_histogram[i] = hypso[i]
        else:
            hypso_histogram[i] = hypso[i] - hypso[i - 1]

    if fig_name is None:  # Finds the name according the ITMIX convention
        backslash_index = raster[::-1].find('\\')
        fig_name = raster[-backslash_index:-4].split('_')[2]

    fig = plt.figure()
    plt.plot(hypso, elevations, label='Cumulative area')
    plt.plot(hypso_histogram, elevations, label='Histogram')
    plt.title(f'Hypsometric curve of {fig_name}')
    plt.xlabel('Relative area')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.grid()

    if normalized:
        plt.ylabel('Relative height')

    try:
        os.makedirs('hypsometric_curves')
    except Exception as e:
        pass

    fig.savefig(f"hypsometric_curves\\{fig_name}_hypso.png")
    plt.close('all')

    if show_figures:
        plt.show()

    return hypso, hypso_histogram


def compare_correlation(xs, ys, show_figures=False):
    # Computes the pearson coefficient for each member of x to every members of y
    # Takes in a DataFrame, could be bettered
    rs = pd.DataFrame(columns=ys.columns, index=xs.columns)
    try:
        os.makedirs('covariance_curves')
    except Exception as e:
        pass

    for i in xs.columns:
        x = xs[i]
        for j in ys.columns:
            y = ys[j]
            # r = ((x*y).mean() - x.mean()*y.mean())/(x.std()*y.std())  # Pearson coefficient
            r = 1 - np.sum((x - y) ** 2) / np.sum((x - x.mean()) ** 2)  # R^2 coefficient
            rs[j][i] = r

            fig = plt.figure()
            plt.plot(x, y)
            plt.plot([0, 1], [0, 1], linestyle='dashed')
            plt.xlabel(i)
            plt.ylabel(j)
            plt.title(f'Covariance comparison of {i} and {j}')
            if show_figures:
                plt.show()

            fig.savefig(f'covariance_curves\\cov_{i}_{j}.png')
            plt.close('all')

    return rs


def compute_every_hypsometry():  # Void function computing the hypsometry of every glacier in itmix
    nbins = 100  # Number of bins for hypsometry

    job_path = 'Surface_DEM_Job.tif'
    kluane_path = 'Surface_DEM_Little_Kluane.tif'
    x_glaciers = {'Job': job_path, 'Little Kluane': kluane_path}

    xs = []

    for x in x_glaciers.keys():  # Appends the hypsometric series(?) for each glacier to be compared
        hypso, hysto = hypsometric_curve(x_glaciers[x], nbins, fig_name=x)
        xs.append(hypso)
    y_path = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\data\itmix\01_ITMIX_input_data'
    ys = []
    y_glaciers = []

    for folder in os.listdir(y_path):  # Iteratively go into each folders and computes the hypsometric series
        files = os.listdir(y_path + '\\' + folder)
        y_glaciers.append(folder)
        for file in files:
            if file.find('02') >= 0 and file.endswith('.asc'):  # The surface DEMs are named that way
                try:
                    print(f'Working on {file}\n')
                    hypso, hysto = hypsometric_curve(y_path + '\\' + folder + '\\' + file, nbins, show_figures=False)
                    ys.append(hypso)
                    print(f'Success\n')
                except Exception as e:
                    print(f'Problem with {file}: \n')
                    print(e)
                    print()

    xs = np.array(xs).transpose().reshape(nbins, len(xs))  # Fixing some problem with 3d array, eh whatever
    ys = np.array(ys).transpose().reshape(nbins, len(ys))

    # Saves the x and y data into csv files to save time
    xs = pd.DataFrame(xs, columns=['Job', 'Little Kluane'])
    ys = pd.DataFrame(ys, columns=y_glaciers)
    xs.to_csv('xs.csv', index=False)
    ys.to_csv('ys.csv', index=False)

    return xs, ys


def compare_every_curve(xs=None, ys=None):  # void function calle when comparing Job and little Kluane to other glaciers
    if (xs or ys) is None:
        xs = pd.read_csv('xs.csv')
        ys = pd.read_csv('ys.csv')

    rs = compare_correlation(xs, ys).transpose()
    rs_csv_path = 'hypsometry_compared_r2'
    rs.to_csv(rs_csv_path)
    job_max = np.max(rs['Job'])
    job_max_index = rs['Job'][rs['Job'] == job_max].index.to_list()
    lk_max = np.max(rs['Little Kluane'])
    lk_max_index = rs['Little Kluane'][rs['Little Kluane'] == lk_max].index.to_list()

    print(f'Job\'s most correlated glacier is {job_max_index} with {job_max}\n')
    print(f'Little Kluane\'s most correlated glacier is {lk_max_index} with {lk_max}\n')
    return rs_csv_path


def csv_ez_read(csv_path):
    csv = pd.read_csv(csv_path, index_col=0)
    job = csv['Job'].sort_values(ascending=False).reset_index()
    lk = csv['Little Kluane'].sort_values(ascending=False).reset_index()
    ez_view = pd.concat([job, lk], ignore_index=True, axis=1)
    ez_view = ez_view[ez_view.index < 5]
    ez_view.columns = ['job_index', 'job_r2', 'lk_index', 'lk_r2']
    ez_view['job_r2'] = ez_view['job_r2'].map('{:,.4f}'.format)
    ez_view['lk_r2'] = ez_view['lk_r2'].map('{:,.4f}'.format)
    ez_view.to_csv(csv_path.replace('.csv', '_ez_view.csv'), index=None)


def burns_csv_to_raster(csv_path, projection, geometry, ras_extent_path, output_fn=None, crop=True):

    # Transforms a x, y, z csv to a raster file
    # Imports the shapefile, its projection and its geometry
    # The shapefile will be used to mask the raster data

    datatype = 'float32'

    # Converts the csv to a geoDataFrame, which is basically a point shapefile
    try:
        df = pd.read_csv(csv_path, delim_whitespace=True, skiprows=13)
    except Exception as e:
        df = csv_path

    # Assigns it the same projection as the shapefile
    gdf = gpd.GeoDataFrame(df, crs=projection,
                           geometry=[Point(xy) for xy in zip(df.iloc[:, 0], df.iloc[:, 1])])

    # Imports the raster to make sure it has the same extent and hence the resolution is identical
    with rasterio.open(ras_extent_path) as ras_extent:
        ras_extent_im, out_transform = rasterio.mask.mask(ras_extent, geometry, crop=crop)
        metadata = ras_extent.meta.copy()
        metadata.update({"driver": "GTiff",
                         "height": ras_extent_im.shape[1],
                         "width": ras_extent_im.shape[2],
                         "transform": out_transform,
                         "crs": projection,
                         'dtype': datatype})

    # Gives the output file an appropriate name if none is given
    if output_fn is None:
        dot_index = csv_path.rfind('.')
        output_fn = csv_path.replace(csv_path[dot_index:],  '_raster.tif')

    # Burns the values onto a raster
    with rasterio.open(output_fn, 'w+', **metadata) as out:
        out_arr = out.read(1)
        shape = ((geom, value) for geom, value in zip(gdf.geometry, gdf.iloc[:, 2]))
        burned = features.rasterize(shapes=shape, fill=0, out=out_arr.astype(datatype), transform=out.transform)
        out.write_band(1, burned)
        return out


def compare_glacier_to_model(glacier, model, geometry, bedrock_path, true_bed_raster, dem_raster,
                             img_folder, crop=True, show_figures=False, percentage=True):
    #  Function comparing a glacier to its ITMIX modelled data
    #  Lots of help from internet, especially from this link:
    #  https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python

    datatype = 'float32'

    # Opens the modelled_bedrock.asv and crops it
    with rasterio.open(bedrock_path) as modelled_bedrock_ras:
        modelled_bedrock_im, out_transform = rasterio.mask.mask(modelled_bedrock_ras, geometry, crop=crop)
        out_meta = modelled_bedrock_ras.meta
        out_meta.update({'driver': 'GTiff',
                         'height': modelled_bedrock_im.shape[1],
                         'width': modelled_bedrock_im.shape[2],
                         'transform': out_transform,
                         'dtype': datatype})

    # Change the nodata values from the bedrock to nan
    modelled_bedrock_im = modelled_bedrock_im.astype(datatype)
    modelled_bedrock_im[modelled_bedrock_im <= 0] = np.NaN

    # Reads the values from the true bed raster as an array and change the nodata values to nan
    true_bed_im = true_bed_raster.read(1).astype(datatype)
    true_bed_im[true_bed_im <= 0] = np.NaN

    # Crops the DEM and change the nodata values to nan
    dem_im, out_transform = rasterio.mask.mask(dem_raster, geometry, crop=crop)
    dem_im = dem_im.astype(datatype)
    dem_im[dem_im <= 0] = np.NaN

    # Computes the modelled thickness and the the true thickness
    modelled_thickness_im = dem_im - modelled_bedrock_im
    true_thickness_im = dem_im - true_bed_im

    # Computes the error
    error = true_thickness_im - modelled_thickness_im
    if percentage:
        error = 100 * error / true_thickness_im

    #  The output filename of the error
    output_fn = f'error_tifs\\{glacier}_{model}_error.tif'

    # Burns the array to a raster
    with rasterio.open(output_fn, 'w+', **out_meta) as out:
        out.write(error)

    # Plots!!!
    fig, axs = plt.subplots(1, 2, figsize=(9, 6))
    fig.suptitle(f'{glacier} by {model}')

    # Ice thickness plot
    im0 = axs[0].imshow(modelled_thickness_im[0])
    axs[0].set_title('Modelled ice thickness (m)')

    # Colorbar 1
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('bottom', size=0.1, pad=0.4)
    fig.colorbar(im0, cax=cax, orientation='horizontal')

    # Error plot
    im1 = axs[1].imshow(error[0])
    axs[1].set_title('Error (%)')

    # Colorbar 2
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('bottom', size=0.1, pad=0.4)
    fig.colorbar(im1, cax=cax, orientation='horizontal')

    # Plotting shapefiles? doesn't work
    for i in range(len(axs)):
        for feature in geometry:
            axs[i].add_patch(PolygonPatch(feature, ec='blue'))

    fig.set_size_inches(9, 6)
    fig.savefig(f'{img_folder}\\{glacier}_{model}_error.png')

    if show_figures:
        plt.show()

    plt.close(fig)
    return error[np.logical_not(np.isnan(error))]


def error_to_table(error):
    # Mean, Std, Median error, Maximum error, Minimum error, Mean absolute error
    table = [error.mean(), error.std(), np.median(error), error.max(), error.min(), np.abs(error).mean()]
    return table


def df_dict_to_excel(df_dict, path):
    # Outputs a dictionary of DataFrame to an excel file with the DataFrame's key as the sheet name
    # https://stackoverflow.com/questions/51696940/how-to-write-a-dict-of-dataframes-to-one-excel-file-in-pandas-key-is-sheet-name
    writer = pd.ExcelWriter(path, engine='xlsxwriter')

    for tab_name, df in df_dict.items():
        df.to_excel(writer, sheet_name=tab_name)

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

if __name__ is '__main__':
    r"""ng_glathida_path = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\data\Thickness_Alaska\RGI60-01.16835_thickness.tif'
    true_bed_path = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\inversion\dem_tifs\NorthGlacier_true_bed_elev.tif'
    ng_dem = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\data\itmix\01_ITMIX_input_data\NorthGlacier\02_surface_NorthGlacier_2007_UTM07.asc'
    
    modelled_thickness = rasterio.open(ng_glathida_path).read(1)
    dem = rasterio.open(ng_dem).read(1)
    true_bed = rasterio.open(true_bed_path).read(1)
    true_bed[true_bed == 0] = np.NaN
    true_thickness = (dem - true_bed)
    error = 100*(true_thickness - modelled_thickness)/true_thickness
    table = error_to_table(error)
    print(table)"""
