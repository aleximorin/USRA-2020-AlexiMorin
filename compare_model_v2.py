import rasterio
import geopandas as gpd
import pandas as pd
import shapely.geometry as geom
from rasterio.mask import mask
from rasterio import features
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from inversion.qgis_functions import crop_raster_to_geometry, resize_ras_to_target
import matplotlib.patches as patches
import os


class Model:

    def __init__(self, model_name, model_path, tag):
        self.name = model_name
        self.model_path = model_path
        self.tag = tag

        self.thickness = None
        self.thickness_array = None
        self.error = None
        self.rel_error = None
        self.error_array = None
        self.rel_error_array = None

    def set_thickness(self, im):
        self.thickness = im

    def set_thickness_array(self, array):
        self.thickness_array = array

    def set_error(self, error_im):
        self.error = error_im
        self.error_array = self.error[~np.isnan(self.error)]

    def set_rel_error(self, rel_error_im):
        self.rel_error = rel_error_im
        self.rel_error_array = self.rel_error[~np.isnan(self.rel_error)]


class Glacier:

    def __init__(self, name, dem_path, outline_path, gpr_path, glate_tag, img_folder='', delimiter=',',
                 whitespace=False, header=None, point_crs=None):

        # Basic parameters needed for saving the outputs
        self.name = name
        self.img_folder = img_folder + f'/{self.name}'
        self.glate_tag = glate_tag
        self.dtype = 'float64'

        # Creates the folder needed for the glacier's images
        try:
            os.makedirs(self.img_folder)
        except WindowsError:
            pass
        except Exception as e:
            print(e)

        # Opens the dem's raster
        self.dem = rasterio.open(dem_path, 'r')
        # Much simpler if the crs is from the DEM, no re-projection needed
        self.crs = self.dem.crs

        # Tries reading the gpr ice thickness data
        # Can either take as input a xyz formatted csv or a Pandas DataFrame
        try:
            df = pd.read_csv(gpr_path, delimiter=delimiter, delim_whitespace=whitespace, header=header)
        except ValueError:
            df = gpr_path
        except Exception as e:
            print("Could not read the csv or DataFrame\n")
            print(e)

        # Ensures the point geometry is the right one
        if point_crs is not None:
            self.gpr = gpd.GeoDataFrame(df, crs=point_crs,
                                        geometry=[geom.Point(xy) for xy in zip(df.iloc[:, 0], df.iloc[:, 1])])
            self.gpr = self.gpr.to_crs(self.crs)
        else:
            self.gpr = gpd.GeoDataFrame(df, crs=self.crs,
                                        geometry=[geom.Point(xy) for xy in zip(df.iloc[:, 0], df.iloc[:, 1])])

        # Ensures that the shapefile for the geometry has the same projection as the DEM
        self.outline = gpd.read_file(outline_path)
        if self.outline.crs != self.crs:
            self.outline = self.outline.to_crs(self.crs)

        # Creates a rectangle to be used in various plots from the extent of the GPR data
        # Current bug where the rectangle can only be used in one plot.
        # Could be fixed but it is weird and shouldn't happen
        c_xmin, c_ymin, c_xmax, c_ymax = self.gpr.total_bounds
        self.c_extent = self.gpr.total_bounds
        self.rectangle = patches.Rectangle((c_xmin, c_ymin), c_xmax - c_xmin, c_ymax - c_ymin,
                                           linewidth=2, edgecolor='black',
                                           facecolor='none')  # , label='Ground penetrating\nradar area')

        # Creates a shapely box to crop rasters from the point extent
        # Is currently not used, could be dropped
        bbox = geom.box(c_xmin, c_xmax, c_ymin, c_ymax)
        self.bbox = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=self.crs).geometry

        # Gets the image of the dem that falls within the extent of the outline
        # Gets the metadata and the extent of the new image
        self.dem_im, self.meta, self.extent = crop_raster_to_geometry(self.dem, self.outline.envelope)

        # Crops the dem from the outline's geometry, will be needed for the true thickness raster
        self.c_dem_im, self.c_meta, c_extent = crop_raster_to_geometry(self.dem, self.outline.geometry)

        # Computes a true thickness image to the shape of the cropped dem
        self.true_thickness_im = np.zeros_like(self.c_dem_im)
        self.true_thickness_im[self.true_thickness_im <= 0] = np.NaN
        shape = ((geometry, value) for geometry, value in zip(self.gpr.geometry, self.gpr.iloc[:, 2]))
        self.true_thickness_im = features.rasterize(shapes=shape, fill=-9999, out=self.true_thickness_im,
                                                    transform=self.c_meta['transform'])
        self.true_thickness_array = self.true_thickness_im[~np.isnan(self.true_thickness_im)]

        # A list of models, to be updated
        self.models = []

    def add_model(self, model):
        # Updates the list
        self.models.append(model)

    def compute_error(self, model):

        # Opens the given model and compute its error
        model_ras = rasterio.open(model.model_path)

        # Checks if the size of the pixels is the same
        if model_ras.transform[0] != self.dem.transform[0]:
            resized_ras = f'{self.img_folder}/{model.tag}_resized.tif'

            # Checks if there's already a created raster for this one
            try:
                model_ras = rasterio.open(resized_ras, 'r')

            # Creates the correct raster, could change the exception message
            except Exception as e:
                model_ras = resize_ras_to_target(model_ras, self.dem, resized_ras)

        # Crops the raster to the extent of the glacier
        im, meta, extent = crop_raster_to_geometry(model_ras, self.outline.geometry)
        model_ras.close()

        # Computes the error and various data
        error = im - self.true_thickness_im
        rel_error = 100 * error / self.true_thickness_im
        model.set_thickness(im)
        model.set_thickness_array(im[~np.isnan(self.true_thickness_im)])
        model.set_error(error)
        model.set_rel_error(rel_error)

    def plot_map(self, im, extent, title, cbar_unit, tag, cmap='jet', view_extent=None,
                 outline=False, points=False, point_color=False, rectangle=False):

        # Main plotting function. Ensures that all other plots have the same parameters

        # Manages the extent array. The order needed here is different
        b = [0, 2, 1, 3]
        extent = [extent[i] for i in b]
        xmin, xmax, ymin, ymax = extent

        if view_extent is not None:
            view_extent = [view_extent[i] for i in b]
            xmin, xmax, ymin, ymax = view_extent

        im = im[0]
        fig = plt.figure()
        ax = plt.gca()
        ax.set_title(f'{title}')

        # Plot the image and add a colorbar
        norm = Normalize(np.nanmin(im), np.nanmax(im))
        scamap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        img = ax.imshow(im, cmap=cmap, extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.2, pad=0.1)
        cbar = fig.colorbar(img, cax=cax, orientation='vertical')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label(f'{cbar_unit}')

        # Plots various map accessories if needed
        if points and point_color:
            ax.scatter(self.gpr.geometry.x, self.gpr.geometry.y, c=scamap.to_rgba(self.gpr.iloc[:, 2]),
                       cmap=cmap)
        elif points:
            self.gpr.plot(ax=ax, markersize=0.5)  # , label=data['points']

        if outline:
            self.outline.plot(ax=ax, facecolor='None', edgecolor='black')

        if rectangle:
            ax.add_patch(self.rectangle)

        # Customises the map
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks().tolist()])
        ax.set_xticklabels(['{:,.0f}'.format(x) for x in ax.get_xticks().tolist()])
        ax.tick_params(axis='x', labelrotation=-45)
        ax.tick_params(axis='y')
        ax.grid()

        # Defines a custom legend for the outline due to a geopandas bug
        # Currently no legend, need to figure out something for every model and dem source
        """handles, labels = ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='black', label=data['outline']))
        handles.append(matplotlib.patches.Patch(color='none', label=crs))
        ax.legend(handles=handles, bbox_to_anchor=(0.5, data['leg_pos']), loc="lower center",
                  bbox_transform=fig.transFigure, ncol=len(handles), frameon=True)"""
        fig.savefig(f'{self.img_folder}/{tag}.png')
        plt.close(fig)

    def plot_elevation(self):
        # Simple function call to plot the DEM
        self.plot_map(self.dem_im, self.extent, f'Surface elevation\n{self.name}',
                      '[m]', 'elevation', 'terrain', outline=True, points=True, rectangle=True)

    def scatterplot(self, model):
        # Plot the measured to modelled data
        fig = plt.figure()
        plt.scatter(self.true_thickness_array, model.thickness_array)
        x = [np.min(self.true_thickness_array), np.max(self.true_thickness_array)]
        plt.plot(x, x, label='$x=y$ line', color='orange')
        plt.xlabel('Measured thickness [m]')
        plt.ylabel('Modelled thickness [m]')
        plt.title(f'Measured and modelled thickness, {model.name} for {self.name}')
        plt.legend()
        fig.savefig(f'{self.img_folder}/xy_scatter_{model.tag}.png')
        plt.close(fig)

    def boxplot(self, errors, labels, title, yaxis, tag):
        # Plots the error for every model on a boxplot
        fig = plt.figure()
        ax = plt.gca()
        plt.title(title)
        plt.ylabel(yaxis)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.boxplot(errors, labels=labels)
        fig.savefig(f'{self.img_folder}/{tag}.png')
        plt.close(fig)

    def plot_boxplots(self):
        # Plot box plots
        labels = [f'{model.name}' for model in self.models]
        self.boxplot([model.error_array for model in self.models], labels, f'Ice thickness error for {self.name}',
                     'Error [m]', 'boxplot')
        self.boxplot([model.rel_error_array for model in self.models], labels,
                     f'Relative ice thickness error for {self.name}',
                     'Error [%]', 'boxplot_rel')

    def histogram(self, array, title, xlabel, tag):
        # Plots a histogram for a given model
        fig = plt.figure()
        n = int(np.ceil(np.sqrt(len(array))))
        plt.title(title)
        plt.ylabel('N')
        plt.xlabel(xlabel)
        plt.hist(array, n)
        fig.savefig(f'{self.img_folder}/{tag}.png')
        plt.close(fig)

    def all_models(self):
        # Main function, is used to compute to error and plot every model
        for model in self.models:
            print(f'Processing trough {model.tag}')
            self.compute_error(model)

            # Plots thickness maps
            self.plot_map(model.thickness, self.extent, f'Modelled thickness of {self.name}\n{model.name}', '[m]',
                          f'thickness_{model.tag}', outline=True, points=True, point_color=True)
            self.plot_map(model.thickness, self.extent, f'Modelled thickness of {self.name}\n{model.name}', '[m]',
                          f'thickness_cropped_{model.tag}', outline=True, points=True, point_color=True,
                          view_extent=self.c_extent)

            # Plots error maps
            self.plot_map(model.error, self.c_extent, f'Error\n{model.name}\n{self.name}', '[m]',
                          f'error_{model.tag}', outline=True,
                          view_extent=self.c_extent)
            self.plot_map(model.rel_error, self.c_extent, f'Relative error\n{model.name}\n{self.name}', '[m]',
                          f'error_rel_{model.tag}', outline=True,
                          view_extent=self.c_extent)

            # Plots scatter and histogram
            self.scatterplot(model)
            self.histogram(model.thickness_array, f'Ice thickness histogram of {model.name}',
                           'Ice thickness [m]', f'hist_thickness_{model.tag}')
            self.histogram(model.error_array, f'Error histogram of {model.name}',
                           'Error [m]', f'hist_error_{model.tag}')
           # self.histogram(model.rel_error_array, f'Relative error histogram of {model.name}',
             #              'Error [%]', f'hist_error_rel_{model.tag}')




if __name__ == '__main__':

    imgs = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\inversion\imgs'

    try:
        os.makedirs(imgs)
    except Exception as e:
        pass


    ng = Glacier('North glacier',
                 '../data/itmix/01_ITMIX_input_data/NorthGlacier/02_surface_NorthGlacier_2007_UTM07.asc',
                 '../data/itmix/01_ITMIX_input_data/NorthGlacier/shapefiles/01_margin_NorthGlacier_2007_UTM07.shp',
                 '../data/kluane/north_south/depth_GL2_080911.xyz', 'north_glacier',
                 whitespace=True, header=None, img_folder=imgs)
    ng.add_model(Model('Farinotti (2019)', '../data/Thickness_Alaska/RGI60-01.16835_thickness.tif', 'consensus'))

    sg = Glacier('South glacier',
                 '..\data\kluane\surface_DEM_RGI60-01.16195.tif',
                 '..\data\kluane\glacier_outlines\south_glacier_utm.shp',
                 '../data/kluane/north_south/depth_GL1_080911.xyz', 'south_glacier',
                 whitespace=True, header=None, img_folder=imgs)
    sg.add_model(Model('Farinotti (2019)', '..\data\Thickness_Alaska\RGI60-01.16195_thickness.tif', 'consensus'))

    lk = Glacier('little Kluane glacier',
                 '..\data\kluane\Surface_DEM_Little_Kluane.tif',
                 '../data\kluane\glacier_outlines\little_kluane.shp',
                 '../data/kluane/lk_gpr_data.csv', 'lk', header=0, img_folder=imgs)
    lk.add_model(Model('Farinotti (2019)', '..\data\kluane\little_kluane_thickness.tif', 'consensus'))

    job = Glacier('Job glacier',
                  '../data/job/resized_lidar_job.tif',
                  '../data\job\job_glacier_utm.shp',
                  '../data/job/gpr_data.csv', 'job', header=0, img_folder=imgs)
    job.add_model(Model('Farinotti (2019)', '..\data\Thickness_West\RGI60-02.01654_thickness.tif', 'consensus'))

    glaciers = [ng, sg, lk, job]
    glate_path = '../glate-master/results/'

    for i in glaciers:
        i.add_model(Model('Langhammer (2019) $\hat{h}^{glac}$', glate_path + f'{i.glate_tag}_unconstrained_hat.tif', 'glate_unconstrained_hat'))
        i.add_model(Model('Langhammer (2019) $h^{glac}$', glate_path + f'{i.glate_tag}_unconstrained_alpha.tif', 'glate_unconstrained_corrected'))
        i.add_model(Model('Langhammer (2019) $h^{est}$', glate_path + f'{i.glate_tag}_fullGlaTe.tif', 'glate_full'))

        print(f'Processing trough {i.name}')

        i.plot_elevation()
        i.all_models()
        i.plot_boxplots()
        i.dem.close()
        print('')
