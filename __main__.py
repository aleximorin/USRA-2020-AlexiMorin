import os
from comparing_models import *
from qgis_functions import *
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
import rasterio.warp
import rasterio.plot
import numpy as np
import traceback
import fiona
from rasterio import features
from shapely.geometry import shape, Polygon, mapping, Point
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import earthpy.spatial as es
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import matplotlib.patches as patches


def error_table(error):
    # Mean, Std, Median error, Maximum error, Minimum error, Mean absolute error, Mean square
    table = [error.mean(), error.std(), np.median(error), error.max(), error.min(), np.abs(error).mean(),
             np.sum(error ** 2) / len(error)]
    return table


if __name__ == '__main__':
    # job_path + 'Surface_DEM_Job_not_clipped.tif' Job lidar data

    # Need to get glacier outline, DEM and true bed thickness data
    data_path = '../data/'
    job_path = data_path + 'job/'
    lk_path = data_path + 'kluane/'

    # Make the directory
    img_folder = 'job_kluane_maps'
    try:
        os.makedirs(img_folder)
    except Exception as e:
        pass

    # Processing job's data
    job = {'name': 'Job glacier',
           'shp_path': job_path + 'outlines/Buffered_job_glacier.shp',  # Glacier outline
           'dem_path': job_path + 'surface_DEM_RGI60-02.01654.tif',  # DEM
           'tb_path': job_path + 'Meager_radar_picked_9Nov2018.csv',  # Radar data
           'glathida_path': '../data/Thickness_West/RGI60-02.01654_thickness.tif',  # GlaThiDa data
           'points': 'Ground penetrating radar\n(Flowers, 2018)',
           'outline': 'Glacier outline\n(GLIMS, XXXX)'}

    lk = {'name': 'Little Kluane glacier',
          'shp_path': lk_path + 'glacier_outlines/little_kluane.shp',  # Glacier outline
          'dem_path': lk_path + 'Surface_DEM_Little_Kluane.tif',  # DEM
          'tb_path': lk_path + '/surging_glacier/2019_radar/lk_radar_points.csv',  # Radar data
          'glathida_path': lk_path + 'little_kluane_thickness.tif',  # GlaThiDa data
          'points': 'Ground penetrating radar\n(Flowers, XXXX)',
          'outline': 'Glacier outline\n(Nolan, XXXX)'}

    datasets = [job, lk]
    cols = ['long', 'lat', 'ice_thickness', 'bedrock']
    alpha = 0.5
    projection = 'epsg:4326'
    errors = []
    rel_errors = []

    for data in datasets:

        # Processing the outline's geometry
        outline = gpd.read_file(data['shp_path'])

        # Process the lidar data, masks it to the extent of the shapefile and masks the no data values
        dem_ras = rasterio.open(data['dem_path'], 'r')
        dem_unclipped_im = dem_ras.read(1)
        dem_unclipped_im[dem_unclipped_im <= 0] = np.NaN
        outline = outline.to_crs(dem_ras.crs)
        geometry = outline.geometry
        dem_im, dem_out_transform = rasterio.mask.mask(dem_ras, geometry, crop=True)
        dem_metadata = dem_ras.meta.copy()
        dem_metadata.update({"driver": "GTiff",
                             "height": dem_im.shape[1],
                             "width": dem_im.shape[2],
                             "transform": dem_out_transform})
        dem_im[dem_im <= 0] = np.NaN

        # Imports the modelled bedrock data ,masks it to the extent of the shapefile and masks the no data values
        mod_thickness_ras = rasterio.open(data['glathida_path'], 'r')
        mod_thickness_im, mod_thickness_out_transform = rasterio.mask.mask(mod_thickness_ras, geometry, crop=True)
        mod_thickness_metadata = mod_thickness_ras.meta.copy()
        mod_thickness_metadata.update({"driver": "GTiff",
                                       "height": mod_thickness_im.shape[1],
                                       "width": mod_thickness_im.shape[2],
                                       "transform": mod_thickness_out_transform})
        mod_thickness_im[mod_thickness_im <= 0] = np.NaN

        # Opens the csv related to the ground penetrating radar data
        df = pd.read_csv(data['tb_path'], delimiter=';')

        # Specific job glacier case
        if data['tb_path'].find('Meager') >= 0:
            df = df[(df['location_name'].str.find('line_7') < 0) & (df['location_name'].str.find('line_6') < 0)]

        # Processing the radar points, ensuring they fall within the geometry
        lat, long = ddmm_to_decimal(df['Latitude_ddmm.mmmm'], df['Longitude_dddmm.mmmm'])
        ice_thickness = df['ice_thickness_m']
        bedrock = df['elevation_m'] - df['ice_thickness_m']
        df = pd.DataFrame([long, lat, ice_thickness, bedrock]).transpose()
        df.columns = cols
        points = gpd.GeoDataFrame(df, crs=projection,
                                  geometry=[Point(xy) for xy in zip(df.iloc[:, 0], df.iloc[:, 1])])
        points = points.to_crs(dem_ras.crs)

        # Gets the projection name for the map
        crs = dem_ras.crs.wkt.split(',')[0].split('"')[1].replace('/ ', '\n')

        # Gets the extent of the raster for the map
        extent = rasterio.plot.plotting_extent(dem_ras)

        # Crops the DEM to the points extents
        cropped_dem_im, cropped_dem_meta = es.crop_image(dem_ras, points)
        cropped_dem_meta.update({"driver": "GTiff",
                                 "height": cropped_dem_im.shape[1],
                                 "width": cropped_dem_im.shape[2],
                                 "transform": cropped_dem_meta["transform"]})

        # Gets the cropped extents for some map
        cropped_dem_extents = rasterio.transform.array_bounds(cropped_dem_meta['height'],
                                                              cropped_dem_meta['width'],
                                                              cropped_dem_meta['transform'])

        # Imports the modelled bedrock data and crops it to the right extent
        cropped_thickness_im, cropped_thickness_meta = es.crop_image(mod_thickness_ras, points)
        cropped_thickness_meta.update({"driver": "GTiff",
                                       "height": cropped_thickness_im.shape[1],
                                       "width": cropped_thickness_im.shape[2],
                                       "transform": cropped_thickness_meta["transform"]})

        cropped_thickness_extents = rasterio.transform.array_bounds(cropped_thickness_meta['height'],
                                                                    cropped_thickness_meta['width'],
                                                                    cropped_thickness_meta['transform'])
        b = [0, 2, 1, 3]
        cropped_extent = [cropped_thickness_extents[i] for i in b]
        cropped_thickness_im[cropped_thickness_im <= 0] = np.NaN

        # Computes the bedrock
        mod_bedrock_im = dem_im - mod_thickness_im
        cropped_bedrock_im = cropped_dem_im - cropped_thickness_im

        # Computes the error
        true_thickness_im = np.zeros_like(cropped_thickness_im)
        shape = ((geom, value) for geom, value in zip(points.geometry, points['ice_thickness']))
        true_thickness_im = features.rasterize(shapes=shape, fill=-9999, out=true_thickness_im,
                                               transform=cropped_thickness_meta['transform'])
        true_thickness_im[true_thickness_im <= 0] = np.NaN
        error = cropped_thickness_im - true_thickness_im
        errors.append(error[np.logical_not(np.isnan(error))])

        # Computes the relative error
        rel_error = 100 * error / true_thickness_im
        rel_errors.append(rel_error[np.logical_not(np.isnan(rel_error))])

        # Computes two arrays of thickness values for a x/y comparison
        true_thickness_array = true_thickness_im[np.logical_not(np.isnan(true_thickness_im))]
        mod_thickness_array = cropped_thickness_im * true_thickness_im / true_thickness_im
        mod_thickness_array = mod_thickness_array[np.logical_not(np.isnan(mod_thickness_array))]

        #########
        # Plots #
        #########

        ice_colormap = 'jet'
        show_3d = False
        show_2d = False
        # Changes the default font
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.size'] = 16
        matplotlib.rcParams['axes.titlesize'] = 20
        matplotlib.rcParams['axes.labelsize'] = 14
        matplotlib.rcParams['legend.fontsize'] = 12
        msize = 10

        # Defines the bounds of the smaller plot
        xmin, xmax, ymin, ymax = cropped_extent

        ##############
        # DEM Figure #
        ##############

        fig = plt.figure(figsize=(9, 7.5))
        ax = plt.gca()
        ax.set_title(data['name'])

        # Plots the DEM and adds a colorbar
        img = ax.imshow(dem_unclipped_im, cmap='terrain', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.2, pad=0.1)
        cbar = fig.colorbar(img, cax=cax, orientation='vertical')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label('Surface elevation (m)', rotation=270)

        # Plots the outline of the glacier and the ground penetrating radar data
        outline.plot(ax=ax, facecolor='None', edgecolor='black', label=data['outline'])
        points.plot(ax=ax, markersize=msize, label=data['points'])

        # Customises the map
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks().tolist()])
        ax.set_xticklabels(['{:,.0f}'.format(x) for x in ax.get_xticks().tolist()])
        ax.tick_params(axis='x', labelrotation=-45)
        ax.grid()
        fig.tight_layout()

        # Defines a custom legend for the outline due to a geopandas bug
        handles, labels = ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='black', label=data['outline']))
        handles.append(matplotlib.patches.Patch(color='none', label=crs))
        ax.legend(handles=handles, bbox_to_anchor=(0.5, -0.01), loc="lower center",
                  bbox_transform=fig.transFigure, ncol=len(handles), frameon=True, fontsize=18)
        fig.savefig(f'{img_folder}/{data["name"]}_elevation.png')

        ########################
        # Ice thickness figure #
        ########################

        fig = plt.figure(figsize=(9, 7.5))
        ax = plt.gca()
        ax.set_title(data['name'])

        # Plots the ice thickness and adds a colorbar
        img = ax.imshow(mod_thickness_im[0], cmap='jet', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.2, pad=0.1)
        cbar = fig.colorbar(img, cax=cax, orientation='vertical')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label('Ice thickness (m)', rotation=270)

        # Customises the map
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks().tolist()])
        ax.set_xticklabels(['{:,.0f}'.format(x) for x in ax.get_xticks().tolist()])
        ax.tick_params(axis='x', labelrotation=-45)
        ax.grid()
        fig.tight_layout()

        # Adds a rectangle to show the delimited area in the second ax
        rectangle = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                      linewidth=2, edgecolor='black', facecolor='none',
                                      label='Ground penetrating\nradar area')
        ax.add_patch(rectangle)

        # Defines a custom legend for the outline due to a geopandas bug
        handles, labels = ax.get_legend_handles_labels()
        handles.append(matplotlib.patches.Patch(color='none', label=crs))
        ax.legend(handles=handles, bbox_to_anchor=(0.5, -0.05), loc="lower center",
                  bbox_transform=fig.transFigure, ncol=len(handles), frameon=True, fontsize=18)
        fig.savefig(f'{img_folder}/{data["name"]}_thickness.png')

        ################################
        # Cropped ice thickness figure #
        ################################

        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_title(data['name'])

        # Plots the ice thickness and adds a colorbar
        norm = Normalize(np.nanmin(cropped_thickness_im), np.nanmax(cropped_thickness_im))
        scamap = plt.cm.ScalarMappable(cmap=ice_colormap, norm=norm)
        img = ax.imshow(cropped_thickness_im[0], cmap='jet', extent=cropped_extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.2, pad=0.1)
        cbar = fig.colorbar(img, cax=cax, orientation='vertical')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label('Ice thickness (m)', rotation=270)

        # Plots the point data and the outline
        ax.scatter(points.geometry.x, points.geometry.y, c=scamap.to_rgba(points['ice_thickness']),
                   label=data['points'], cmap=ice_colormap)
        outline.plot(ax=ax, facecolor='None', edgecolor='black')

        # Customises the map
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks().tolist()])
        ax.set_xticklabels(['{:,.0f}'.format(x) for x in ax.get_xticks().tolist()])
        ax.tick_params(axis='x', labelrotation=-45)
        ax.grid()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        fig.tight_layout()

        # Defines a custom legend for the outline due to a geopandas bug
        handles, labels = ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='black', label=data['outline']))
        handles.append(matplotlib.patches.Patch(color='none', label=crs))
        ax.legend(handles=handles, bbox_to_anchor=(0.5, -0.01), loc="lower center",
                  bbox_transform=fig.transFigure, ncol=len(handles), frameon=True, fontsize=15)
        fig.savefig(f'{img_folder}/{data["name"]}_cropped_thickness.png')

        #############################
        # Ice thickness subplot map #
        #############################

        fig, axs = plt.subplots(1, 2)
        fig.suptitle(data['name'], fontsize=30)

        # Plots the ice thickness
        norm = Normalize(np.nanmin(mod_thickness_im), np.nanmax(mod_thickness_im))
        scamap = plt.cm.ScalarMappable(cmap=ice_colormap, norm=norm)
        img0 = axs[0].imshow(mod_thickness_im[0], cmap=ice_colormap, extent=extent)
        img1 = axs[1].imshow(cropped_thickness_im[0], cmap=ice_colormap,
                             vmin=np.nanmin(mod_thickness_im), vmax=np.nanmax(mod_thickness_im),
                             extent=cropped_extent)
        # Adds a colorbar
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size=0.2, pad=0.1)
        cbar = fig.colorbar(img0, cax=cax, orientation='vertical')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label('Ice thickness (m)', rotation=270)

        # Bounds
        axs[1].set_xlim(xmin, xmax)
        axs[1].set_ylim(ymin, ymax)

        # Steps applied to both maps
        for i in range(len(axs)):
            # Plots the outline of the glacier and the ground penetrating radar data
            outline.plot(ax=axs[i], facecolor='None', edgecolor='black')
            axs[i].scatter(points.geometry.x, points.geometry.y, c=scamap.to_rgba(points['ice_thickness']),
                           cmap=ice_colormap, label=data['points'])

            # Customises the map
            axs[i].set_xlabel('x (m)')
            axs[i].set_yticklabels(['{:,.0f}'.format(y) for y in axs[i].get_yticks().tolist()])
            axs[i].set_xticklabels(['{:,.0f}'.format(x) for x in axs[i].get_xticks().tolist()])
            axs[i].tick_params(axis='x', labelrotation=-45)
            axs[i].grid()

        # Adds a rectangle to show the delimited area in the second ax
        rectangle = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                      linewidth=2, edgecolor='black', facecolor='none')
        axs[0].add_patch(rectangle)

        # Adds some other stuff
        axs[0].set_ylabel('y (m)')
        fig.tight_layout()

        # Defines a custom legend for the outline due to a geopandas bug
        handles, labels = axs[0].get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='black', label=data['outline']))
        handles.append(matplotlib.patches.Patch(color='none', label=crs))
        axs[0].legend(handles=handles, bbox_to_anchor=(0.5, -0.03), loc="lower center",
                      bbox_transform=fig.transFigure, ncol=len(handles), frameon=True, fontsize=18)
        fig.savefig(f'{img_folder}/{data["name"]}_thickness_subplots.png')

        #############
        # Error map #
        #############

        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_title(data['name'])

        # Plots the error and adds a colorbar
        img = ax.imshow(error[0], cmap='jet', extent=cropped_extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.2, pad=0.1)
        cbar = fig.colorbar(img, cax=cax, orientation='vertical')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label('Error (m)', rotation=270)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Plots the outline of the glacier and the ground penetrating radar data
        outline.plot(ax=ax, facecolor='None', edgecolor='black', label=data['outline'])
        points.plot(ax=ax, markersize=msize, color='black', label=data['points'], alpha=alpha)

        # Customises the map
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks().tolist()])
        ax.set_xticklabels(['{:,.0f}'.format(x) for x in ax.get_xticks().tolist()])
        ax.tick_params(axis='x', labelrotation=-45)
        ax.grid()
        fig.tight_layout()

        # Defines a custom legend for the outline due to a geopandas bug
        handles, labels = ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='black', label=data['outline']))
        handles.append(matplotlib.patches.Patch(color='none', label=crs))
        ax.legend(handles=handles, bbox_to_anchor=(0.5, -0.01), loc="lower center",
                  bbox_transform=fig.transFigure, ncol=len(handles), frameon=True)
        fig.savefig(f'{img_folder}/{data["name"]}_error.png')

        ######################
        # Relative error map #
        ######################

        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_title(data['name'])

        # Plots the error and adds a colorbar
        img = ax.imshow(rel_error[0], cmap='jet', extent=cropped_extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.2, pad=0.1)
        cbar = fig.colorbar(img, cax=cax, orientation='vertical')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label('Error (%)', rotation=270)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Plots the outline of the glacier and the ground penetrating radar data
        outline.plot(ax=ax, facecolor='None', edgecolor='black', label=data['outline'])
        points.plot(ax=ax, markersize=msize, color='black', label=data['points'], alpha=alpha)

        # Customises the map
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks().tolist()])
        ax.set_xticklabels(['{:,.0f}'.format(x) for x in ax.get_xticks().tolist()])
        ax.tick_params(axis='x', labelrotation=-45)
        ax.grid()
        fig.tight_layout()

        # Defines a custom legend for the outline due to a geopandas bug
        handles, labels = ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='black', label=data['outline']))
        handles.append(matplotlib.patches.Patch(color='none', label=crs))
        ax.legend(handles=handles, bbox_to_anchor=(0.5, -0.01), loc="lower center",
                  bbox_transform=fig.transFigure, ncol=len(handles), frameon=True)
        fig.savefig(f'{img_folder}/{data["name"]}_rel_error.png')

        ######
        # XY #
        ######

        # Reset plot settings
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)

        fig = plt.figure()
        ax = plt.gca()
        ax.scatter(true_thickness_array, mod_thickness_array)
        x = [np.min(true_thickness_array), np.max(true_thickness_array)]
        ax.plot(x, x, linestyle='dashed', color='orange', label='$x=y$ line')
        ax.set_xlabel('True thickness (m)')
        ax.set_ylabel('Modelled thickness (m)')
        ax.set_title(f'Measured versus modelled thickness data of {data["name"]}')
        ax.legend()
        fig.savefig(f'{img_folder}/{data["name"]}_xy.png')

        if show_2d:
            plt.show()
        plt.close('all')

        ############
        # 3D plots #
        ############

        # Name some colors and lines for the legend
        bedrock_color = 'grey'
        ice_color = 'cyan'
        point_color = 'blue'
        bed_line = mlines.Line2D([0], [0], linestyle='none', c=bedrock_color, marker='o')
        surface_line = mlines.Line2D([0], [0], linestyle='none', c=ice_color, marker='o')
        points_line = mlines.Line2D([0], [0], linestyle='none', c=point_color, marker='o')

        # Plots a 3D figure of the whole glacier area
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x = np.linspace(extent[0], extent[1], dem_im.shape[-1])
        y = np.linspace(extent[3], extent[2], dem_im.shape[-2])
        x, y = np.meshgrid(x, y)
        ax.plot_surface(x, y, dem_im[0], alpha=alpha, color=ice_color)
        ax.scatter(points.geometry.x, points.geometry.y, points['bedrock'])
        ax.set_title(f'Surface elevation of {data["name"]}')
        ax.w_xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
        ax.w_yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('Elevation (m)')
        ax.legend([surface_line, points_line], ['Surface elevation', data['points']], numpoints=1, loc='lower right')
        if show_3d:
            plt.show()
        fig.savefig(f'{img_folder}/3D_{data["name"]}_surface_bed.svg')

        # Plots a 3D figure of the glacier and the measured thickness points
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x = np.linspace(cropped_dem_extents[0], cropped_dem_extents[2], cropped_dem_im.shape[-1])
        y = np.linspace(cropped_dem_extents[3], cropped_dem_extents[1], cropped_dem_im.shape[-2])
        x, y = np.meshgrid(x, y)
        ax.plot_surface(x, y, cropped_dem_im[0], alpha=alpha, color=ice_color)
        ax.scatter(points.geometry.x, points.geometry.y, points['bedrock'])
        ax.set_title(f'Surface elevation of {data["name"]}')
        ax.w_xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
        ax.w_yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('Elevation (m)')
        ax.legend([surface_line, points_line], ['Surface elevation', data['points']], numpoints=1, loc='lower right')
        if show_3d:
            plt.show()
        fig.savefig(f'{img_folder}/3D_{data["name"]}_surface_true_bed.svg')

        # Plots a 3D figure of the glacier and the modelled ice thickness
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x = np.linspace(cropped_thickness_extents[0], cropped_thickness_extents[2], cropped_thickness_im.shape[-1])
        y = np.linspace(cropped_thickness_extents[3], cropped_thickness_extents[1], cropped_thickness_im.shape[-2])
        x, y = np.meshgrid(x, y)
        ax.plot_surface(x, y, cropped_thickness_im[0], alpha=alpha, color=ice_color)
        ax.scatter(points.geometry.x, points.geometry.y, points['ice_thickness'])
        ax.set_title(f'Ice thickness of {data["name"]}')
        ax.w_xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
        ax.w_yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('Ice thickness (m)')
        ax.legend([surface_line, points_line], ['Ice thickness', data['points']], numpoints=1, loc='lower right')
        if show_3d:
            plt.show()
        fig.savefig(f'{img_folder}/3D_{data["name"]}_ice_thickness.svg')

        # Plots a 3D figure of the glacier bedrock and the point data
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, cropped_bedrock_im[0], color=bedrock_color)
        ax.scatter(points.geometry.x, points.geometry.y, points['bedrock'], label=data['points'])
        ax.set_title(f'Bedrock of {data["name"]}')
        ax.w_xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
        ax.w_yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('Ice thickness (m)')
        ax.legend([bed_line], ['Modelled bedrock'], numpoints=1, loc='lower right')
        if show_3d:
            plt.show()
        fig.savefig(f'{img_folder}/3D_{data["name"]}_surface_bedrock.svg')

        if show_3d:
            plt.show()
        plt.close('all')

        mod_thickness_ras.close()
        dem_ras.close()

    # Box plots
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title('Error of the ice thickness estimates')
    ax.set_ylabel('Error (m)')
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.boxplot(errors, labels=[f'Job glacier\nN = {len(errors[0])}',
                               f'Little Kluane\nN = {len(errors[1])}'])
    fig.savefig(f'{img_folder}/boxplots.png')

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title('Relative error of the ice thickness estimates')
    ax.set_ylabel('Error (%)')
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.boxplot(rel_errors, labels=[f'Job glacier\nN = {len(errors[0])}',
                                   f'Little Kluane\nN = {len(errors[1])}'])
    fig.savefig(f'{img_folder}/rel_boxplots.png')

    with open(f'{img_folder}/data.csv', 'w') as txtfile:
        cols = 'Mean,Std,Median,Maximum,Minimum,Absolute mean,Mean square,'
        txtfile.write('Glacier,' + cols + '\n')
        for i in range(len(errors)):
            txtfile.write(f"{datasets[i]['name']},")
            table = error_table(errors[i])
            for num in table:
                txtfile.write(f'{num:2.4},')
            txtfile.write('\n')
        for i in range(len(errors)):
            txtfile.write(f"{datasets[i]['name']} (\%),")
            table = error_table(rel_errors[i])
            for num in table:
                txtfile.write(f'{num:2.4},')
            txtfile.write('\n')
