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
from inversion.qgis_functions import crop_raster_to_geometry, resize_ras_to_target, points_to_raster, df_dict_to_excel
import matplotlib.patches as patches
import os
from Glaciers import Glacier, Model


def add_models(glaciers, glate_path):
    for glacier in glaciers:
        glacier.add_model(Model('GlaTe $\hat{h}^{glac}$', glate_path + f'{glacier.tag}_unconstrained_hat.tif',
                                'glate_unconstrained_hat'))

        glacier.add_model(Model('GlaTe $h^{glac}$', glate_path + f'{glacier.tag}_unconstrained_alpha.tif',
                                'glate_unconstrained_corrected'))

        glacier.add_model(Model('GlaTe $h^{est}$', glate_path + f'{glacier.tag}_fullGlaTe.tif',
                                'glate_full'))

        glacier.add_model(Model('GlaTe + F $h^{glac}$', glate_path + f'{glacier.tag}_model_unconstrained_alpha.tif',
                                'glate_farinotti_alpha'))

        glacier.add_model(Model('GlaTe + F $h^{est}$', glate_path + f'{glacier.tag}_model_fullGlate.tif',
                                'glate_farinotti_full'))


def plot_stuff(glaciers, df_dict, xl_path):
    for glacier in glaciers:
        print(f'Processing trough {glacier.name}')

        model_list = ['glate_unconstrained_hat', 'glate_unconstrained_corrected',
                      'glate_full']
        models = [glacier.models[i] for i in model_list]
        subtitles = [i.name.split(' ')[1] for i in models]
        glacier.maps_subplot(models, glacier.extent, 'glate_thickness_all_models',
                             vertical=True, subtitles=subtitles)
        glacier.maps_subplot(models, glacier.point_extent, 'glate_error_all_models',
                             vertical=True, attribute='error', subtitles=subtitles)
        glacier.plot_elevation()
        glacier.all_models()
        glacier.plot_boxplots()

        df_dict[glacier.name] = glacier.statistics

    df_dict_to_excel(df_dict, xl_path, header=False, index=False)


if __name__ == '__main__':

    imgs = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\inversion\imgs'
    xl_path = imgs + '/statistics.xlsx'
    try:
        os.makedirs(imgs)
    except OSError as e:
        pass

    ng = Glacier('North glacier', 'north_glacier',
                 '../data/kluane\surface_DEM_RGI60-01.16835.tif',
                 '../data/kluane/glacier_outlines/north_glacier_utm.shp',
                 '../data/kluane/north_south/depth_GL2_080911.xyz',
                 whitespace=True, header=None, img_folder=imgs)

    ng.add_model(Model('Farinotti (2019)', '../data/Thickness_Alaska/RGI60-01.16835_thickness.tif', 'consensus'))

    """
    ng.add_model(Model('Farinotti + Glate (few points), $h^{est}$',
                       '../glate-master/results/north_glacier_specific_fullGlate.tif',
                       'fewpoints_glate_farinotti_full'))
    ng.add_model(Model('Farinotti + Glate (few points), $h^{glac}$',
                       '../glate-master/results/north_glacier_specific_model_unconstrained_alpha.tif',
                       'fewpoints_glate_farinotti_alpha'))

    sg = Glacier('South glacier', 'south_glacier',
                 '..\data\kluane\surface_DEM_RGI60-01.16195.tif',
                 '..\data\kluane\glacier_outlines\south_glacier_utm.shp',
                 '../data/kluane/north_south/depth_GL1_080911.xyz',
                 whitespace=True, header=None, img_folder=imgs)
    sg.add_model(Model('Farinotti (2019)', '..\data\Thickness_Alaska\RGI60-01.16195_thickness.tif', 'consensus'))

    lk = Glacier('little Kluane glacier', 'lk',
                 '..\data\kluane\Surface_DEM_Little_Kluane.tif',
                 '../data\kluane\glacier_outlines\little_kluane.shp',
                 '../data/kluane/lk_gpr_data.csv', header=0, img_folder=imgs)
    lk.add_model(Model('Farinotti (2019)', '..\data\kluane\little_kluane_thickness.tif', 'consensus'))

    job = Glacier('Job glacier', 'job',
                  '../data/job/resized_lidar_job.tif',
                  '../data\job\job_glacier_utm.shp',
                  '../data/job/gpr_data.csv', header=0, img_folder=imgs)
    job.add_model(Model('Farinotti (2019)', '..\data\Thickness_West\RGI60-02.01654_thickness.tif', 'consensus'))

    kaska = Glacier('Kaskawulsh glacier', 'kaska',
                    '../data/kluane/surface_DEM_kaska.tif',
                    '../data/kluane/kaskawulsh_outline.shp',
                    '../data/kluane/kaskawulsh/kaska_gps_for_m.csv', header=0, img_folder=imgs)

    kaska.add_model(Model('Farinotti (2019)', '../data/kluane/kaska_thickness.tif', 'consensus'))

    glaciers = [kaska, ng, sg, job]
    glate_path = '../glate-master/results/'
    df_dict = {}
    transect_path = '../data/kluane/kaskawulsh/Transect_lines.shp'
    shape_path = '../data/kluane/kaskawulsh/terminus_line.shp'
    """
    """ 
    add_models(glaciers, glate_path)
    kaska_models = ['consensus', 'glate_farinotti_alpha']
    kaska_models = {key: kaska.models[key] for key in kaska_models}
    df = pd.read_csv('../data/kluane/kaskawulsh/kaska_gps.csv', keep_default_na=False)
    kaska.plot_transects(transect_path, kaska_models, merge_field='GATE', plot_surface=False, plot_outline=False,
                            vertical=False)
   
    kaska.plot_map(kaska.true_thickness_im, kaska.extent, 'Kaska\'s thickness', '[m]', 'true_thickness',
                   view_extent=kaska.point_extent, outline=True, point_color=True, points=True)

    kaska.plot_transects(transect_path, kaska_models, true_bed_df=df, merge_field='GATE', plot_surface=False,
                            plot_outline=False, vertical=False)
                            """

    ng.glate_optimize(ng.models['consensus'], save_output='test.tif')
    ng.add_model(Model('test', 'test.tif', 'test'))
    ng.glate_optimize(ng.models['test'])
    #kaska.compute_volume(kaska.models, shape_path, 'total_volume')
    #plot_stuff(glaciers,  df_dict, xl_path)
