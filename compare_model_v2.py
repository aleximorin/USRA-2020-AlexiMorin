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
from pyGM.pyGM_funcs import crop_raster_to_geometry, resize_ras_to_target, \
    points_to_raster, df_dict_to_excel, crop_image_to_geometry
import pyGM.pyGM_funcs as pgmf
import matplotlib.patches as patches
import os
from pyGM.pyGM import Glacier, Model
import matplotlib.font_manager
from inversion.compare_alpha import alpha_joyplot, animate_samples


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


def plot_stuff(glaciers, xl_path=None):
    df_dict = dict()
    figsize = (3, 9)
    sf = False
    vertical = True
    for glacier in glaciers:
        print(f'Processing trough {glacier.name}')

        model_list = ['glate_unconstrained_hat', 'glate_unconstrained_corrected',
                      'glate_full']
        models = [glacier.models[i] for i in model_list]
        subtitles = ['$\\hat{h}^{glac}$', '$h^{glac}$', '$h^{est}$']

        glacier.maps_subplot(models, None, 'glate_thickness_all_models',
                             vertical=vertical, subtitles=subtitles, figsize=figsize, showfig=sf)
        model_list = ['consensus', 'glate_farinotti_alpha', 'glate_farinotti_full']
        models = [glacier.models[i] for i in model_list]

        glacier.maps_subplot(models, None, 'farinotti_glate_thickness_all_models',
                             vertical=vertical, attribute='thickness', subtitles=subtitles, figsize=figsize, showfig=sf)

        df_dict[glacier.name] = glacier.statistics

    if xl_path is not None:
        df_dict_to_excel(df_dict, xl_path, header=False, index=False)


def plot_lots_of_maps(glacier, model, showplot=True):
    fs = (6, 5)
    glacier.plot_map(im=glacier.dem_im,
                     cbar_unit='[m asl]',
                     tag=f'elevation',
                     outline=True,
                     cmap='terrain',
                     alpha=0.7,
                     hillshade=True,
                     ashape=True,
                     ticks=True,
                     labels=True,
                     showplot=showplot,
                     figsize=fs)

    glacier.plot_map(im=model.thickness,
                     cbar_unit='[m]',
                     tag=f'{model.tag}_thickness',
                     outline=True,
                     points=True,
                     point_color=True,
                     hillshade=True,
                     ticks=True,
                     labels=True,
                     meta=model.meta,
                     showplot=showplot,
                     figsize=fs)

    glacier.plot_map(im=model.thickness,
                     cbar_unit='[m]',
                     view_extent=model.point_extent,
                     tag=f'{model.tag}_cropped_thickness',
                     outline=True,
                     points=True,
                     point_color=True,
                     hillshade=True,
                     ticks=True,
                     labels=True,
                     meta=model.meta,
                     showplot=showplot,
                     figsize=fs)

    glacier.plot_map(im=model.error,
                     cbar_unit='[m]',
                     view_extent=model.point_extent,
                     tag=f'{model.tag}_cropped_error',
                     outline=True,
                     hillshade=True,
                     ticks=True,
                     labels=True,
                     meta=model.meta,
                     showplot=showplot,
                     figsize=fs)

    print()
    glacier.scatterplot(model,
                        showplot=showplot, figsize=(4, 4))

    glacier.histogram(model.error, xlabel='Error [m]',
                      tag=f'{model.tag}_error', showplot=showplot)


def plot_outlines_interlapping():
    with rasterio.open('../bayesian_thickness/lk/postsurge/lk_dem_postsurge.tif', 'r') as out:
        im = out.read(1)
        meta = out.meta.copy()
    bounds = [rasterio.transform.array_bounds(*im.shape, meta['transform'])[i] for i in [0, 2, 1, 3]]

    kluane_outline = gpd.read_file('../bayesian_thickness/lk/kluane_outline.shp').to_crs(lk_postsurge.crs)
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(pgmf.im_to_hillshade(im, 255, 40), extent=bounds, cmap='Greys')
    kluane_outline.plot(ax=ax, facecolor='green', alpha=0.6, edgecolor='black')
    lk_postsurge.outline.plot(ax=ax, facecolor='blue', alpha=0.6, edgecolor='black')
    xmin, ymin, xmax, ymax = kluane_outline.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax.set_xlabel('Eastings [m]')
    ax.set_ylabel('Northing [m]')
    plt.tight_layout()
    fig.savefig('../bayesian_thickness/lk/kluane_outlines.png')


def compute_vals(glacier, model):
    dA = model.meta['transform'][0] * -model.meta['transform'][4]
    gpr = model.thickness_array.size * dA * 10 ** -6
    model_area = (~np.isnan(model.thickness)).sum() * dA * 10 ** - 6

    dem = pgmf.crop_image_to_geometry(glacier.dem_im, glacier.meta, glacier.outline.geometry)
    maxH = np.nanmax(dem)
    minH = np.nanmin(dem)
    range_elev = maxH - minH

    print(f'{glacier.name} gpr coverage: {gpr}')
    print(f'{glacier.name} total area: {model_area}')
    print(f'{glacier.name} relative gpr coverage: {gpr / model_area * 100}')
    print(f'{glacier.name} max and min elevation: {maxH:.2f} - {minH:.2f}')
    print(f'{glacier.name} elevation range: {range_elev}')
    print()


def together_histogram(glaciers):
    data = [i.models['consensus'].error_array for i in glaciers]
    names = [i.name for i in glaciers]

    fig, axs = plt.subplots(len(glaciers), sharex=True)

    for ax, glacier in zip(axs.flatten(), glaciers):
        y = glacier.models['consensus'].error_array
        n = int(np.sqrt(len(y)))
        ax.hist(y, n)
        ax.set_title(glacier.name)
        ax.axvline(0, c='k', ls='dashed')
        # ax.grid(axis='x')

    axs[int(len(glaciers) / 2)].set_ylabel('N', rotation=0, labelpad=20)
    axs[-1].set_xlabel('Error [m]')
    axs[-1].xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.tight_layout()
    plt.show()
    fig.savefig('imgs/hists.png')


def five_glaciers_plot(glaciers, model_tag, showfig: bool = False):
    fig = plt.figure()
    grid = plt.GridSpec(3, 3, figure=fig)
    names = ['North', 'South', 'Kaskawulsh', 'Job', 'L. Kluane']
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])
    ax3 = fig.add_subplot(grid[1, 2])
    ax4 = fig.add_subplot(grid[2, 2])

    axs = [ax0, ax1, ax2, ax3, ax4]

    histax = fig.add_subplot(grid[1:, :2])

    for ax, glacier, name in zip(axs, glaciers, names):
        model = glacier.models[model_tag]
        glacier.scatterplot(model, ax=ax, label=False, legend=False, title=name, nbins=3, same_aspect=False)
        histax.hist(model.error_array, bins=int(np.sqrt(len(model.error_array))),
                    alpha=0.5, label=name, density=True)
    histax.legend()
    plt.tight_layout()
    if showfig:
        plt.show()
    fig.savefig(f'{imgs}/fiveplot_{model_tag}.png')


if __name__ == '__main__':

    imgs = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\inversion\imgs'
    xl_path = imgs + '/statistics.xlsx'
    try:
        os.makedirs(imgs)
    except OSError as e:
        pass

    ng = Glacier('North glacier', 'north_glacier',
                 '../bayesian_thickness/ng/ng_2007_dem.tif',
                 '../data/kluane/glacier_outlines/north_glacier_utm.shp',
                 '../data/kluane/north_south/depth_GL2_080911.xyz',
                 whitespace=True, header=None, img_folder=imgs)

    ng.add_model(Model('Farinotti (2019)', '../data/Thickness_Alaska/RGI60-01.16835_thickness.tif', 'consensus'))

    ng.add_model(Model('Farinotti + Glate (few points), $h^{est}$',
                       '../glate-master/results/north_glacier_specific_fullGlate.tif',
                       'fewpoints_glate_farinotti_full'))
    ng.add_model(Model('Farinotti + Glate (few points), $h^{glac}$',
                       '../glate-master/results/north_glacier_specific_model_unconstrained_alpha.tif',
                       'fewpoints_glate_farinotti_alpha'))

    sg = Glacier('South glacier', 'south_glacier',
                 '..\data\kluane\sg_highres_2018.tif',
                 '..\data\kluane\glacier_outlines\south_glacier_utm.shp',
                 '../data/kluane/north_south/depth_GL1_080911.xyz',
                 whitespace=True, header=None, img_folder=imgs)
    sg.add_model(Model('Farinotti (2019)', '..\data\Thickness_Alaska\RGI60-01.16195_thickness.tif', 'consensus'))

    lk = Glacier('little Kluane glacier', 'lk',
                 '..\data\kluane\Surface_DEM_Little_Kluane.tif',
                 '../data\kluane\glacier_outlines\little_kluane.shp',
                 '../data/kluane/lk_gpr_data.csv', header=0, img_folder=imgs)
    lk.add_model(Model('Farinotti (2019)', '..\data\kluane\little_kluane_thickness.tif', 'consensus'))

    """lk_postsurge = Glacier('little Kluane glacier, post-surge', 'lk_postsurge',
                           '../bayesian_thickness/lk/postsurge/lk_dem_postsurge.tif',
                           '../bayesian_thickness/lk/postsurge/lk_outline_postsurge.shp',
                           '../data/kluane/lk_gpr_data.csv', header=0, img_folder=imgs)
    lk_postsurge.add_model(Model('Farinotti (2019)', '..\data\kluane\little_kluane_thickness.tif', 'consensus'))"""

    kaska = Glacier('Kaskawulsh glacier', 'kaska',
                    '../data/kluane/surface_DEM_kaska.tif',
                    '../data/kluane/kaskawulsh_outline.shp',
                    '../data/kluane/kaskawulsh/kaska_gps.csv', header=0, img_folder=imgs)

    kaska.add_model(Model('Farinotti (2019)', '../data/kluane/kaska_thickness.tif', 'consensus'))

    job_points = gpd.read_file(r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\data\job\job_transects.shp')
    job_points = job_points[job_points['location_n'].str.find('line_6') < 0]
    job_points['location_n'] = [i.split('/')[0] for i in job_points['location_n']]
    # job_points['location_n'] = [i.split('_')[1] for i in job_points['location_n']]
    cols = job_points.columns.values

    job_points = job_points[[cols[i] for i in [2, 3, 0, 1, 4]]]

    job = Glacier('Job glacier', 'job',
                  '../data/job/clipped_lidar_job.tif',
                  '../data\job\job_glacier_utm.shp',
                  job_points, img_folder=imgs)
    job.add_model(Model('Farinotti (2019)', '..\data\Thickness_West\RGI60-02.01654_thickness.tif', 'consensus'))

    ng_fewpoints = Glacier('North glacier, fewpoints', 'ng_few',
                           '../bayesian_thickness/ng/ng_2007_dem.tif',
                           '../data/kluane/glacier_outlines/north_glacier_utm.shp',
                           '../data/kluane/specific_points_north_glacier.csv',
                           header=None, img_folder=imgs)
    mod = Model('Farinotti + Glate (few points), $h^{est}$',
                '../glate-master/results/north_glacier_specific_fullGlate.tif',
                'fewpoints_glate_farinotti_full')
    ng_fewpoints.add_model(mod)

    ng_fewpoints.plot_map(im=mod.thickness,
                          cbar_unit='[m]',
                          tag=f'{mod.tag}_thickness_ashape',
                          ashape=True,
                          outline=True,
                          hillshade=True,
                          ticks=True,
                          labels=True,
                          meta=mod.meta,
                          figsize=(6, 5))

    #plot_lots_of_maps(ng_fewpoints, ng_fewpoints.models['fewpoints_glate_farinotti_full'], showplot=False)

    glaciers = [ng, sg, kaska, job, lk]
    together_histogram(glaciers)
    glate_path = '../glate-master/results/'
    df_dict = {}
    plt.close('all')
    matplotlib.get_cachedir()
    add_models(glaciers, glate_path)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.rcParams.update({'figure.autolayout': False})
    # plot_stuff(glaciers)
    # five_glaciers_plot(glaciers, 'glate_full')
    # ng.scatterplot(ng.models['fewpoints_glate_farinotti_full'])
    for glacier in glaciers:
        model_list = ['glate_unconstrained_hat', 'glate_unconstrained_corrected',
                      'glate_full', 'glate_farinotti_alpha', 'glate_farinotti_full']
        models = [glacier.models[i] for i in model_list]
        for model in models:
            glacier.histogram(model.error_array, xlabel='Error [m]', tag=f'{model.tag}_error')
            glacier.scatterplot(model, figsize=(4, 4))
    plt.close('all')
    showplot = False
    add_models([ng, sg], glate_path)
    df = pd.DataFrame()


    for glacier in glaciers:
        
        plot_lots_of_maps(glacier, glacier.models['consensus'], showplot=False)

        #plot_outlines_interlapping()
        #compute_vals(glacier, glacier.models['consensus'])
        
        pass
    plt.close('all')
    r"""transects = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\data\kluane\kaskawulsh\Transect_lines.shp'
    tb_path = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\data\kluane\kaskawulsh\kaska_gps.csv'
    kaska.plot_transects_v2([kaska.models['consensus']], transects, field_name='GATE',
                          merge=True, showplot=True)

    transect_path = '../data/kluane/kaskawulsh/Transect_lines.shp'
    shape_path = '../data/kluane/kaskawulsh/terminus_line.shp'

    job.histogram(job.models['consensus'].rel_error, xlabel='Error [%]',
                  tag=f"consensus_rel_error")
    kaska.histogram(kaska.models['consensus'].rel_error, xlabel='Error [%]',
                  tag=f"consensus_rel_error")
    job_transects = gpd.GeoDataFrame(
        job_points.groupby('location_n')['geometry'].apply(lambda x: geom.LineString(x.tolist())), crs=job.crs,
        geometry='geometry')
    job.plot_transects_v2([job.models['consensus']], job_transects,
                          field_name='location_n', showplot=True, point_dist=100, text_color='white')"""

    """temp_path = 'glate_model.tif'
    beta, mod_im = kaska.glate_optimize(kaska.models['consensus'], save_output=temp_path)
    print(beta)
    glate_model = Model(f'{beta:.2f}_model', temp_path, tag='alpha')
    kaska.add_model(glate_model)
    kaska.scatterplot(glate_model, figsize=(6, 4), showplot=True)"""

    """cons = lk_postsurge.models['consensus']
    tt = (cons.thickness - cons.error).astype(cons.meta['dtype'])
    with rasterio.open('lk_true_thickness.tif', 'w+', **cons.meta) as out:
        out.write(tt)"""

    """df = pd.DataFrame()
    rdf = pd.DataFrame()
    
    cols = 'Glacier,Mean,Std,Median,Maximum,Minimum,Mean absolute,RMSE'.split(',')
    
    for glacier in glaciers:
        #plot_lots_of_maps(glacier, glacier.models['consensus'], showplot=True)

        print(f'Glacier {glacier.name}\'s:\n'
              f'GPR coverage is of {glacier.ashape.area[0] / 10 ** 6} km^2.\n'
              f'Total area is of {glacier.outline.area[0] / 10 ** 6} km^2.\n'
              f'Relative coverage is of {glacier.ashape.area[0] / glacier.outline.area[0]} %\n')
        rdf = rdf.append(glacier.statistics)
        df.loc[glacier.name, 'rel'] = 100 * glacier.ashape.area[0] / glacier.outline.area[0]
    df['rel'] = df['rel'].apply(lambda x: f'{x:.2f}')
    df.index.name = 'glac'
    df.to_csv(f'{imgs}/rel.csv')
    print()


    adf = df.loc[2]
    adf[0] = [i.name for i in glaciers]
    adf.columns = cols
    adf.iloc[:, 1:] = adf.iloc[:, 1:].applymap(lambda x: f'{x:.2f}')
    adf.to_csv(f'{imgs}/stats-abs.csv', index=False)
    print()

    rdf = rdf.loc[3]
    rdf[0] = [i.name for i in glaciers]
    rdf.columns = cols
    rdf[rdf.columns[1:]] = rdf[rdf.columns[1:]].applymap(lambda x: f'{x:.2f}')
    rdf.to_csv(f'{imgs}/stats-rel.csv', index=False)
    print()"""

    r"""job_transects = gpd.GeoDataFrame(
        job_points.groupby('location_n')['geometry'].apply(lambda x: geom.LineString(x.tolist())), crs=job.crs,
        geometry='geometry')
    transects = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\data\kluane\kaskawulsh\Transect_lines.shp'
    kaska.plot_transects_from_rasters([kaska.models['consensus']], transects, field_name='GATE',
                          merge=True, showplot=True)
    job.plot_transects_from_rasters([job.models['consensus']], job_transects,
                                       field_name='location_n', showplot=True,
                                    point_dist=100, text_color='white', simplify=5)
    kaska.plot_transects_from_gpr_points([kaska.models['consensus']], transects, field_name='GATE',
                                      merge=True, showplot=True, interp_field='Inferred')
    job.plot_transects_from_gpr_points([job.models['consensus']], job_transects,
                                    field_name='location_n', showplot=True, point_dist=100, text_color='white')"""
