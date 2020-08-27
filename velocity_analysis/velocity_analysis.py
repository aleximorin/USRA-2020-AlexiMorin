import rasterio
from rasterio.windows import Window
from rasterio.mask import geometry_mask
from pyGM.pyGM_funcs import *
import geopandas as gpd
import numpy as np
from pyGM.pyGM import Glacier, Model
import matplotlib.pyplot as plt
import shapely.geometry as geo
import requests
import os
import rasterio.warp as warp
from scipy.io import netcdf
from matplotlib import gridspec, rc
from datetime import datetime, date
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shapely.ops as ops
from scipy.signal import savgol_filter

def download_ITS_LIVE_data(region: str, years, output_path=''):
    if output_path != '':
        try:
            os.makedirs(output_path)
        except OSError:
            pass

    for year in years:

        url = f'http://its-live-data.jpl.nasa.gov.s3.amazonaws.com/velocity_mosaic/' \
              f'landsat/v00.0/annual/{region}_G0240_{year}.nc'
        try:
            print(f'Trying to fetch:\n{url}')
            r = requests.get(url, allow_redirects=True)

            with open(f'{output_path}/{region}_{year}.nc', 'wb') as dll:
                print(f'Writing down file in {output_path}...')
                dll.write(r.content)
        except Exception as e:
            print('Oups! Something occured.')
            print(e)


def open_ITS_LIVE_data(file_path, var, out_meta):

    file_path = f'netcdf:{file_path}:{var}'

    # Creates a box to crop the opened .nc from (needed?)
    bounds = rasterio.transform.array_bounds(out_meta['height'], out_meta['width'], out_meta['transform'])
    bbox = geo.box(*bounds)
    dst_gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=out_meta['crs'])

    with rasterio.open(file_path, 'r') as out:
        src_meta = out.meta.copy()
        im = out.read()

    src_bounds = rasterio.transform.array_bounds(src_meta['height'], src_meta['width'], src_meta['transform'])
    dst_t, dst_w, dst_h = warp.calculate_default_transform(src_meta['crs'], out_meta['crs'],
                                                           im.shape[-2], im.shape[-1], *src_bounds)

    dst_im = np.zeros((dst_h, dst_w))
    dst_im, dst_transform = warp.reproject(im, dst_im, src_transform=src_meta['transform'], src_crs=src_meta['crs'],
                                           dst_transform=dst_t, dst_crs=out_meta['crs'], dst_nodata=np.nan)

    src_meta.update({'height': dst_im.shape[-2],
                     'width': dst_im.shape[-1],
                     'transform': dst_transform,
                     'crs': out_meta['crs']})

    indices = np.where(~geometry_mask(dst_gdf.geometry, dst_im.shape, dst_transform))
    cr_im = dst_im[np.min(indices[0]):np.max(indices[0]) + 1, np.min(indices[1]): np.max(indices[1] + 1)]

    cr_im = cr_im.astype('float')
    cr_im[cr_im == src_meta['nodata']] = np.nan

    cr_transform, cr_w, cr_h = warp.calculate_default_transform(out_meta['crs'],
                                                                out_meta['crs'],
                                                                cr_im.shape[-1], cr_im.shape[-2],
                                                                *bounds)

    src_meta.update({'height': cr_im.shape[-2],
                     'width': cr_im.shape[-1],
                     'transform': cr_transform})  # transform is wrong here!!

    return cr_im, src_meta


def parse_ITS_LIVE_dates(mu_d, mu_dt, backward=True):
    months = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
              7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    d1 = date.fromordinal(mu_d)
    d0 = date.fromordinal(mu_d - mu_dt)
    timescale = f'{d0.month}-{d0.year} to {d1.month}-{d1.year}'
    return timescale


def vector_field_plots(vx_maps, vy_maps, vc_maps):
    N = int(len(vx_maps)/2)
    n = 3
    spec = gridspec.GridSpec(ncols=N, nrows=n)
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    vec_axs = [fig.add_subplot(spec[0, i]) for i in range(N)]

    xmin, xmax, ymin, ymax = [lk.extent[i] for i in [0, 2, 1, 3]]
    x = np.linspace(xmin, xmax, velocity_maps[0].shape[-1])
    y = np.linspace(ymin, ymax, velocity_maps[0].shape[-2])
    x, y = np.meshgrid(x, y)

    for vx_im, vy_im, vc_im, ax in zip(vx_maps, vy_maps, vc_maps, vec_axs):
        ax.quiver(x, y, vx_im/vc_im, vy_im/vc_im)
        ax.set_aspect('equal')

    plt.show()


def flowline_velocity_plots(lines, v_maps, timescales, meta, share_x=False, showfigs=False, filter=False):
    colors = plt.cm.get_cmap('jet')(np.linspace(0, 1, len(v_maps)))
    indices = []
    xys = []

    for line in lines:
        ii, xy = indices_along_line(line, v_maps[0].shape, meta)
        indices.append(ii)
        xys.append(xy)

    distances = [cumulative_distances(xy[0], xy[1]) for xy in xys]
    xmax = max([max(i) for i in distances])
    flowline_id = 0

    for x_dist, index, flowline in zip(distances, indices, flowlines):
        i = 0
        N = len(v_maps)
        spec = gridspec.GridSpec(ncols=N, nrows=1)
        fig = plt.figure()
        fig.set_size_inches(10, 5)
        v_ax = fig.add_subplot(spec[0, :int(N / 2) + 1])
        map_ax = fig.add_subplot(spec[0, int(N / 2) + 1:])
        # vec_axs = [fig.add_subplot(spec[1, i]) for i in range(N)]
        #fig.suptitle('Evolution of the velocity field descending a given centerline\nLittle Kluane glacier')

        map_ax.plot(*flowline.xy, c='red')

        n = int(np.floor(flowline.length/1000))
        points = [flowline.interpolate(i*1000) for i in range(1, n+1)]
        x = [i.x for i in points]
        y = [i.y for i in points]
        map_ax.scatter(x, y, c='red')

        lk.plot_map(lk.dem_im, outline=True, ax=map_ax, hillshade=True, alpha=0)

        for v_im, label in zip(v_maps, timescales):
            vc = v_im[index]
            if filter:
                x_dist = x_dist[~np.isnan(vc)]
                vc = vc[~np.isnan(vc)]
                #l = np.floor(len(vc)/2) + 1 if np.floor(len(vc)/2) % 2 == 0 else np.floor(len(vc)/2)
                l = 7
                vc = savgol_filter(vc, int(l), 3)

            v_ax.plot(x_dist, vc, label=label, c=colors[i], marker='.')
            i += 1

        v_ax.set_xlabel('Distance along centerline $[m]$')
        v_ax.set_ylabel('$u_s$ [m$\,$year$^{-1}$]')

        if share_x:
            v_ax.set_xlim(0, xmax)

        v_ax.legend()
        v_ax.grid()
        plt.tight_layout()
        if showfigs:
            plt.show()
        fig.savefig(f'lk_velocity_yearly_flowline{flowline_id}.png')
        v_ax.set_yscale('log')
        fig.savefig(f'log_lk_velocity_yearly_flowline{flowline_id}.png')
        flowline_id += 1
        plt.close(fig)


def compare_velocities(v_ims, v_metas, flowlines, labels=None, title=f'lk_velocities_compared_',
                       showfigs=False, filter=False, ylabel='', aspect=None):

    b = [0, 2, 1, 3]
    v_bounds = [rasterio.transform.array_bounds(*im.shape, meta['transform'])[i] for i in b
                for im, meta in zip(v_ims, v_metas)]
    i = 0
    for line in flowlines:
        spec = gridspec.GridSpec(ncols=2, nrows=1)
        fig = plt.figure()
        fig.set_size_inches(10, 5)
        v_ax = fig.add_subplot(spec[0, 0])
        map_ax = fig.add_subplot(spec[0, 1])
        v_ax.grid()

        lk.plot_map(lk.dem_im, outline=True, ax=map_ax, hillshade=True,  grid=True, alpha=0)
        x, y = line.xy
        map_ax.plot(x, y, c='red')
        n = int(np.floor(line.length / 1000))
        points = [line.interpolate(i * 1000) for i in range(1, n + 1)]
        x = [i.x for i in points]
        y = [i.y for i in points]
        map_ax.scatter(x, y, c='red')

        for im, meta in zip(v_ims, v_metas):
            indices, points = indices_along_line(line, im.shape, meta)
            v = im[indices]
            d = cumulative_distances(*points)
            if filter:
                l = len(v) - 1 if len(v) % 2 == 0 else len(v)
                v = savgol_filter(v, l, 3)
            v_ax.plot(d, v)

        if labels is not None:
            v_ax.legend(labels)
        v_ax.set_ylabel(ylabel, rotation=0, labelpad=20)
        v_ax.set_xlabel('[m]')
        if aspect is not None:
            v_ax.set_aspect(aspect)
        plt.tight_layout()
        fig.savefig(title + f'_flowline_{i}')

        if showfigs:
            plt.show()
        plt.close(fig)
        i += 1


def timescales_plots(dates_maps, dt_maps, velocity_maps, timescales, meta):
    i = 0
    extent = [lk.extent[i] for i in [0, 2, 1, 3]]
    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=4, nrows=len(dates_maps))
    fig.set_size_inches(10, 10)

    mask = geometry_mask(outline.geometry, dt_maps[0].shape, meta['transform'])
    scale_dict = {'length': 2, 'color': 'black'}
    for date_im, dt_im, v_im, scale, in zip(dates_maps, dt_maps, velocity_maps, timescales):

        for im in [date_im, dt_im, v_im]:
            im[mask] = np.NaN

        text_ax = fig.add_subplot(spec[i, 0])
        date_ax = fig.add_subplot(spec[i, 1])
        dt_ax = fig.add_subplot(spec[i, 2])
        v_ax = fig.add_subplot(spec[i, 3])

        for ax in [date_ax, dt_ax, v_ax]:
            lk.plot_map(lk.dem_im, outline=True, ax=ax, hillshade=True, scale_dict=scale_dict, grid=False,
                        ticks=False)

        date_ax.imshow(date_im - np.nanmin(date_im), extent=extent)
        date_ax.set_title('Number of days')

        dt_ax.imshow(dt_im, extent=extent)
        dt_ax.set_title('$\\Delta t$ [days]')

        v_ax.imshow(v_im, extent=extent)
        v_ax.set_title('Velocity')

        day_max = date.fromordinal(int(np.nanmax(date_im)))
        day_min = date.fromordinal(int(np.nanmin(date_im)))
        day_mu = date.fromordinal(int(np.nanmean(date_im)))

        string = f'Year {year0 + i}\nTimescale from {scale} ' \
                 f'\n Day min: {day_min}, Day max: {day_max}' \
                 f'\n Average day: {day_mu}, Average $\\Delta t$: {int(np.nanmean(dt_im))}'

        text_ax.text(0.5, 0.5, string, ha='center', va='center', transform=text_ax.transAxes)
        text_ax.set_axis_off()

        for ax in [date_ax, dt_ax, v_ax]:
            img = ax.images[-1]
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size=0.1, pad=0.05)
            cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
            cbar.ax.get_xaxis().labelpad = 5
            ax.set_xlim(*extent[:2])
            ax.set_ylim(*extent[2:])
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        i += 1

    plt.tight_layout()
    plt.show()
    fig.savefig(f'timescales_{year0}-{year0 + i}')


def get_data(get_from_raster, year0):
    if get_from_raster:
        velocity_maps = []
        dates_maps = []
        dt_maps = []
        timescales = []
        vx_maps = []
        vy_maps = []

        for nc_file in os.listdir(its_live_path):
            year = nc_file[4:-3]
            if nc_file.endswith('.nc') and int(year) >= year0:
                file_path = its_live_path + '/' + nc_file

                vc_im, vc_meta = open_ITS_LIVE_data(file_path, vars[0], meta)
                dates_im, dates_meta = open_ITS_LIVE_data(file_path, vars[6], meta)
                dt_im, dt_meta = open_ITS_LIVE_data(file_path, vars[7], meta)
                vx_im, vx_meta = open_ITS_LIVE_data(file_path, vars[1], meta)
                vy_im, vy_meta = open_ITS_LIVE_data(file_path, vars[2], meta)

                mean_date = int(np.nanmean(dates_im))
                mean_dt = int(np.nanmean(dt_im))

                timescale = parse_ITS_LIVE_dates(mean_date, mean_dt)

                # Collect velocity data in a list
                velocity_maps.append(vc_im)
                dates_maps.append(dates_im)
                dt_maps.append(dt_im)
                timescales.append(timescale)
                vx_maps.append(vx_im)
                vy_maps.append(vy_im)

        pickle.dump(velocity_maps, open('velocity_maps.p', 'wb'))
        pickle.dump(dates_maps, open('dates_maps.p', 'wb'))
        pickle.dump(dt_maps, open('dt_maps.p', 'wb'))
        pickle.dump(vc_meta, open('vc_meta.p', 'wb'))
        pickle.dump(timescales, open('timescales.p', 'wb'))
        pickle.dump(vx_maps, open('vx_maps.p', 'wb'))
        pickle.dump(vy_maps, open('vy_maps.p', 'wb'))

    else:
        velocity_maps = pickle.load(open('velocity_maps.p', 'rb'))
        dates_maps = pickle.load(open('dates_maps.p', 'rb'))
        dt_maps = pickle.load(open('dt_maps.p', 'rb'))
        timescales = pickle.load(open('timescales.p', 'rb'))
        vc_meta = pickle.load(open('vc_meta.p', 'rb'))
        vx_maps = pickle.load(open('vx_maps.p', 'rb'))
        vy_maps = pickle.load(open('vy_maps.p', 'rb'))

    return velocity_maps, dates_maps, dt_maps, vx_maps, vy_maps, timescales, vc_meta


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.rcParams.update({'figure.autolayout': False})
    print('caca')
    # Initialise path and parametes
    its_live_path = '../../data/ITS_LIVE'
    v_map_path = '../../bayesian_thickness/lk/postsurge/Velocity_31aout_1Oct/Correl_PAN_128to32_w32_AMP_m.year.tif'
    flowline_path = '../../bayesian_thickness/lk/postsurge/centerlines_postsurge.shp'
    #flowline_path = '../lk_oggm/glacier_centrelines_filled_arcticdem.shp'
    dem_path = '../../data/kluane/Surface_DEM_Little_Kluane.tif'
    shp_path = '../../data/kluane/glacier_outlines/little_kluane.shp'

    pre_dem = '../../bayesian_thickness/lk/presurge/lk_dem_presurge.tif'
    post_dem = '../../bayesian_thickness/lk/postsurge/lk_dem_postsurge.tif'

    dll = False

    vars = 'v vx vy v_err vx_err vy_err date dt count chip_size_max ocean rock ice'.split(' ')

    polar_crs = 'EPSG:3413'

    # Instantiates a glacier object, used for generating the elevation map
    lk = Glacier('little Kluane glacier', 'lk', dem_path, shp_path, img_folder='imgs')
    outline = gpd.read_file(shp_path)

    # Download the data if asked
    if dll:
        download_ITS_LIVE_data('ALA', [2019], its_live_path)

    # Opens the DEM and gets the important data

    with rasterio.open(dem_path) as dem_ras:
        dem_im = dem_ras.read(1)
        meta = dem_ras.meta.copy()
        crs = dem_ras.crs

    dem_im[dem_im < 0] = np.NaN

    flowlines_shp = gpd.read_file(flowline_path).to_crs(crs)
    flowlines = [line for line in flowlines_shp.geometry]

    year0 = 2008
    velocity_maps, dates_maps, dt_maps, vx_maps, vy_maps, timescales, vc_meta = get_data(False, year0)
    #vector_field_plots(vx_maps, vy_maps, velocity_maps)
    flowline_velocity_plots(flowlines, velocity_maps, timescales, vc_meta)#, filter=True)
    #timescales_plots(dates_maps, dt_maps, velocity_maps, timescales, vc_meta)

    #il_im, il_meta = open_ITS_LIVE_data(its_live_path + '/ALA_2018.nc', 'v', meta)
    """ with rasterio.open(v_map_path, 'r') as out:
        v_im = out.read(1)
        v_meta = out.meta.copy()


    v_im = crop_image_to_geometry(v_im, v_meta, lk.outline.geometry)
    lk.plot_map(v_im, tag='velocity', meta=v_meta, outline=True, cbar_unit='[$m/y$]',
                hillshade=True, showplot=True)"""
    #compare_velocities([il_im, v_im], [il_meta, v_meta], flowlines, ['ITS_LIVE', 'Étienne'])

    with rasterio.open(post_dem) as out:
        post_im = out.read(1)
        post_meta = out.meta.copy()

    with rasterio.open(pre_dem) as out:
        pre_im = out.read(1)
        pre_meta = out.meta.copy()

    for im in [pre_im, post_im]:
        im[im <= 0] = np.nan

    dhdt = '../../bayesian_thickness/dh_2018-2007_within150.tif'
    with rasterio.open(dhdt) as out:
        im = out.read(1)/11
        meta = out.meta.copy()

    compare_velocities([pre_im, post_im], [pre_meta, post_meta], flowlines,
                       ['2007', '2018'], title=f'lk_dem_compared', ylabel='[m$\\,$asl]')
    compare_velocities([im], [meta], flowlines, ['$\\frac{\\partial h} {\\partial t}$'], title='lk_dhdt',
                       ylabel='[m\\,year$^{-1}$]')



