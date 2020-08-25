import os
from oggm import cfg, workflow, tasks, graphics
from oggm.utils import gettempdir, get_demo_file
import oggm
import geopandas as gpd
import pandas as pd
import salem
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

orig_rgi_id = 'RGI60-01.16198'
ng_rgi_id = 'RGI60-01.16835'

pre_surge_dem = r'../bayesian_thickness/lk/presurge/lk_dem_2007.tif'
post_surge_dem = r'../bayesian_thickness/lk/postsurge/lk_dem_2018.tif'

pre_surge_shp = r'../bayesian_thickness/lk/presurge/lk_outline_presurge.shp'
post_surge_shp = r'../bayesian_thickness/lk/postsurge/lk_outline_postsurge.shp'

cfg.initialize()
cfg.PATHS['working_dir'] = '../bayesian_thickness/data'
cfg.PARAMS['border'] = 10
cfg.PARAMS['use_multiprocessing'] = False

cfg.PATHS['dem_file'] = post_surge_dem

gl = oggm.utils.get_rgi_glacier_entities([orig_rgi_id])

outline = gpd.read_file(post_surge_shp).to_crs('EPSG:4326')

template = pd.concat([gl], ignore_index=True)
template['Name'] = 'Little Kluane'
template['geometry'] = outline.geometry

for i, geom in template[['geometry']].iterrows():
    cenlon, cenlat = geom.geometry.centroid.xy
    template.loc[i, 'CenLon'] = cenlon
    template.loc[i, 'CenLat'] = cenlat

cfg.PARAMS['use_rgi_area'] = False
cfg.PARAMS['use_intersects'] = False

gdir = workflow.init_glacier_regions(template, reset=True, force=True)[0]

tasks.glacier_masks(gdir)
tasks.compute_centerlines(gdir)
tasks.initialize_flowlines(gdir)
tasks.compute_downstream_line(gdir)
tasks.catchment_area(gdir)
tasks.catchment_width_geom(gdir)
tasks.catchment_width_correction(gdir)

oggm.utils.write_centerlines_to_shape([gdir], filesuffix='_lk_postsurge', path=True)
"""
# Default parameters
# Deformation: from Cuffey and Patterson 2010
glen_a = 2.4e-24
# Sliding: from Oerlemans 1997
fs = 5.7e-20

# Correction factors
factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
factors += [1.1, 1.2, 1.3, 1.5, 1.7, 2, 2.5, 3, 4, 5]
factors += [6, 7, 8, 9, 10]

# Run the inversions tasks with the given factors
for f in factors:
    # Without sliding
    suf = '_{:03d}_without_fs'.format(int(f * 10))
    workflow.execute_entity_task(tasks.mass_conservation_inversion, gdir,
                                 glen_a=glen_a*f, fs=0)
    workflow.execute_entity_task(tasks.filter_inversion_output, gdir)
    # Store the results of the inversion only
    oggm.utils.compile_glacier_statistics(gdir, filesuffix=suf,
                                     inversion_only=True)

    # With sliding
    suf = '_{:03d}_with_fs'.format(int(f * 10))
    workflow.execute_entity_task(tasks.mass_conservation_inversion, gdir,
                                 glen_a=glen_a*f, fs=fs)
    workflow.execute_entity_task(tasks.filter_inversion_output, gdir)
    # Store the results of the inversion only
    oggm.utils.compile_glacier_statistics(gdir, filesuffix=suf,
                                     inversion_only=True)

# Log"""
