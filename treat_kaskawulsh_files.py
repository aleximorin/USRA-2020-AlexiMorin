import pandas as pd
import numpy as np
import geopandas as gpd
from dataclasses import dataclass
import shapely.geometry as geom
import fiona
import shapely.ops as ops
from Glaciers import Glacier, Model
import matplotlib.pyplot as plt
from typing import List

# This code was used to treat Becca's files and to produce ice thickness maps for Kaskawulsh glacier

# Simple class to keep track of lines and what reference line to
@dataclass
class Line:
    title: str
    ref: str
    flip: int


# Simple function to determine what points are in the box
def in_box(x, y, box):
    xmin, ymin, xmax, ymax = box
    return (xmin <= x) & (ymin <= y) & (x <= xmax) & (y <= ymax)



# Initalise paths and lines
path = r'C:\Users\Adalia Rose\Desktop\2scool4cool\e2020\data\kluane\kaskawulsh'
titles = ['KWL', 'NA', 'CA', 'SA', 'SW', 'KW1', 'KW2', 'KW3', 'KW4', 'KW5']
ref = 'lin nomig,nonlin nomig,lin nomig,nonlin mig,lin mig,nonlin nomig,nonlin nomig,lin nomig,lin mig,' \
      'nonlin nomig'.split(',')
flip = [0, 0, 0, 0, 0, 1, 0, 1, 1, 0]
lines = [Line(i, j, k) for i, j, k in zip(titles, ref, flip)]

# Opening the transect lines shapefile
shp = gpd.read_file('../data/kluane/kaskawulsh/Transect_lines.shp')

# The final dataframe to keep track of everything
totaldf = pd.DataFrame()

# Iteration over different lines
for line in lines:
    try:
        print(f'Processing line {line.title}')

        # Opening the allData file, containing the reference lines
        all_data = pd.read_csv(path + f'/CODE_DATABASE_WITH_relevant_files/LINE {line.title}_allData.csv')
        reference = line.ref.split(' ')
        reference = ''.join([reference[0], ' ref ', reference[-1]]).replace(' ', '_')
        thickness = all_data[reference].dropna()

        # Need to keep track of the number of thickness values to split the line accordingly
        N = len(thickness)

        # Temporary DataFrame to put the data in
        df = pd.DataFrame()

        # Split the line
        multilines = shp[shp['GATE'] == line.title].sort_values('Id')
        total_line = ops.linemerge(geom.MultiLineString([np.round(i.coords, 4) for i in multilines.geometry]))
        points = geom.MultiPoint([total_line.interpolate(i/N, normalized=True) for i in range(N)])
        coordinates = np.array([(i.x, i.y) for i in points])

        # Keep track of the x and y values
        x, y = coordinates[:, 0], coordinates[:, 1]
        df['x'] = x
        df['y'] = y

        if line.flip:
            df = df[::-1].reset_index(drop=True)

        df['Thickness'] = thickness
        df['GATE'] = line.title
        df['Inferred'] = False

        # We want to tag the measured and the interpolated data.
        # Here the inferred variable refers to every line from the shp with the tag "Inferred".
        inferred = multilines[multilines['Inferred'] == 1]

        # For every inferred line, we get their extent and check for points that fall within the box.
        # We tag those points accordingly
        for extent in [i.bounds for i in inferred.envelope]:
            df.loc[in_box(df['x'], df['y'], extent), 'Inferred'] = True

        # Concatenates the temporary dataframe to the more complete one
        totaldf = pd.concat([totaldf, df])

    except Exception as e:
        print(e)

totaldf.to_csv(path + '/kaska_gps.csv', index=False)

# Saves a correct csv for matlab
totaldf = totaldf[['x', 'y', 'Thickness']][totaldf['Inferred'] == False]
totaldf.to_csv(path + '/kaska_gps_for_m.csv', index=False)
