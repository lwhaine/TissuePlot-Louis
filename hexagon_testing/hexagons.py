import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
from matplotlib.patches import RegularPolygon
from matplotlib.colors import Normalize


import matplotlib.pyplot as plt
from matplotlib import cm

df = pd.read_csv('SpotPositions_SP2.csv')

# def assign_to_grid()

df['hex_x'] = np.array(df.x - (np.sqrt(3)/3) * df.y)
df['hex_y'] = np.array(df.y )
X = np.array([df.hex_x, df.hex_y])

# Can we just cluster in the x and y axes separately?

# plt.hist(X[0],bins = 100)
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=40, min_samples=1).fit(X[0].reshape(-1,1))
x_labels= db.labels_

# plt.plot(labels)
plt.figure(figsize = (6,6))
plt.scatter(df.hex_x,df.hex_y, c=x_labels, cmap='rainbow', s = 1)

db = DBSCAN(eps=40, min_samples=1).fit(X[1].reshape(-1,1))
y_labels= db.labels_

# plt.plot(labels)
plt.figure(figsize = (10,10))
plt.scatter(df.hex_x,df.hex_y, c=y_labels, cmap='flag', s = 10)
plt.colorbar()


df['x_labels_dbscan'] = x_labels
df['y_labels_dbscan'] = y_labels

# now we want to use these labels to place the points within a grid 
df['group_x'] = df.groupby('x_labels_dbscan')['hex_x'].transform('mean')
df['group_y'] = df.groupby('y_labels_dbscan')['hex_y'].transform('mean')

# add some kind of warning here if std deviation is too high due to lots of gaps in the grid?
#  disable zoom scaling if this is the case?
x_dists = np.diff(np.sort(np.unique(df.group_x)))
y_dists = np.diff(np.sort(np.unique(df.group_y)))
mean_x_dist = np.mean(x_dists)
mean_y_dist = np.mean(y_dists)

recalc_x_dists = x_dists[np.abs(x_dists - mean_x_dist) < (mean_x_dist / 2)]
recalc_y_dists = y_dists[np.abs(y_dists - mean_y_dist) < (mean_y_dist / 2)]
recalc_mean_x_dist = np.mean(recalc_x_dists)
recalc_mean_y_dist = np.mean(recalc_y_dists)


# now we have our grid spacing, how do we choose a zero point
xmin = np.min(df.group_x)
xmax = np.max(df.group_x)
ymin = np.min(df.group_y)
ymax = np.max(df.group_y)

n_x = int(np.abs(xmax - xmin) // recalc_mean_x_dist) + 2
n_y = int(np.abs(ymax - ymin) // recalc_mean_y_dist) + 1

gridlines_x = df.group_x.min() + np.arange(0, n_x) * recalc_mean_x_dist
gridlines_y = df.group_y.min() + np.arange(0, n_y) * recalc_mean_y_dist

for i,row in df.iterrows():
    x_grid_ix = np.argmin(np.abs(gridlines_x - row.group_x))
    y_grid_ix = np.argmin(np.abs(gridlines_y - row.group_y))
    df.loc[i,'grid_x'] = x_grid_ix
    df.loc[i,'grid_y'] = y_grid_ix

plt.figure(figsize = (10,10))
plt.scatter(df.x,df.y, c=df.grid_x, s = 10)
plt.colorbar()

df = df.astype({'grid_x': 'int64', 'grid_y': 'int64'})

# Then we will group the spots into an array

genes=  pd.read_csv('/content/TopExpressedGenes_SP2.csv')

total_gene_expr = genes.sum(axis = 1)
df.loc[:,'total_gene_expr'] = total_gene_expr

plt.plot(total_gene_expr)

#  add spot objects to a grid to work out aggregation strategies

class Spot:
    def __init__(
            self,
            x = None,
            y = None,
            hex_x = None,
            hex_y = None,
            grid_x = None,
            grid_y = None,
            radius = 28.477165 + 40,
            gene_expr = -1,
            empty = False,
            ) -> None:
       self.x = x
       self.y = y
       self.hex_x = hex_x
       self.hex_y = hex_y
       self.grid_x = grid_x
       self.grid_y = grid_y
       self.radius = radius
       self.gene_expr = gene_expr
       self.empty = empty

neighbour_ix_offsets = [
    (0,0), #central spot
    (1,0), #immediate neighbours
    (-1,0),
    (0,1),
    (0,-1),
    (-1,1),
    (1,-1),
    (1,1), #corner spots 
    (2,-1),
    (-1,2),
    (-2,1),
    (-1,-1),
    (1,-2),
]
weights = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1/6,
    1/6,
    1/6,
    1/6,
    1/6,
    1/6,
]

def aggregate_spot_properties(spots,weights):
    total_gene_expr = 0
    for weight,spot in zip(weights,spots):
        total_gene_expr += spot.gene_expr * weight
    return total_gene_expr / np.sum(weights)


def construct_empty_spot_arrays(n_levels = 2):
    levels = []

    # set dimensions of first level
    this_level_dims = np.array([n_x,n_y])
    
    for n in range(n_levels):
        print(f'{this_level_dims = }')
        this_level_spots = np.full((this_level_dims[0],this_level_dims[1]),None,dtype = object)
        levels.append(this_level_spots)
        # reduce dimensions for next level, rounding up
        this_level_dims[0] = np.ceil(this_level_dims[0] / 3)
        this_level_dims[1] = np.ceil(this_level_dims[1] / 3)


    return levels


def plot_spots(spots,**plot_kwargs):
    norm = Normalize(vmin=total_gene_expr.min(), vmax=total_gene_expr.max(), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

    plt.gca().set_aspect('equal')
    for spot in spots.ravel():
        if spot is not None:
            # fix radius here
            colour = 'white' if spot.empty else mapper.to_rgba(spot.gene_expr) 
            edgecolour = '#c3c3bc' if spot.empty else 'white'
            hexagon = RegularPolygon((spot.x,spot.y), numVertices=6, radius=spot.radius, facecolor=colour,edgecolor = edgecolour,**plot_kwargs)
            plt.gca().add_patch(hexagon)
    plt.autoscale(enable = True)


def create_spot_array(df,gridlines_x,gridlines_y,n_levels = 2):
    spots = construct_empty_spot_arrays(n_levels = n_levels)

    print(f'{len(spots) = }')
    
    # level 0 
    for i,x in enumerate(gridlines_x):
        for j,y in enumerate(gridlines_y):
            spots[0][i,j] = Spot(
                x = x + (np.sqrt(3) / 3) * y,
                y = y,
                hex_x = x,
                hex_y = y,
                empty=  True
            )
    
    for i,row in df.iterrows():
        spot = Spot(
            x = row.x,
            y = row.y,
            hex_x = row.hex_x,
            hex_y = row.hex_y,
            grid_x = row.grid_x,
            grid_y = row.grid_y,
            radius = row.radius + 40,
            gene_expr = row.total_gene_expr,
            empty = False
        )
        spots[0][row.grid_x,row.grid_y] = spot

    # subsequent levels
    # loop over all spots in level 0
    previous_level = spots[0]
    print(len(spots))
    for this_level in range(1,len(spots)):
        print(f"Aggregating level {this_level}")
        # print()
        for i, row in enumerate(previous_level):
            for j,spot in enumerate(row):
                #  picking out the center of each larger hexagon
                if i%3==0 and j%3==0:
                    spots_to_aggregate = []
                    weights_to_aggregate = []
                    # print(f'New center spot ({i},{j}) in level of shape {previous_level.shape}')
                    # figure out which of the surroudning spots are occupied and add  them to a list
                    for weight,offset in zip(weights,neighbour_ix_offsets):
                        offset_i = i + offset[0]
                        offset_j = j + offset[1]   

                        # skip any spots outside the boundaries
                        if offset_i < 0 or offset_i >= previous_level.shape[0] or offset_j < 0 or offset_j >= previous_level.shape[1]:
                            continue
                        
                        # print(offset_i,offset_j)
                        candidate = previous_level[offset_i,offset_j]

                        # skip any empty spots
                        if not candidate.empty:
                            spots_to_aggregate.append(candidate)
                            weights_to_aggregate.append(weight)

                    # set -1 to be gene expression for empty cell

                    aggregated_spot_empty = len(spots_to_aggregate) < 1

                    # calculate aggregated properties if there are any spots there
                    if aggregated_spot_empty:
                        total_expr = -1
                    else:
                        total_expr = aggregate_spot_properties(spots_to_aggregate,weights_to_aggregate)

                    # Assign new spot object
                    aggregated_spot = Spot(
                        x = spot.x,
                        y = spot.y,
                        hex_x = spot.hex_x,
                        hex_y = spot.hex_y,
                        grid_x = spot.grid_x,
                        grid_y = spot.grid_y,
                        gene_expr= total_expr,
                        radius = 3 * spot.radius,
                        empty = aggregated_spot_empty,
                        )
                    # print(i//3,j//3)
                    # print(i,j)
                    spots[this_level][i//3,j//3] = aggregated_spot
        # setup for next level
        previous_level = spots[this_level]

    return spots


spots = create_spot_array(df,gridlines_x,gridlines_y,n_levels = 4)

for level in spots:
    fig = plt.figure(figsize = (10,10))
    plot_spots(level)