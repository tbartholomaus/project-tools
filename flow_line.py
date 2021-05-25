#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  flow_line.py
'''
Script reads in a velocity dataset (currently only dataset of Joughin et al., 2017
is implemented), and then, given a starting point, integrates a flowline through
that starting point.  Core of computational funcationality is from
matplotlib.pyplot.streamplot.

This py script contains two key functions, and a couple of smaller, helper functions.
These two key functions are load_vel_field and flowline.  These should be run by
another script or notebook that extracts the flow lines.

Before running, be certain that the path to the velocity dataset is correct.

Created by Timothy Bartholomaus on 5/21/21.
 
'''

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import rasterio as rio

from scipy import stats
import time



def load_vel_field(source='Joughin2017'):
    '''
    Takes the a source (defaults to Joughin et al., 2017), and returns a dictionary
    vel_field, which consists of 1D fields x and y, and 2D fields X, Y, U, and V.
    U is the x direction velocity scalar, and V is the y direction velocity scalar.
    '''
    if source == 'Joughin2017':
        data_path = Path('/Users/timb/syncs/OneDrive - University of Idaho/RESEARCHs/ICESat-2_Greenland_termini/vel_data/')
        file_vx = Path('greenland_vel_mosaic250_vx_v1.tif') # https://nsidc.org/data/nsidc-0670/
        file_vy = Path('greenland_vel_mosaic250_vy_v1.tif')

    data_vx = rio.open( str(data_path / file_vx) )
    data_vy = rio.open( str(data_path / file_vy) )

    vel_mag = np.sqrt(data_vx.read(1)**2 + data_vy.read(1)**2) # m/a. Velocity magnitude
    # Filter out the no-data points
    masked = data_vx.read_masks(1) == 0
    vel_mag[masked] = np.nan

    im_extent = np.array([data_vx.bounds[0], data_vx.bounds[2], data_vx.bounds[1], data_vx.bounds[3]])
    im_extent = im_extent/1000 # km  change the plotting extents to kilometers
    
    x = np.arange(im_extent[0], im_extent[1], data_vx.transform[0]/1000)
    y = np.arange(im_extent[2], im_extent[3], -data_vx.transform[4]/1000)
    X, Y = np.meshgrid(x,y)
    U_temp = data_vx.read(1)
    U_temp[masked] = np.nan
    U = np.flipud(U_temp)
    V_temp = data_vy.read(1)
    V_temp[masked] = np.nan
    V = np.flipud(V_temp)
    
    if y[1] - y[0] < 0: # Identifies if y is decreasing rather than increasing
        y = np.flipud(y)
        Y = np.flipud(Y)
        U = np.flipud(U_temp)
        V = np.flipud(V_temp)
    
    vel_field = {"x":x, "y":y, "X":X, "Y":Y, "U":U, "V":V}
    return vel_field



def flowline(vel_field, starting_point):
    '''
    Takes a dictionary vel_field, consisting of velocity field information, and a
    starting location for the flowline called starting_point.  Returns the variable
    flowline, a Nx2 array consisting of the vertices of a flowline.
    '''
    # Unpack the dictionary of velocity field information
    x = vel_field['x'] # Assumed to be in km
    y = vel_field['y']
    X = vel_field['X'] # Assumed to be in km
    Y = vel_field['Y']
    U = vel_field['U']
    V = vel_field['V']
    
    x_ind_starting = np.argmin(np.abs(starting_point[0]-x))
    y_ind_starting = np.argmin(np.abs(starting_point[1]-y))
    xlen = len(x)
    ylen = len(y)

    diffs = np.diff(x)
    data_resolution = stats.mode(diffs)[0][0] # Resolution of the original velocity raster

    # Find what region the starting point is in.  This allows the program to only calculate streamlines within the appropriate region.  There are two options; uncomment one of them:
#    x_inds, y_inds = find_raster_subset_to_divide(starting_point, x_ind_starting, y_ind_starting) # Function defined below
    x_inds, y_inds = find_raster_subset_near_terminus(starting_point, x_ind_starting, y_ind_starting) # Function defined below

    figjunk, axjunk = plt.subplots(figsize=[0.1, 0.1]) # Make throw-away axes for use with streamplot
    
    factor = 100
    density = 1
    run_time = 1
    
    # Keep re-running streamplot, while the streamline resolution is far off
    # the data_resolution, and the run_time isn't too long
    while (factor > 1.01) & (run_time < 100):

        tic = time.perf_counter()
        flowline, line_deltas = exec_streamplot(
                                X[y_inds, x_inds], Y[y_inds, x_inds],
                                U[y_inds, x_inds], V[y_inds, x_inds], density,
                                starting_point, axjunk)
        toc = time.perf_counter()
        run_time = toc-tic
        
        # Identify the factor by which the density must be increased for the line spacing to equal the resolution of the underlying raster
        factor = line_deltas.max() / data_resolution
        print(f'Factor = {factor:.1f}; Max line spacing = {line_deltas.max():.2f} km; Run time = {run_time:.1f} s')
        density = density * factor

    axjunk.remove()

    print(f'Maximum point spacing on flowline is {line_deltas.max()*1000:.1f} m,'+
          f' and resolution of the original velocity dataset is {data_resolution*1000:.1f} m.')

    return flowline



def exec_streamplot( X, Y, U, V, density, starting_point, axjunk ):
    strm = axjunk.streamplot(X, Y, U, V, density=density, start_points=np.array([starting_point]))#, maxlength=100)
    num_pts = len(strm.lines.get_segments()) # count the number of points in the flowline
    flowline = np.full((num_pts, 2), np.nan) # initialize the flowline points
    for i in range(num_pts): # fill the flowline
        flowline[i,:] = strm.lines.get_segments()[i][0,:]
    
    line_deltas = np.sqrt(np.diff(flowline,axis=0)[:,0]**2 + np.diff(flowline,axis=0)[:,1]**2) # km
    line_length = np.sum( line_deltas ) # km
    dist_per_pt = line_length / num_pts
#    print(str(line_deltas.max()) + '  ' + str(dist_per_pt) )
    return flowline, line_deltas



def find_raster_subset_near_terminus(starting_point, x_ind_starting, y_ind_starting):
    # Find what region the starting point is in.  This allows the program to only calculate streamlines within the appropriate region.
    #   Loading in only slices of the velocity data speeds up the flowline calculation by nearly a factor of 10.
    if starting_point[1] < -2800: # Southern
        print('Short Flowline - starting point in region: South')
        x_inds = slice(x_ind_starting -400, x_ind_starting +400)
        y_inds = slice(y_ind_starting -200, y_ind_starting +400)
    elif (starting_point[1] < -1300) and (starting_point[0] < -50): # West
        print('Short Flowline - starting point in region:  West')
        x_inds = slice(x_ind_starting -400, x_ind_starting +400)
        y_inds = slice(y_ind_starting -200, y_ind_starting +200)
    elif (starting_point[1] < -1300) and (starting_point[0] > -50): # East
        print('Short Flowline - starting point in region:  East')
        x_inds = slice(x_ind_starting -600, x_ind_starting +400)
        y_inds = slice(y_ind_starting -200, y_ind_starting +200)
    elif (starting_point[1] >= -1300): # North
        print('Short Flowline - starting point in region: North')
        x_inds = slice(x_ind_starting -200, x_ind_starting +200)
        y_inds = slice(y_ind_starting -400, y_ind_starting +400)
    else:
        print('Short Flowline - starting point: Region Not Identified')
    return x_inds, y_inds
    
    

def find_raster_subset_to_divide(starting_point, x_ind_starting, y_ind_starting):
    # Find what region the starting point is in.  This allows the program to only calculate streamlines within the appropriate region.
    #   Loading in only slices of the velocity data speeds up the flowline calculation by nearly a factor of 10.
    if starting_point[1] < -2500: # Southern
        print('starting point in region: South')
        x_inds = slice(981, 4980)
        y_inds = slice(y_ind_starting - 1000, y_ind_starting + 1000)
    elif (starting_point[1] < -1950) and (starting_point[0] < 330): # Central West
        print('starting point in region: Central West')
#        x_inds = np.arange(981, 3840)
#        y_inds = np.arange(y_ind_starting - 1000, y_ind_starting + 1000)
        x_inds = slice(981, 3840)
        y_inds = slice(y_ind_starting - 1000, y_ind_starting + 1000)
    elif (starting_point[1] < -1950) and (starting_point[0] > 150): # Central East
        print('starting point in region: Central East')
        x_inds = slice(3380, 5990)
        y_inds = slice(y_ind_starting - 1000, y_ind_starting + 1000)
    elif (starting_point[1] < -1400) and (starting_point[0] < 300): # North West
        print('starting point in region: North West')
        x_inds = slice(60, 3780)
        y_inds = slice(y_ind_starting - 1000, y_ind_starting + 1000)
    elif (starting_point[1] < -1400) and (starting_point[0] > 0): # North East
        print('starting point in region: North East')
        x_inds = slice(2550, 5800)
        y_inds = slice(5680, 10280)
    elif (starting_point[1] >= -1400): # North
        print('starting point in region: North')
        x_inds = slice(60, 5800)
        y_inds = slice(5680, 10280)
    else:
        print('starting point: Region Not Identified')
    return x_inds, y_inds


#def find_flowline(X, Y, U, V, starting_point, density=10):
#
#    figjunk, axjunk = plt.subplots(figsize=[0.1, 0.1])
#
#    strm = plt.streamplot(X, Y, U, V, density=density, start_points=starting_point)
#
#    num_pts = len(strm.lines.get_segments())
#    flowline = np.full((num_pts, 2), np.nan)
#    for i in range(num_pts):
#        flowline[i,:] = strm.lines.get_segments()[i][0,:]
#
#    axjunk.remove()
#    return flowline
