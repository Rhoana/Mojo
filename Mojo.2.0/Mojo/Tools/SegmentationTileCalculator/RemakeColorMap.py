import os
import sys
import string
import math
import mahotas
import PIL
import PIL.Image
import numpy
import scipy
import scipy.io
import cv2
import h5py
import lxml
import lxml.etree
import glob
import sqlite3
import colorsys

output_path                   = 'C:\\dev\\datasets\\Cube1x10\\mojo'
ncolors                       = 10000
output_ids_path                = output_path + '\\ids'
output_color_map_file         = output_ids_path + '\\colorMap.hdf5'

# Make a color map
color_map = numpy.zeros( (ncolors + 1, 3), dtype=numpy.uint8 );
for color_i in xrange( 1, ncolors + 1 ):
    rand_vals = numpy.random.rand(3);
    #color_map[ color_i ] = [ x*255 for x in colorsys.hsv_to_rgb( rand_vals[0], rand_vals[1] * 0.3 + 0.7, rand_vals[2] * 0.3 + 0.7 ) ];
    color_map[ color_i ] = [ rand_vals[0]*255, rand_vals[1]*255, rand_vals[2]*255 ];

print 'Writing colorMap file (hdf5)'

hdf5             = h5py.File( output_color_map_file, 'w' )

hdf5['idColorMap'] = color_map

hdf5.close()

