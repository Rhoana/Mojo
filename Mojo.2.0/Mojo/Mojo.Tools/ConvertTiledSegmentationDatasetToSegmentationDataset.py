import os
import sys
import string
import math
import mahotas
import PIL
import PIL.Image
import numpy as np
import scipy
import scipy.io
import cv2
import h5py
import lxml
import lxml.etree
import glob
import colorsys
import MojoUtil

#
# Instead of importing sqlite3 from the standard library, we import from
# the pysqlite2 package. This is because the version of sqlite3 in the standard
# library is inconsistent across platforms and Python distributions. For example,
# on the 32-bit Windows Enthought Python Distribution, the standard library sqlite3
# is too old to read the db files that are read and written to by Mojo. On the other
# hand, the standard library sqlite3 on On Mac OSX is new enough to load Mojo db
# files. By using pysqlite2, we achieve more consistency across Python distributions. - MR
#
from pysqlite2 import dbapi2 as sqlite3



tile_num_voxels_y                           = 512
tile_num_voxels_x                           = 512

input_top_level_mojo_image_directory        = 'C:\\Users\\mike\\Data\\Local\\2013_mojo\\isbi_submission_mojo\\isbi_submission_mojoimg'
input_top_level_mojo_segmentation_directory = 'C:\\Users\\mike\\Data\\Local\\2013_mojo\\isbi_submission_edited_mojo\\isbi_submission_edited_mojoseg'
input_resolution_as_w_index                 = 0
input_tile_image_path                       = input_top_level_mojo_image_directory        + '\\images\\tiles\\w=%08d' % input_resolution_as_w_index
input_tile_id_path                          = input_top_level_mojo_segmentation_directory + '\\ids\\tiles\\w=%08d'    % input_resolution_as_w_index
input_tile_id_volume_file                   = input_top_level_mojo_segmentation_directory + '\\ids\\tiledVolumeDescription.xml'
input_color_map_file                        = input_top_level_mojo_segmentation_directory + '\\ids\\colorMap.hdf5'
input_segment_info_db_file                  = input_top_level_mojo_segmentation_directory + '\\ids\\segmentInfo.db'
input_image_file_format                     = 'tif'
input_id_file_format                        = 'hdf5'

output_image_path                           = 'C:\\Users\\mike\\Data\\Local\\2013_mojo\\isbi_submission_edited\\input_images'
output_id_path                              = 'C:\\Users\\mike\\Data\\Local\\2013_mojo\\isbi_submission_edited\\labels'
output_overlay_path                         = 'C:\\Users\\mike\\Data\\Local\\2013_mojo\\isbi_submission_edited\\overlay'
output_image_file_format                    = 'png'
output_id_file_format                       = 'hdf5'
output_overlay_file_format                  = 'png'



# Open input volume database
print 'Reading segmentInfo file (sqlite) {0}.'.format( input_segment_info_db_file )

in_con = sqlite3.connect( input_segment_info_db_file )
cur    = in_con.cursor()

# Get max segment id
cur.execute('SELECT MAX(id) FROM segmentInfo;')

id_max        = cur.fetchone()[0]
segment_remap = np.arange(0, id_max + 1, dtype=np.uint32)

# Read in id remap table
cur.execute( 'CREATE TABLE IF NOT EXISTS relabelMap ( fromId int PRIMARY KEY, toId int);' )
cur.execute( 'SELECT fromId, toId FROM relabelMap WHERE fromId != toId ORDER BY fromId;' )

while True:
    remap_row = cur.fetchone()

    if remap_row == None:
        break

    segment_remap[ remap_row[ 0 ] ] = remap_row[ 1 ]

in_con.close()



color_map = MojoUtil.load_hdf5( input_color_map_file, "idColorMap" )



MojoUtil.mkdir_safe( output_image_path )
MojoUtil.mkdir_safe( output_id_path )
MojoUtil.mkdir_safe( output_overlay_path )



input_tile_id_volume_file_xml = lxml.etree.parse( input_tile_id_volume_file )
num_tiles_y                   = int( math.ceil( int( input_tile_id_volume_file_xml.getroot().get( 'numTilesY' ) )  / pow( 2, input_resolution_as_w_index ) ) )
num_tiles_x                   = int( math.ceil( int( input_tile_id_volume_file_xml.getroot().get( 'numTilesX' ) )  / pow( 2, input_resolution_as_w_index ) ) )
num_voxels_y                  = int( math.ceil( int( input_tile_id_volume_file_xml.getroot().get( 'numVoxelsY' ) ) / pow( 2, input_resolution_as_w_index ) ) )
num_voxels_x                  = int( math.ceil( int( input_tile_id_volume_file_xml.getroot().get( 'numVoxelsX' ) ) / pow( 2, input_resolution_as_w_index ) ) )
slices_z                      = [ int( z.strip( 'z=' ) ) for z in sorted( os.listdir( input_tile_id_path ) ) ]

for slice_z in slices_z:

    stitched_image_path   = output_image_path   + '\\z=%08d' % ( slice_z ) + '.' + output_image_file_format
    stitched_id_path      = output_id_path      + '\\z=%08d' % ( slice_z ) + '.' + output_id_file_format
    stitched_overlay_path = output_overlay_path + '\\z=%08d' % ( slice_z ) + '.' + output_overlay_file_format

    stitched_image_slice   = np.zeros( ( num_voxels_y, num_voxels_x ), dtype=np.int32 )
    stitched_id_slice      = np.zeros( ( num_voxels_y, num_voxels_x ), dtype=np.int32 )
    stitched_overlay_slice = np.zeros( ( num_voxels_y, num_voxels_x ), dtype=np.int32 )

    for tile_y in range( num_tiles_y ):
        for tile_x in range( num_tiles_x ):

            current_tile_image_path = input_tile_image_path + '\\' + 'z=%08d' % ( slice_z ) + '\\' + 'y=%08d,x=%08d' % ( tile_y, tile_x ) + '.' + input_image_file_format
            current_tile_id_path    = input_tile_id_path    + '\\' + 'z=%08d' % ( slice_z ) + '\\' + 'y=%08d,x=%08d' % ( tile_y, tile_x ) + '.' + input_id_file_format
            current_tile_image      = np.array( PIL.Image.open( current_tile_image_path ) )
            current_tile_id         = MojoUtil.load_hdf5( current_tile_id_path, 'IdMap' )
            start_voxel_y           = tile_y * tile_num_voxels_y
            start_voxel_x           = tile_x * tile_num_voxels_x
            end_voxel_y             = min( ( tile_y + 1 ) * tile_num_voxels_y, num_voxels_y )
            end_voxel_x             = min( ( tile_x + 1 ) * tile_num_voxels_x, num_voxels_x )

            stitched_image_slice[ start_voxel_y:end_voxel_y, start_voxel_x:end_voxel_x ] = current_tile_image[ :,: ]
            stitched_id_slice[ start_voxel_y:end_voxel_y, start_voxel_x:end_voxel_x ]    = current_tile_id[ :,: ]

    old_ids = stitched_id_slice
    new_ids = stitched_id_slice

    slice_changing = True
    while slice_changing:
        old_ids        = new_ids
        new_ids        = segment_remap[ new_ids ]
        slice_changing = np.any( old_ids != new_ids )

    stitched_id_slice_remapped = new_ids

    stitched_id_slice_color     = color_map[ np.mod( stitched_id_slice_remapped, color_map.shape[0] ) ]
    stitched_image_slice_pil    = PIL.Image.fromarray( np.uint8( stitched_image_slice ) )
    stitched_id_slice_color_pil = PIL.Image.fromarray( stitched_id_slice_color )
    stitched_overlay_slice_pil  = PIL.Image.blend( stitched_image_slice_pil.convert( 'RGBA' ), stitched_id_slice_color_pil.convert( 'RGBA' ), 0.5 )

    stitched_image_slice_pil.save( stitched_image_path, output_image_file_format )
    MojoUtil.save_hdf5( stitched_id_path, "IdMap", stitched_id_slice_remapped )
    stitched_overlay_slice_pil.save( stitched_overlay_path, output_overlay_file_format )

    print stitched_image_path
    print stitched_id_path
    print stitched_overlay_path
    print
