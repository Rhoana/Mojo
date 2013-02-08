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


tile_num_pixels_y             = 512
tile_num_pixels_x             = 512


original_input_ids_path       = 'C:\\dev\\datasets\\NewPipelineResults0\\labels'
output_path                   = 'C:\\dev\\datasets\\NewPipelineResults0\\mojo'
nimages_to_process            = 52
ncolors                       = 10000


##original_input_color_map_path = 'C:\\dev\\datasets\\conn\\main_dataset\\cube2\\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1\\res_from_Nov29_PF\\FS=1\\cmap2.mat'
##original_input_ids_path       = 'C:\\dev\\datasets\\conn\\main_dataset\\cube2\\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1\\res_from_Nov29_PF\\FS=1\\stitched\\labels_grow'
##output_path                    = 'C:\\dev\\datasets\\Cube2x1124\\mojo'
####nimages_to_process            = 50
##nimages_to_process            = 1124


##original_input_color_map_path = 'C:\\dev\\datasets\\conn\\main_dataset\\ac3train\\res_from_sept_30_minotrC_PF\\FS=1\\cube_coloring\\cmap.mat'
##original_input_ids_path       = 'C:\\dev\\datasets\\conn\\main_dataset\\ac3train\\res_from_sept_30_minotrC_PF\\FS=1\\stitched\\labels_grow'
##original_input_color_map_path = 'C:\\dev\\datasets\\conn\\main_dataset\\ac3train\\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1\\res_from_sept_30_minotrC_PF\\FS=1\\cube_coloring\\cmap.mat'
##original_input_ids_path       = 'C:\\dev\\datasets\\conn\\main_dataset\\ac3train\\diced_xy=512_z=32_xyOv=128_zOv=12_dwnSmp=1\\res_from_sept_30_minotrC_PF\\FS=1\\stitched\\labels_grow'
##output_path                    = 'C:\\dev\\datasets\\ac3x20\\mojo'
##nimages_to_process            = 20


output_ids_path                = output_path + '\\ids'
output_tile_ids_path           = output_ids_path + '\\tiles'

output_tile_volume_file       = output_ids_path + '\\tiledVolumeDescription.xml'
output_seg_info_file          = output_ids_path + '\\idInfo.hdf5'
output_tile_index_file        = output_ids_path + '\\idTileIndex.db'

#color_map_variable_name       = 'cmap'
ids_upscale_factor            = 1


def mkdir_safe( dir_to_make ):

    if not os.path.exists( dir_to_make ):
        execute_string = 'mkdir ' + '"' + dir_to_make + '"'
        print execute_string
        print
        os.system( execute_string )



def save_hdf5( file_path, dataset_name, array ):
    
    hdf5             = h5py.File( file_path, 'w' )
    #dataset          = hdf5.create_dataset( dataset_name, data=array, chunks=True, compression='gzip' )
    dataset          = hdf5.create_dataset( dataset_name, data=array )
    hdf5.close()

    print file_path
    print



def save_image( file_path, image ):

    image.save( file_path )
    print file_path
    print

    

#color_map_mat_dict   = scipy.io.loadmat( original_input_color_map_path )
#id_color_map         = color_map_mat_dict[ 'cmap' ]
files                = sorted( glob.glob( original_input_ids_path + '\\*.tif' ) )
id_max               = 0;
id_counts            = numpy.zeros( 0, dtype=numpy.uint32 );
id_tile_list         = [];
tile_index_z         = 0

# Make a color index
#id_label = id_color_map[ :, 0 ] + id_color_map[ :, 1 ] * 2**8 + id_color_map[ :, 2 ] * 2**16
#id_label_index = numpy.zeros( numpy.max(id_label) + 1, dtype=int )
#id_label_index[ id_label ] = range(len(id_label))

# Make a color map
color_map = numpy.zeros( (ncolors + 1, 3), dtype=numpy.uint8 );
for color_i in xrange( 1, ncolors + 1 ):
    rand_vals = numpy.random.rand(3);
    #color_map[ color_i ] = [ x*255 for x in colorsys.hsv_to_rgb( rand_vals[0], rand_vals[1] * 0.5 + 0.5, rand_vals[2] * 0.5 + 0.5 ) ];
    color_map[ color_i ] = [ rand_vals[0]*255, rand_vals[1]*255, rand_vals[2]*255 ];

for file in files:

    original_input_ids_name = file

##    ids_mat_dict            = scipy.io.loadmat( original_input_ids_name )
##    ids_raw                 = ids_mat_dict[ ids_variable_name ]
##    
##    original_ids            = numpy.kron( ids_raw, numpy.ones( ( ids_upscale_factor, ids_upscale_factor ) ) ).astype( numpy.uint32 ).copy()

    original_ids = numpy.transpose( numpy.int32( numpy.array( mahotas.imread( original_input_ids_name ) ) ) )
    
    #original_colors = numpy.array(PIL.Image.open( original_input_ids_name ))
    #original_labels = original_colors[ :, :, 0 ] + original_colors[ :, :, 1 ] * 2**8 + original_colors[ :, :, 2 ] * 2**16
    #original_ids = id_label_index[ original_labels ]

    current_image_counts = numpy.bincount( original_ids.flatten() )
    current_image_counts_ids = numpy.nonzero( current_image_counts )[0]
    current_max = numpy.max( current_image_counts_ids )
    
    if id_max < current_max:
        id_max = current_max;
        id_counts.resize( id_max + 1 );
        
    id_counts[ current_image_counts_ids ] = id_counts[ current_image_counts_ids ] + numpy.uint32( current_image_counts [ current_image_counts_ids ] )
    
    ( original_image_num_pixels_x, original_image_num_pixels_y ) = original_ids.shape

    current_image_num_pixels_y = original_image_num_pixels_y
    current_image_num_pixels_x = original_image_num_pixels_x
    current_tile_data_space_y  = tile_num_pixels_y
    current_tile_data_space_x  = tile_num_pixels_x
    tile_index_w               = 0
    ids_stride                 = 1
    
    while current_image_num_pixels_y > tile_num_pixels_y / 2 or current_image_num_pixels_x > tile_num_pixels_x / 2:

        #current_pyramid_ids_path = output_pyramid_ids_path  + '\\' + 'w=' + '%08d' % ( tile_index_w )
        #current_pyramid_ids_name = current_pyramid_ids_path + '\\' + 'z=' + '%08d' % ( tile_index_z ) + '.hdf5'
        current_tile_ids_path    = output_tile_ids_path     + '\\' + 'w=' + '%08d' % ( tile_index_w ) + '\\' + 'z=' + '%08d' % ( tile_index_z )
    
        #current_pyramid_colors_path = output_pyramid_colors_path  + '\\' + 'w=' + '%08d' % ( tile_index_w )
        #current_pyramid_colors_name = current_pyramid_colors_path + '\\' + 'z=' + '%08d' % ( tile_index_z ) + '.png'
        #current_tile_colors_path    = output_tile_colors_path     + '\\' + 'w=' + '%08d' % ( tile_index_w ) + '\\' + 'z=' + '%08d' % ( tile_index_z )

        mkdir_safe( current_tile_ids_path )
        #mkdir_safe( current_pyramid_ids_path )
        #mkdir_safe( current_tile_colors_path )
        #mkdir_safe( current_pyramid_colors_path )

        current_ids = original_ids[ ::ids_stride, ::ids_stride ]
        #save_hdf5( current_pyramid_ids_name, 'IdMap', current_ids )
                    
        #current_colors       = id_color_map[ current_ids ]
        #current_colors_image = PIL.Image.fromarray( current_colors )
        #save_image( current_pyramid_colors_name, current_colors_image )
        
        num_tiles_y = int( math.ceil( float( current_image_num_pixels_y ) / tile_num_pixels_y ) )
        num_tiles_x = int( math.ceil( float( current_image_num_pixels_x ) / tile_num_pixels_x ) )

        for tile_index_y in range( num_tiles_y ):
            for tile_index_x in range( num_tiles_x ):

                y = tile_index_y * tile_num_pixels_y
                x = tile_index_x * tile_num_pixels_x
                
                current_tile_ids_name    = current_tile_ids_path    + '\\' + 'y=' + '%08d' % ( tile_index_y ) + ','  + 'x=' + '%08d' % ( tile_index_x ) + '.hdf5'
                #current_tile_colors_name = current_tile_colors_path + '\\' + 'y=' + '%08d' % ( tile_index_y ) + ','  + 'x=' + '%08d' % ( tile_index_x ) + '.png'

                tile_ids                                                                   = numpy.zeros( ( tile_num_pixels_y, tile_num_pixels_x ), numpy.uint32 )
                tile_ids_non_padded                                                        = current_ids[ y : y + tile_num_pixels_y, x : x + tile_num_pixels_x ]
                tile_ids[ 0:tile_ids_non_padded.shape[0], 0:tile_ids_non_padded.shape[1] ] = tile_ids_non_padded[:,:]
                save_hdf5( current_tile_ids_name, 'IdMap', tile_ids )

                #tile_colors       = id_color_map[ tile_ids ]
                #tile_colors_image = PIL.Image.fromarray( tile_colors )
                #save_image( current_tile_colors_name, tile_colors_image )

                #tile_index      = ( tile_index_x, tile_index_y, tile_index_z, tile_index_w )                
                unique_tile_ids = numpy.unique( tile_ids )
                
                for unique_tile_id in unique_tile_ids:

                    id_tile_list.append( (unique_tile_id, tile_index_w, tile_index_z, tile_index_y, tile_index_x ) );
                    
                    #if not unique_tile_id in id_tile_map.keys():
                    #    id_tile_map[ unique_tile_id ] = []
                    #    
                    #id_tile_map[ unique_tile_id ].append( tile_index )
                        
        current_image_num_pixels_y = current_image_num_pixels_y / 2
        current_image_num_pixels_x = current_image_num_pixels_x / 2
        current_tile_data_space_y  = current_tile_data_space_y  * 2
        current_tile_data_space_x  = current_tile_data_space_x  * 2
        tile_index_w               = tile_index_w               + 1
        ids_stride                 = ids_stride                 * 2
        
    tile_index_z = tile_index_z + 1



    if tile_index_z >= nimages_to_process:
        break


## Sort the tile list so that the same id appears together
id_tile_list = numpy.array( sorted( id_tile_list ), numpy.uint32 )

max_id = numpy.max( [ id_tile_list[ 0, -1 ], id_counts.shape[0] - 1 ] )
print 'Got id max of:'
print id_tile_list[ -1, 0 ]
print id_counts.shape[0] - 1

## Write all segment info to a single file
    
hdf5             = h5py.File( output_seg_info_file, 'w' )

print 'Writing idInfo file (hdf5)'

hdf5['idMax'] = numpy.uint32( max_id );
hdf5['idColorMap'] = color_map
hdf5['idVoxelCountMap'] = id_counts

hdf5.close()

print 'Writing idTileIndex file (sqlite)'

con = sqlite3.connect(output_tile_index_file)

cur = con.cursor()

cur.execute('DROP TABLE IF EXISTS idTileIndex')
cur.execute('CREATE TABLE idTileIndex (id int, w int, z int, y int, x int)')
cur.execute('CREATE INDEX I_idTileIndex ON idTileIndex (id)')

for entry_index in xrange(0, id_tile_list.shape[0]):
    cur.execute("INSERT INTO idTileIndex VALUES({0}, {1}, {2}, {3}, {4})".format(*id_tile_list[entry_index, :]))

con.commit()

con.close()

#Output TiledVolumeDescription xml file

print 'Writing TiledVolumeDescription file'

tiledVolumeDescription = lxml.etree.Element( "tiledVolumeDescription",
    fileExtension = "hdf5",
    numTilesX = str( int( math.ceil( original_image_num_pixels_x / tile_num_pixels_x ) ) ),
    numTilesY = str( int( math.ceil( original_image_num_pixels_y / tile_num_pixels_y ) ) ),
    numTilesZ = str( tile_index_z ),
    numTilesW = str( tile_index_w ),
    numVoxelsPerTileX = str( tile_num_pixels_x ),
    numVoxelsPerTileY = str( tile_num_pixels_y ),
    numVoxelsPerTileZ = str( 1 ),
    numVoxelsX = str( original_image_num_pixels_x ),
    numVoxelsY = str( original_image_num_pixels_y ),
    numVoxelsZ = str( tile_index_z ),
    dxgiFormat = 'R32_UInt',
    numBytesPerVoxel = str( 4 ),      
    isSigned = str( False ).lower() )
    
with open( output_tile_volume_file, 'w' ) as file:
    file.write( lxml.etree.tostring( tiledVolumeDescription, pretty_print = True ) )
