import os
import sys
import shutil
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
import sqlite3
import colorsys

## Process existing mojo volume and output new volume with regenerated tile index

input_mojo_path               = r'E:\CurrentIDs_9-1-2014_mojo\mojo'
input_mojo_image_path         = r'Z:\PC1_Iteration_4\MojoData\full\mojo'
output_regen_path             = r'E:\CurrentIDs_9-1-2014_regen'

min_slices_per_subvolume      = 0
overlap_slices                = 1

input_tile_images_path        = input_mojo_image_path + '\\images\\tiles'
input_tile_images_volume_file = input_mojo_image_path + '\\images\\tiledVolumeDescription.xml'

input_ids_path                = input_mojo_path + '\\ids'
input_tile_ids_path           = input_ids_path + '\\tiles'
input_tile_ids_volume_file    = input_ids_path + '\\tiledVolumeDescription.xml'
input_color_map_file          = input_ids_path + '\\colorMap.hdf5'
input_segment_info_db_file    = input_ids_path + '\\segmentInfo.db'

copy_images                   = False
copy_ids                      = False

def mkdir_safe( dir_to_make ):

    if not os.path.exists( dir_to_make ):
        os.makedirs(dir_to_make)

## Open input volume xml 

print 'Reading TiledVolumeDescription files'

with open( input_tile_images_volume_file, 'r' ) as file:
    imageTiledVolumeDescription = lxml.etree.parse(file).getroot()

with open( input_tile_ids_volume_file, 'r' ) as file:
    idTiledVolumeDescription = lxml.etree.parse(file).getroot()



## Open input volume database

print 'Reading segmentInfo file (sqlite) {0}.'.format(input_segment_info_db_file)

in_con = sqlite3.connect(input_segment_info_db_file)

cur = in_con.cursor()

# Get max segment id
cur.execute('SELECT MAX(id) FROM segmentInfo;')
id_max = cur.fetchone()[0]

segment_remap = np.arange(0, id_max + 1, dtype=np.uint32)
segment_confidence = np.zeros(id_max + 1, dtype=np.int8)
segment_names = (id_max + 1) * [ None ]

# Read in name / id table
cur.execute("SELECT id, name, confidence FROM segmentInfo;")
while True:
    segment_info_row = cur.fetchone()

    if segment_info_row == None:
        break

    segment_names[segment_info_row[0]] = segment_info_row[1]
    segment_confidence[segment_info_row[0]] = segment_info_row[2]

# Read in id remap table
cur.execute("CREATE TABLE IF NOT EXISTS relabelMap ( fromId int PRIMARY KEY, toId int);")
cur.execute("SELECT fromId, toId FROM relabelMap WHERE fromId != toId ORDER BY fromId;")
while True:
    remap_row = cur.fetchone()

    if remap_row == None:
        break

    segment_remap[remap_row[0]] = remap_row[1]

in_con.close()

original_image_num_pixels_x = int ( idTiledVolumeDescription.xpath('@numVoxelsX')[0] )
original_image_num_pixels_y = int ( idTiledVolumeDescription.xpath('@numVoxelsY')[0] )

original_image_num_tiles_z = int ( idTiledVolumeDescription.xpath('@numTilesZ')[0] )

tile_num_pixels_x = int ( idTiledVolumeDescription.xpath('@numVoxelsPerTileX')[0] )
tile_num_pixels_y = int ( idTiledVolumeDescription.xpath('@numVoxelsPerTileY')[0] )


## Calculate subvolume sizes
if min_slices_per_subvolume == 0 or original_image_num_tiles_z < min_slices_per_subvolume:
    min_slices_per_subvolume = original_image_num_tiles_z

n_subvolumes = int(math.floor(original_image_num_tiles_z / min_slices_per_subvolume))
subvolume_size = int(math.floor(original_image_num_tiles_z / n_subvolumes))
subvolume_start_indices = range(0, n_subvolumes * subvolume_size, subvolume_size)


## Loop for each subvolume
for subvolume_i in [len(subvolume_start_indices) - 1]: #range(len(subvolume_start_indices)):

    subvolume_first_z = subvolume_start_indices[subvolume_i]
    if subvolume_i == len(subvolume_start_indices) - 1:
        subvolume_last_z = original_image_num_tiles_z - 1
    else:
        subvolume_last_z = subvolume_start_indices[subvolume_i + 1] - 1 + overlap_slices

    print "Subvolume {0}: z = {1} : {2}.".format(subvolume_i, subvolume_first_z, subvolume_last_z)

    output_path                    = output_regen_path + '\\z={0:04d}-{1:04d}\\mojo'.format(subvolume_first_z, subvolume_last_z)

    output_tile_images_path        = output_path + '\\images\\tiles'
    output_tile_images_volume_file = output_path + '\\images\\tiledVolumeDescription.xml'

    output_ids_path                = output_path + '\\ids'
    output_tile_ids_path           = output_ids_path + '\\tiles'

    output_tile_ids_volume_file    = output_ids_path + '\\tiledVolumeDescription.xml'
    output_color_map_file          = output_ids_path + '\\colorMap.hdf5'
    output_segment_info_db_file    = output_ids_path + '\\segmentInfo.db'

    segment_sizes = np.zeros(id_max + 1, dtype=np.int64)
    id_tile_list         = [];

    for tile_index_z in range(subvolume_last_z - subvolume_first_z + 1):
        from_tile_index_z = subvolume_first_z + tile_index_z

        ## Copy tile images (measure segment sizes for w=0)

        current_image_num_pixels_y = original_image_num_pixels_y
        current_image_num_pixels_x = original_image_num_pixels_x
        current_tile_data_space_y  = tile_num_pixels_y
        current_tile_data_space_x  = tile_num_pixels_x
        tile_index_w               = 0

        while current_image_num_pixels_y > tile_num_pixels_y / 2 or current_image_num_pixels_x > tile_num_pixels_x / 2:

            from_tile_ids_path       = input_tile_ids_path           + '\\' + 'w=' + '%08d' % ( tile_index_w ) + '\\' + 'z=' + '%08d' % ( from_tile_index_z )
            current_tile_ids_path    = output_tile_ids_path     + '\\' + 'w=' + '%08d' % ( tile_index_w ) + '\\' + 'z=' + '%08d' % ( tile_index_z )
            if copy_ids: mkdir_safe( current_tile_ids_path )

            from_tile_images_path       = input_tile_images_path           + '\\' + 'w=' + '%08d' % ( tile_index_w ) + '\\' + 'z=' + '%08d' % ( from_tile_index_z )
            current_tile_images_path    = output_tile_images_path     + '\\' + 'w=' + '%08d' % ( tile_index_w ) + '\\' + 'z=' + '%08d' % ( tile_index_z )
            if copy_images: mkdir_safe( current_tile_images_path )

            num_tiles_y = int( math.ceil( float( current_image_num_pixels_y ) / tile_num_pixels_y ) )
            num_tiles_x = int( math.ceil( float( current_image_num_pixels_x ) / tile_num_pixels_x ) )

            for tile_index_y in range( num_tiles_y ):
                for tile_index_x in range( num_tiles_x ):

                    y = tile_index_y * tile_num_pixels_y
                    x = tile_index_x * tile_num_pixels_x
                    
                    from_tile_ids_name       = from_tile_ids_path       + '\\' + 'y=' + '%08d' % ( tile_index_y ) + ','  + 'x=' + '%08d' % ( tile_index_x ) + '.' + idTiledVolumeDescription.xpath('@fileExtension')[0]
                    current_tile_ids_name    = current_tile_ids_path    + '\\' + 'y=' + '%08d' % ( tile_index_y ) + ','  + 'x=' + '%08d' % ( tile_index_x ) + '.' + idTiledVolumeDescription.xpath('@fileExtension')[0]

                    from_tile_images_name    = from_tile_images_path       + '\\' + 'y=' + '%08d' % ( tile_index_y ) + ','  + 'x=' + '%08d' % ( tile_index_x ) + '.' + imageTiledVolumeDescription.xpath('@fileExtension')[0]
                    current_tile_images_name = current_tile_images_path    + '\\' + 'y=' + '%08d' % ( tile_index_y ) + ','  + 'x=' + '%08d' % ( tile_index_x ) + '.' + imageTiledVolumeDescription.xpath('@fileExtension')[0]

                    tile_hdf5        = h5py.File( from_tile_ids_name, 'r' )
                    tile_ids         = tile_hdf5['IdMap'][:,:]
                    tile_hdf5.close()

                    current_max = np.max(tile_ids)

                    if id_max < current_max:
                        segment_remap = np.resize(segment_remap, current_max + 1)
                        segment_remap[id_max + 1:current_max + 1] = np.arange(id_max + 1, current_max + 1)
                        id_max = current_max
                        segment_sizes.resize( id_max + 1 )

                    # Remap all ids
                    tile_changing = True
                    while tile_changing:
                        old_ids = tile_ids
                        tile_ids = segment_remap[ tile_ids ]
                        tile_changing = np.any(old_ids != tile_ids)

                    unique_tile_ids = np.unique( tile_ids )
                    for unique_tile_id in unique_tile_ids:
                        id_tile_list.append( (unique_tile_id, tile_index_w, tile_index_z, tile_index_y, tile_index_x ) );

                    if tile_index_w == 0:

                        current_image_counts = np.bincount( tile_ids.ravel() )
                        current_image_counts_ids = np.nonzero( current_image_counts )[0]
                        current_max = np.max( current_image_counts_ids )
        
                        if id_max < current_max:
                            print 'WARNING: Found id ({0}) greater than id_max ({1})!'.format(current_max, id_max)
                            id_max = current_max;
                            segment_sizes.resize( id_max + 1 )

                        segment_sizes[ current_image_counts_ids ] = segment_sizes[ current_image_counts_ids ] + np.uint32( current_image_counts [ current_image_counts_ids ] )

                    if copy_ids:
                        shutil.copyfile(from_tile_ids_name, current_tile_ids_name)
                        print current_tile_ids_name
                    if copy_images: 
                        shutil.copyfile(from_tile_images_name, current_tile_images_name)
                        print current_tile_images_name

            current_image_num_pixels_y = (current_image_num_pixels_y + 1) / 2
            current_image_num_pixels_x = (current_image_num_pixels_x + 1) / 2
            current_tile_data_space_y  = current_tile_data_space_y  * 2
            current_tile_data_space_x  = current_tile_data_space_x  * 2
            tile_index_w               = tile_index_w               + 1


    ## Sort the tile list so that the same id appears together
    id_tile_list = np.array( sorted( id_tile_list ), np.uint32 )

    ## Save subvolume xml and database files

    print 'Copying colorMap file (hdf5)'
    shutil.copyfile(input_color_map_file, output_color_map_file)

    print 'Writing segmentInfo file (sqlite)'
        
    if os.path.exists(output_segment_info_db_file):
        os.remove(output_segment_info_db_file)
        print "Deleted existing database file."

    con = sqlite3.connect(output_segment_info_db_file)

    cur = con.cursor()

    cur.execute('PRAGMA main.cache_size=10000;')
    cur.execute('PRAGMA main.locking_mode=EXCLUSIVE;')
    cur.execute('PRAGMA main.synchronous=OFF;')
    cur.execute('PRAGMA main.journal_mode=WAL;')
    cur.execute('PRAGMA count_changes=OFF;')
    cur.execute('PRAGMA main.temp_store=MEMORY;')

    cur.execute('DROP TABLE IF EXISTS idTileIndex;')
    cur.execute('CREATE TABLE idTileIndex (id int, w int, z int, y int, x int);')
    cur.execute('CREATE INDEX I_idTileIndex ON idTileIndex (id);')

    cur.execute('DROP TABLE IF EXISTS segmentInfo;')
    cur.execute('CREATE TABLE segmentInfo (id int, name text, size int, confidence int);')
    cur.execute('CREATE UNIQUE INDEX I_segmentInfo ON segmentInfo (id);')

    cur.execute('DROP TABLE IF EXISTS relabelMap;')
    cur.execute('CREATE TABLE relabelMap ( fromId int PRIMARY KEY, toId int);')

    for entry_index in xrange(0, id_tile_list.shape[0]):
        cur.execute("INSERT INTO idTileIndex VALUES({0}, {1}, {2}, {3}, {4});".format( *id_tile_list[entry_index, :] ))

    for segment_index in xrange( 1, id_max + 1 ):
        if len( segment_sizes ) > segment_index and segment_sizes[ segment_index ] > 0:

            ## Add the segment info entry
            if segment_index == 0:
                segment_name = '__boundary__'
            elif segment_names[segment_index] != None:
                segment_name = segment_names[segment_index]
            else:
                print 'WARNING: Found segment id ({0}) with no name in database!'.format(segment_index)
                segment_name = "segment{0}".format( segment_index )
            cur.execute('INSERT INTO segmentInfo VALUES({0}, "{1}", {2}, {3});'.format( segment_index, segment_name, segment_sizes[ segment_index ], segment_confidence[ segment_index ] ))

        ## Add the segment remap entry
        if len( segment_remap ) > segment_index and segment_remap[ segment_index ] != segment_index:
            cur.execute('INSERT INTO relabelMap VALUES({0}, {1});'.format( segment_index, segment_remap[ segment_index ]))

    con.commit()

    con.close()

    #Output TiledVolumeDescription xml files

    print 'Writing TiledVolumeDescription files'

    output_imageTiledVolumeDescription = lxml.etree.Element( "tiledVolumeDescription",
        fileExtension = imageTiledVolumeDescription.xpath('@fileExtension')[0],
        numTilesX = imageTiledVolumeDescription.xpath('@numTilesX')[0],
        numTilesY = imageTiledVolumeDescription.xpath('@numTilesY')[0],
        numTilesZ = str( tile_index_z + 1 ),
        numTilesW = imageTiledVolumeDescription.xpath('@numTilesW')[0],
        numVoxelsPerTileX = imageTiledVolumeDescription.xpath('@numVoxelsPerTileX')[0],
        numVoxelsPerTileY = imageTiledVolumeDescription.xpath('@numVoxelsPerTileY')[0],
        numVoxelsPerTileZ = imageTiledVolumeDescription.xpath('@numVoxelsPerTileZ')[0],
        numVoxelsX = imageTiledVolumeDescription.xpath('@numVoxelsX')[0],
        numVoxelsY = imageTiledVolumeDescription.xpath('@numVoxelsY')[0],
        numVoxelsZ = str( tile_index_z + 1 ),
        dxgiFormat = imageTiledVolumeDescription.xpath('@dxgiFormat')[0],
        numBytesPerVoxel = imageTiledVolumeDescription.xpath('@numBytesPerVoxel')[0],      
        isSigned = imageTiledVolumeDescription.xpath('@isSigned')[0] )
    
    with open( output_tile_images_volume_file, 'w' ) as file:
        file.write( lxml.etree.tostring( output_imageTiledVolumeDescription, pretty_print = True ) )

    output_idTiledVolumeDescription = lxml.etree.Element( "tiledVolumeDescription",
        fileExtension = idTiledVolumeDescription.xpath('@fileExtension')[0],
        numTilesX = idTiledVolumeDescription.xpath('@numTilesX')[0],
        numTilesY = idTiledVolumeDescription.xpath('@numTilesY')[0],
        numTilesZ = str( tile_index_z + 1 ),
        numTilesW = idTiledVolumeDescription.xpath('@numTilesW')[0],
        numVoxelsPerTileX = idTiledVolumeDescription.xpath('@numVoxelsPerTileX')[0],
        numVoxelsPerTileY = idTiledVolumeDescription.xpath('@numVoxelsPerTileY')[0],
        numVoxelsPerTileZ = idTiledVolumeDescription.xpath('@numVoxelsPerTileZ')[0],
        numVoxelsX = idTiledVolumeDescription.xpath('@numVoxelsX')[0],
        numVoxelsY = idTiledVolumeDescription.xpath('@numVoxelsY')[0],
        numVoxelsZ = str( tile_index_z + 1 ),
        dxgiFormat = idTiledVolumeDescription.xpath('@dxgiFormat')[0],
        numBytesPerVoxel = idTiledVolumeDescription.xpath('@numBytesPerVoxel')[0],      
        isSigned = idTiledVolumeDescription.xpath('@isSigned')[0] )
        
    with open( output_tile_ids_volume_file, 'w' ) as file:
        file.write( lxml.etree.tostring( output_idTiledVolumeDescription, pretty_print = True ) )

    print
    print "Subvolume {0} of {1} created.".format( subvolume_i + 1, len(subvolume_start_indices))
    print
