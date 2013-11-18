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
import csv

## Open existing volume and create a series of subvolumes
## (with the same ids / names etc)

input_mojo_path               = 'E:\\dev\\November5\\LeftMerge1e2k_Nov5'
input_mojo_ids_path           = input_mojo_path + '\\ids'

output_export_path              = 'E:\\dev\\November5\\LeftMerge1e2k_Export_cc'

def mkdir_safe( dir_to_make ):

    if not os.path.exists( dir_to_make ):
        os.makedirs( dir_to_make )
        #execute_string = 'mkdir ' + '"' + dir_to_make + '"'
        #print execute_string
        #print
        #os.system( execute_string )

def read_volume( ids_path, locked_ids_only, name_file=None, merge_file=None, target_id=None ):

    print ''
    print 'Reading volume from ' + ids_path

    print 'Reading TiledVolumeDescription files'
    input_tile_ids_path           = ids_path + '\\tiles'
    input_tile_ids_volume_file    = ids_path + '\\tiledVolumeDescription.xml'
    input_color_map_file          = ids_path + '\\colorMap.hdf5'
    input_segment_info_db_file    = ids_path + '\\segmentInfo.db'

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
    segment_names = {}

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

    # Generate a lockmask table
    segment_lockmask = np.zeros((id_max+1), dtype=np.uint32)
    # Only locked ids
    for segi in range(id_max):
        if segment_confidence[segi] > 0:
            segment_lockmask[segi] = segi

    if name_file is not None and merge_file is not None and target_id is not None:
        # Find all the target_id spines

        # no remap
        segment_remap = np.arange(0, id_max + 1, dtype=np.uint32)
        # reset confidence
        segment_confidence = np.zeros(id_max + 1, dtype=np.int8)
        # reset locks
        segment_lockmask = np.zeros((id_max + 1), dtype=np.uint32)


        # Lock all children of target_id with names containing "spine"
        with open(merge_file, 'rb') as merge_csvfile:
            mergereader = csv.reader(merge_csvfile, delimiter=' ', quotechar='"', skipinitialspace=True)
            for row in mergereader:
                parent_id = int(row[0])
                child_id = int(row[1])
                if parent_id == target_id:
                    segment_lockmask[child_id] = child_id
                    segment_confidence[child_id] = 100

        # Read names
        with open(name_file, 'rb') as csvfile:
            namereader = csv.reader(csvfile, delimiter=' ', quotechar='"', skipinitialspace=True)
            for row in namereader:
                name_id = int(row[0])
                name_name = row[24]
                if segment_lockmask[name_id] != 0:
                    segment_names[name_id] = name_name
                    print '{0}:{1}'.format(name_id, name_name)
                    if name_name.lower().find('spine') == -1:
                        segment_lockmask[name_id] = 0
                        segment_confidence[name_id] = 0
                        print ('(not a spine)')

    original_image_num_pixels_x = int ( idTiledVolumeDescription.xpath('@numVoxelsX')[0] )
    original_image_num_pixels_y = int ( idTiledVolumeDescription.xpath('@numVoxelsY')[0] )

    original_image_num_tiles_z = int ( idTiledVolumeDescription.xpath('@numTilesZ')[0] )

    ################################
    # Override for testing
    #original_image_num_tiles_z = 10
    ################################

    tile_num_pixels_x = int ( idTiledVolumeDescription.xpath('@numVoxelsPerTileX')[0] )
    tile_num_pixels_y = int ( idTiledVolumeDescription.xpath('@numVoxelsPerTileY')[0] )

    print "Volume size = {0}".format((original_image_num_pixels_x, original_image_num_pixels_y, original_image_num_tiles_z))
    full_volume = np.zeros((original_image_num_pixels_x, original_image_num_pixels_y, original_image_num_tiles_z), dtype=np.uint32)

    # Read in the locked volume and record the segment sizes

    segment_sizes = np.zeros(id_max + 1, dtype=np.int64)

    for tile_index_z in range(original_image_num_tiles_z):

        ## Copy tile images (measure segment sizes for w=0)

        current_image_num_pixels_y = original_image_num_pixels_y
        current_image_num_pixels_x = original_image_num_pixels_x
        current_tile_data_space_y  = tile_num_pixels_y
        current_tile_data_space_x  = tile_num_pixels_x
        tile_index_w               = 0

        from_tile_ids_path       = input_tile_ids_path           + '\\' + 'w=' + '%08d' % ( tile_index_w ) + '\\' + 'z=' + '%08d' % ( tile_index_z )

        num_tiles_y = int( math.ceil( float( current_image_num_pixels_y ) / tile_num_pixels_y ) )
        num_tiles_x = int( math.ceil( float( current_image_num_pixels_x ) / tile_num_pixels_x ) )

        for tile_index_y in range( num_tiles_y ):
            for tile_index_x in range( num_tiles_x ):

                y = tile_index_y * tile_num_pixels_y
                x = tile_index_x * tile_num_pixels_x
                
                from_tile_ids_name       = from_tile_ids_path       + '\\' + 'y=' + '%08d' % ( tile_index_y ) + ','  + 'x=' + '%08d' % ( tile_index_x ) + '.' + idTiledVolumeDescription.xpath('@fileExtension')[0]

                tile_hdf5        = h5py.File( from_tile_ids_name, 'r' )
                tile_ids         = tile_hdf5['IdMap'][:,:]
                tile_hdf5.close()

                # Remap all ids
                tile_changing = True
                while tile_changing:
                    old_ids = tile_ids
                    tile_ids = segment_remap[ tile_ids ]
                    tile_changing = np.any(old_ids != tile_ids)

                if tile_index_w == 0:

                    if locked_ids_only:
                        # remap all non-locked ids to zero
                        tile_ids = segment_lockmask[tile_ids]

                    full_volume[ y : y + tile_num_pixels_y, x : x + tile_num_pixels_x, tile_index_z ] = tile_ids

                    current_image_counts = np.bincount( tile_ids.ravel() )
                    current_image_counts_ids = np.nonzero( current_image_counts )[0]
                    current_max = np.max( current_image_counts_ids )
    
                    if id_max < current_max:
                        print 'WARNING: Found id ({0}) greater than id_max ({1})!'.format(current_max, id_max)
                        id_max = current_max;
                        segment_sizes.resize( id_max + 1 )

                    segment_sizes[ current_image_counts_ids ] = segment_sizes[ current_image_counts_ids ] + np.uint32( current_image_counts [ current_image_counts_ids ] )

        if tile_index_z % 10 == 0:
            print "Read z={0}".format(tile_index_z)

    return (full_volume, segment_names, segment_sizes, segment_confidence, id_max)


# Read in the locked volume
(full_volume, segment_names, segment_sizes, segment_confidence, id_max) = read_volume(input_mojo_ids_path, True)

# Remove unconnected components (3D)
for seg_i in range(1,id_max):
    if segment_sizes[seg_i] > 0:
        this_seg = full_volume == seg_i
        components, ncomp = mahotas.label(this_seg)
        print "Segment {0} has {1} component(s).".format(seg_i, ncomp)
        if ncomp > 1:
            comp_sizes = mahotas.labeled.labeled_size(components)
            max_i = np.argmax(comp_sizes[1:]) + 1
            print "Max size = {0}.".format(comp_sizes[max_i])
            invalid = np.logical_and(this_seg, components != max_i)
            full_volume[invalid] = 0

if id_max > 2**24:
    print "WARNING: id_max will not fit in 24-bits."

# Export

mkdir_safe( output_export_path )

for index_z in range(full_volume.shape[2]):

    if index_z % 10 == 0:
        print 'Exporting z={0}'.format(index_z)

    # Vast-style RGB export
    export_ids = np.zeros((full_volume.shape[0], full_volume.shape[1], 3), dtype=uint8)

    export_ids[:,:,0] = np.uint8(full_volume[:, :, index_z] // (2**16) % (2**8))
    export_ids[:,:,1] = np.uint8(full_volume[:, :, index_z] // (2**8) % (2**8))
    export_ids[:,:,2] = np.uint8(full_volume[:, :, index_z] % (2**8))

    mahotas.imsave(output_export_path + '\\''z=' + '%08d.png' % ( index_z ), export_ids)

print "Wrote export volume."
