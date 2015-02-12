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
import sqlite3
import colorsys

tile_num_pixels_y             = 512
tile_num_pixels_x             = 512

generate_memorable_names      = True
compress_ids                  = True

original_input_ids_path       = r'C:\dev\\datasets\conn\main_dataset\ac3train\output_labels'
output_path                   = r'C:\dev\datasets\ac3x20\mojo'
nimages_to_process            = 20
start_at_z                    = 0

ncolors                       = 10000
#input_file_format             = 'tif'
input_file_format             = 'png'
crop_size                     = None

output_ids_path                = output_path + '\\ids'
output_tile_ids_path           = output_ids_path + '\\tiles'

output_tile_volume_file       = output_ids_path + '\\tiledVolumeDescription.xml'
output_color_map_file         = output_ids_path + '\\colorMap.hdf5'
output_segment_info_db_file   = output_ids_path + '\\segmentInfo.db'

def mkdir_safe( dir_to_make ):

    if not os.path.exists( dir_to_make ):
        os.makedirs(dir_to_make)
        # execute_string = 'mkdir ' + '"' + dir_to_make + '"'
        # print execute_string
        # print
        # os.system( execute_string )

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

def load_id_image ( file_path ):

    ids = np.int32( np.array( mahotas.imread( file_path ) ) )

    if len(ids.shape) == 3 and ids.shape[2] == 3:
        # Assume VAST-format png (Red = most significant byte, Blue = least significant byte)
        ids = ids[:,:,0] * 2**16 + ids[:,:,1] * 2**8 + ids[:,:,2]
    elif len(ids.shape) == 3 and ids.shape[2] == 4:
        # Assume VAST-format png with alpha channel as most-significant byte
        ids = ids[ :, :, 0 ] * 2**16 + ids[ :, :, 1 ] * 2**8 + ids[ :, :, 2 ] + ids[ :, :, 3 ] * 2**24
    else:
        # Read old pipeline format
        #ids = ids.transpose() - 1
        ids = np.rot90(ids, 3)

    if crop_size is not None:
        ids = ids[0:crop_size[0], 0:crop_size[1]]
        
    return ids

def sbdm_string_hash( in_string ):
    hash = 0
    for i in xrange(len(in_string)):
        hash = ord(in_string[i]) + (hash << 6) + (hash << 16) - hash
    return np.uint32(hash % 2**32)
    
input_search_string  = original_input_ids_path + '\\*.' + input_file_format
files                = sorted( glob.glob( input_search_string ) )
print "Found {0} input images in {1}".format( len(files), input_search_string )
files = files[start_at_z:]

if len(files) > 0:

    #Only load names if there is something to name
    if generate_memorable_names:
        print 'Loading words for memorable name generation.'
        import nltk
        import random
        from nltk.corpus import wordnet
        
        # Seed based on input path so that names will be the same for multiple volumes
        random.seed( sbdm_string_hash( original_input_ids_path ) )

        nouns, verbs, adjectives, adverbs = [list(wordnet.all_synsets(pos=POS)) for POS in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]]
        nouns_verbs = nouns + verbs
        adjectives_adverbs = adjectives + adverbs

        nouns_verbs = [x for x in nouns_verbs if not ( '_' in x.lemmas[0].name or '-' in x.lemmas[0].name )]
        adjectives_adverbs = [x for x in adjectives_adverbs if not ( '_' in x.lemmas[0].name or '-' in x.lemmas[0].name )]
        
        def make_memorable_name():

            while True:
                word1 = random.choice(random.choice(adjectives_adverbs).lemmas).name
                #ignore hyphenated words
                if not ('_' in word1 or '-' in word1):
                    break

            while True:
                word2 = random.choice(random.choice(nouns_verbs).lemmas).name
                #ignore hyphenated words
                if not ('_' in word2 or '-' in word2):
                    break

            return word1.capitalize() + word2.capitalize()

    else:
        print 'Using boring names.'


    id_max               = 0;
    id_counts            = np.zeros( 0, dtype=np.int64 );
    id_tile_list         = [];
    tile_index_z         = 0

    # Make a color map
    color_map = np.zeros( (ncolors + 1, 3), dtype=np.uint8 );
    for color_i in xrange( 1, ncolors + 1 ):
        rand_vals = np.random.rand(3);
        color_map[ color_i ] = [ rand_vals[0]*255, rand_vals[1]*255, rand_vals[2]*255 ];

    # Make a compressed id map
    if compress_ids:
        print "Compressing ids..."
        compressed_id_map = np.zeros( 1, dtype=np.uint32 );
        # Read all files so that ids will be consistent across for multiple volumes
        for imgfile in files:
            original_input_ids_name = imgfile
            original_ids = load_id_image( original_input_ids_name )

            unique_ids = np.unique( original_ids )
            max_unique = np.max( unique_ids )

            if max_unique >= compressed_id_map.shape[0]:
                compressed_id_map.resize( max_unique + 1 )

            compressed_id_map[ unique_ids ] = 1

            print "Read file {0}. Max id = {1}.".format(imgfile, compressed_id_map.shape[0])

        compressed_id_map[ 0 ] = 0
        valid_ids = np.nonzero(compressed_id_map)[0]
        compressed_id_map[ valid_ids ] = np.arange(1, len(valid_ids) + 1, dtype=np.uint32)

        print "Compressing {0} ids down to {1}.".format(compressed_id_map.shape[0], len(valid_ids))

    for imgfile in files:

        original_input_ids_name = imgfile

        original_ids = load_id_image( original_input_ids_name )

        ## Grow regions until there are no boundaries

        ## Method 4 - watershed
        original_ids = mahotas.cwatershed(np.zeros(original_ids.shape, dtype=np.uint32), original_ids, return_lines=False)

        if compress_ids:
            original_ids = compressed_id_map[original_ids]

        current_image_counts = np.bincount( original_ids.ravel() )
        current_image_counts_ids = np.nonzero( current_image_counts )[0]
        current_max = np.max( current_image_counts_ids )
        
        if id_max < current_max:
            id_max = current_max;
            id_counts.resize( id_max + 1 );
            
        id_counts[ current_image_counts_ids ] = id_counts[ current_image_counts_ids ] + np.int64( current_image_counts [ current_image_counts_ids ] )
        
        # Note - size returns reverse order to shape
        ( original_image_num_pixels_y, original_image_num_pixels_x ) = original_ids.shape

        current_image_num_pixels_y = original_image_num_pixels_y
        current_image_num_pixels_x = original_image_num_pixels_x
        current_tile_data_space_y  = tile_num_pixels_y
        current_tile_data_space_x  = tile_num_pixels_x
        tile_index_w               = 0
        ids_stride                 = 1
        
        while current_image_num_pixels_y > tile_num_pixels_y / 2 or current_image_num_pixels_x > tile_num_pixels_x / 2:

            current_tile_ids_path    = output_tile_ids_path     + '\\' + 'w=' + '%08d' % ( tile_index_w ) + '\\' + 'z=' + '%08d' % ( tile_index_z )
        
            mkdir_safe( current_tile_ids_path )

            current_ids = original_ids[ ::ids_stride, ::ids_stride ]
            
            num_tiles_y = int( math.ceil( float( current_image_num_pixels_y ) / tile_num_pixels_y ) )
            num_tiles_x = int( math.ceil( float( current_image_num_pixels_x ) / tile_num_pixels_x ) )

            for tile_index_y in range( num_tiles_y ):
                for tile_index_x in range( num_tiles_x ):

                    y = tile_index_y * tile_num_pixels_y
                    x = tile_index_x * tile_num_pixels_x
                    
                    current_tile_ids_name    = current_tile_ids_path    + '\\' + 'y=' + '%08d' % ( tile_index_y ) + ','  + 'x=' + '%08d' % ( tile_index_x ) + '.hdf5'

                    tile_ids                                                                   = np.zeros( ( tile_num_pixels_y, tile_num_pixels_x ), np.uint32 )
                    tile_ids_non_padded                                                        = current_ids[ y : y + tile_num_pixels_y, x : x + tile_num_pixels_x ]
                    tile_ids[ 0:tile_ids_non_padded.shape[0], 0:tile_ids_non_padded.shape[1] ] = tile_ids_non_padded[:,:]
                    save_hdf5( current_tile_ids_name, 'IdMap', tile_ids )

                    unique_tile_ids = np.unique( tile_ids )
                    
                    for unique_tile_id in unique_tile_ids:

                        id_tile_list.append( (unique_tile_id, tile_index_w, tile_index_z, tile_index_y, tile_index_x ) );
                                                    
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
    id_tile_list = np.array( sorted( id_tile_list ), np.uint32 )

    ## Write all segment info to a single file

    print 'Writing colorMap file (hdf5)'

    hdf5             = h5py.File( output_color_map_file, 'w' )

    hdf5['idColorMap'] = color_map

    hdf5.close()

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

    taken_names = {}

    for segment_index in xrange( 1, id_max + 1 ):
        if len( id_counts ) > segment_index and id_counts[ segment_index ] > 0:
            if segment_index == 0:
                new_name = '__boundary__'
            elif generate_memorable_names:
                new_name = make_memorable_name()
                while new_name in taken_names:
                    print 'Duplicate name - regenerating.'
                    new_name = make_memorable_name()
                taken_names[ new_name ] = 1
            else:
                new_name = "segment{0}".format( segment_index )
            cur.execute('INSERT INTO segmentInfo VALUES({0}, "{1}", {2}, {3});'.format( segment_index, new_name, id_counts[ segment_index ], 0 ))

    con.commit()

    con.close()

    #Output TiledVolumeDescription xml file

    print 'Writing TiledVolumeDescription file'

    tiledVolumeDescription = lxml.etree.Element( "tiledVolumeDescription",
        fileExtension = "hdf5",
        numTilesX = str( int( math.ceil( original_image_num_pixels_x / tile_num_pixels_x ) ) + ((original_image_num_pixels_x % tile_num_pixels_x) > 0) ),
        numTilesY = str( int( math.ceil( original_image_num_pixels_y / tile_num_pixels_y ) ) + ((original_image_num_pixels_x % tile_num_pixels_x) > 0) ),
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
        
    with open( output_tile_volume_file, 'w' ) as xmlfile:
        xmlfile.write( lxml.etree.tostring( tiledVolumeDescription, pretty_print = True ) )
