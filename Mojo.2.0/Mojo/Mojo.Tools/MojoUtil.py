import os
import sys
import string
import math
import mahotas
import PIL
import PIL.Image
import numpy as np
import h5py



def mkdir_safe( dir_to_make ):

    if not os.path.exists( dir_to_make ):
        execute_string = 'mkdir ' + '"' + dir_to_make + '"'
        print execute_string
        print
        os.system( execute_string )



def load_hdf5( file_path, dataset_name ):
    
    hdf5          = h5py.File( file_path, 'r' )
    dataset       = hdf5[ dataset_name ]
    numpy_dataset = np.array( dataset ).copy()
    hdf5.close()
    print file_path
    print
    return numpy_dataset


def save_hdf5( file_path, dataset_name, array ):
    
    hdf5             = h5py.File( file_path, 'w' )
    dataset          = hdf5.create_dataset( dataset_name, data=array )
    hdf5.close()
    print file_path
    print



def save_image( file_path, image ):

    image.save( file_path )
    print file_path
    print



def load_id_image( file_path, transpose_label_images=False, subtract_one_from_label_images=False ):

    ids = np.int32( np.array( mahotas.imread( file_path ) ) )

    if len( ids.shape ) == 3:
        ids = ids[ :, :, 0 ] + ids[ :, :, 1 ] * 2**8 + ids[ :, :, 2 ] * 2**16

    if transpose_label_images:
        ids = ids.transpose()

    if subtract_one_from_label_images:
        ids = ids - 1

    return ids


    
def sbdm_string_hash( in_string ):
    hash = 0
    for i in xrange(len(in_string)):
        hash = ord(in_string[i]) + (hash << 6) + (hash << 16) - hash
    return np.uint32(hash % 2**32)
