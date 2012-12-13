import os
import sys
import string
import math
import PIL
import PIL.Image
import numpy
import scipy
import scipy.misc
import matplotlib
import matplotlib.pyplot
import cv
import h5py

tile_num_pixels_y                                         = 512
tile_num_pixels_x                                         = 512

pyramid_scaling                                           = 0.5
num_pyramid_levels                                        = 10 
window_size_for_computing_averages                        = 40
num_iterations_per_pyramid_level                          = 50
neighborhood_size                                         = 7
gaussian_standard_deviation                               = 1.5
optical_flow_visualization_multiplier                     = 25
optical_flow_color_map                                    = matplotlib.pyplot.imread( 'OpticalFlowColorMap.png' )

input_image_pyramid_path                                  = 'C:\\dev\\datasets\\verenaTestOutput\\mojo\\images\\pyramid'

output_optical_flow_forward_pyramid_path                  = 'C:\\dev\\datasets\\challengeCubeFirstTenSlices2\\mojo\\opticalflow\\forward\\pyramid'
output_optical_flow_forward_debug_pyramid_flow_path       = 'C:\\dev\\datasets\\challengeCubeFirstTenSlices2\\mojo\\opticalflow\\forward\\debug\\pyramid\\flow'
output_optical_flow_forward_debug_pyramid_alignment_path  = 'C:\\dev\\datasets\\challengeCubeFirstTenSlices2\\mojo\\opticalflow\\forward\\debug\\pyramid\\alignment'

output_optical_flow_forward_tiles_path                    = 'C:\\dev\\datasets\\challengeCubeFirstTenSlices2\\mojo\\opticalflow\\forward\\tiles'
output_optical_flow_forward_debug_tiles_flow_path         = 'C:\\dev\\datasets\\challengeCubeFirstTenSlices2\\mojo\\opticalflow\\forward\\debug\\tiles\\flow'
output_optical_flow_forward_debug_tiles_alignment_path    = 'C:\\dev\\datasets\\challengeCubeFirstTenSlices2\\mojo\\opticalflow\\forward\\debug\\tiles\\alignment'

output_optical_flow_backward_pyramid_path                 = 'C:\\dev\\datasets\\challengeCubeFirstTenSlices2\\mojo\\opticalflow\\backward\\pyramid'
output_optical_flow_backward_debug_pyramid_flow_path      = 'C:\\dev\\datasets\\challengeCubeFirstTenSlices2\\mojo\\opticalflow\\backward\\debug\\pyramid\\flow'
output_optical_flow_backward_debug_pyramid_alignment_path = 'C:\\dev\\datasets\\challengeCubeFirstTenSlices2\\mojo\\opticalflow\\backward\\debug\\pyramid\\alignment'

output_optical_flow_backward_tiles_path                   = 'C:\\dev\\datasets\\challengeCubeFirstTenSlices2\\mojo\\opticalflow\\backward\\tiles'
output_optical_flow_backward_debug_tiles_flow_path        = 'C:\\dev\\datasets\\challengeCubeFirstTenSlices2\\mojo\\opticalflow\\backward\\debug\\tiles\\flow'
output_optical_flow_backward_debug_tiles_alignment_path   = 'C:\\dev\\datasets\\challengeCubeFirstTenSlices2\\mojo\\opticalflow\\backward\\debug\\tiles\\alignment'



def mkdir_safe( dir_to_make ):

    if not os.path.exists( dir_to_make ):
        execute_string = 'mkdir ' + '"' + dir_to_make + '"'
        print execute_string
        print
        os.system( execute_string )



def save_hdf5( file_path, dataset_name, array ):
    
    hdf5             = h5py.File( file_path, 'w' )
    dataset          = hdf5.create_dataset( dataset_name, data=array )
    hdf5.close()

    print file_path
    print


    
def save_flow_visualization( file_name, optical_flow_image, optical_flow_color_map ):

    optical_flow_x, optical_flow_y       = tuple( numpy.rollaxis( numpy.asarray( optical_flow_image ), 2 ) )

    optical_flow_color_map_height        = optical_flow_color_map.shape[0]
    optical_flow_color_map_width         = optical_flow_color_map.shape[1]

    optical_flow_visualization_indices_y = ( ( optical_flow_y * optical_flow_visualization_multiplier ) + ( optical_flow_color_map_height / 2 ) ).astype( numpy.int32 )
    optical_flow_visualization_indices_y = numpy.clip( optical_flow_visualization_indices_y, 0, optical_flow_color_map_height - 1 )

    optical_flow_visualization_indices_x = ( ( optical_flow_x * optical_flow_visualization_multiplier ) + ( optical_flow_color_map_width  / 2 ) ).astype( numpy.int32 )
    optical_flow_visualization_indices_x = numpy.clip( optical_flow_visualization_indices_x, 0, optical_flow_color_map_width - 1 )
    
    optical_flow_visualization           = optical_flow_color_map[optical_flow_visualization_indices_y,optical_flow_visualization_indices_x]

    scipy.misc.imsave( file_name, optical_flow_visualization )
    print file_name
    print



def save_alignment_visualization( file_name, prev_image, next_image, optical_flow_image ):

    optical_flow_x, optical_flow_y   = tuple( numpy.rollaxis( numpy.asarray( optical_flow_image ), 2 ) )
    
    prev_array                       = numpy.asarray( prev_image )
    next_array                       = numpy.asarray( next_image )
    
    regular_index_x, regular_index_y = numpy.meshgrid( numpy.arange( prev_image.cols ), numpy.arange( prev_image.rows ) )
    
    warped_index_y                   = ( regular_index_y + optical_flow_y ).astype( numpy.int32 )
    warped_index_y                   = numpy.clip( warped_index_y, 0, prev_image.rows - 1 )

    warped_index_x                   = ( regular_index_x + optical_flow_x ).astype( numpy.int32 )
    warped_index_x                   = numpy.clip( warped_index_x, 0, prev_image.cols - 1 )

    prev_aligned_to_next                                 = numpy.ones_like( prev_array ) * 127
    prev_aligned_to_next[warped_index_y, warped_index_x] = prev_array[regular_index_y, regular_index_x]

    scipy.misc.imsave( file_name, prev_aligned_to_next )
    print file_name
    print


                                  
assert( os.path.isdir( input_image_pyramid_path ) )

dir_contents   = sorted( os.listdir( input_image_pyramid_path ) )
tile_indices_w = [ tile_index_w for tile_index_w in dir_contents if os.path.isdir( input_image_pyramid_path + '\\' + tile_index_w ) ]
 
for tile_index_w in tile_indices_w:
    
    input_image_pyramid_path_w                           = input_image_pyramid_path + '\\' + tile_index_w

    optical_flow_forward_pyramid_path_w                  = output_optical_flow_forward_pyramid_path                  + '\\' + tile_index_w
    optical_flow_forward_debug_pyramid_flow_path_w       = output_optical_flow_forward_debug_pyramid_flow_path       + '\\' + tile_index_w
    optical_flow_forward_debug_pyramid_alignment_path_w  = output_optical_flow_forward_debug_pyramid_alignment_path  + '\\' + tile_index_w
    optical_flow_forward_tiles_path_w                    = output_optical_flow_forward_tiles_path                    + '\\' + tile_index_w
    optical_flow_forward_debug_tiles_flow_path_w         = output_optical_flow_forward_debug_tiles_flow_path         + '\\' + tile_index_w
    optical_flow_forward_debug_tiles_alignment_path_w    = output_optical_flow_forward_debug_tiles_alignment_path    + '\\' + tile_index_w

    optical_flow_backward_pyramid_path_w                 = output_optical_flow_backward_pyramid_path                 + '\\' + tile_index_w
    optical_flow_backward_debug_pyramid_flow_path_w      = output_optical_flow_backward_debug_pyramid_flow_path      + '\\' + tile_index_w
    optical_flow_backward_debug_pyramid_alignment_path_w = output_optical_flow_backward_debug_pyramid_alignment_path + '\\' + tile_index_w
    optical_flow_backward_tiles_path_w                   = output_optical_flow_backward_tiles_path                   + '\\' + tile_index_w
    optical_flow_backward_debug_tiles_flow_path_w        = output_optical_flow_backward_debug_tiles_flow_path        + '\\' + tile_index_w
    optical_flow_backward_debug_tiles_alignment_path_w   = output_optical_flow_backward_debug_tiles_alignment_path   + '\\' + tile_index_w

    mkdir_safe( optical_flow_forward_pyramid_path_w )
    mkdir_safe( optical_flow_forward_debug_pyramid_flow_path_w )
    mkdir_safe( optical_flow_forward_debug_pyramid_alignment_path_w )
    
    mkdir_safe( optical_flow_backward_pyramid_path_w )
    mkdir_safe( optical_flow_backward_debug_pyramid_flow_path_w )
    mkdir_safe( optical_flow_backward_debug_pyramid_alignment_path_w )
        
    assert( os.path.isdir( input_image_pyramid_path_w ) )

    dir_contents                = sorted( os.listdir( input_image_pyramid_path_w ) )
    image_files_z               = [ image_file_z for image_file_z in dir_contents if os.path.isfile( input_image_pyramid_path_w + '\\' + image_file_z ) ]
    
    prev_image_files            = image_files_z[:-1]
    next_image_files            = image_files_z[1:]

    image_files_z_full_path     = [ input_image_pyramid_path_w + '\\' + image_file_z for image_file_z in image_files_z ]
    prev_image_files_full_path  = image_files_z_full_path[:-1]
    next_image_files_full_path  = image_files_z_full_path[1:]
    prev_next_image_file_tuples = zip( prev_image_files, next_image_files, prev_image_files_full_path, next_image_files_full_path )

    for ( prev_image_file, next_image_file, prev_image_file_full_path, next_image_file_full_path ) in prev_next_image_file_tuples:

        prev_tile_index_z                                   = prev_image_file[2:-4]
        next_tile_index_z                                   = next_image_file[2:-4]

        optical_flow_forward_tiles_path_wz                  = optical_flow_forward_tiles_path_w                    + '\\' + 'z=' + prev_tile_index_z + ',' + next_tile_index_z
        optical_flow_forward_debug_tiles_flow_path_wz       = optical_flow_forward_debug_tiles_flow_path_w         + '\\' + 'z=' + prev_tile_index_z + ',' + next_tile_index_z
        optical_flow_forward_debug_tiles_alignment_path_wz  = optical_flow_forward_debug_tiles_alignment_path_w    + '\\' + 'z=' + prev_tile_index_z + ',' + next_tile_index_z
        
        optical_flow_forward_pyramid_name                   = optical_flow_forward_pyramid_path_w                  + '\\' + 'z=' + prev_tile_index_z + ',' + next_tile_index_z + '.hdf5'
        optical_flow_forward_debug_pyramid_flow_name        = optical_flow_forward_debug_pyramid_flow_path_w       + '\\' + 'z=' + prev_tile_index_z + ',' + next_tile_index_z + '.png'
        optical_flow_forward_debug_pyramid_alignment_name   = optical_flow_forward_debug_pyramid_alignment_path_w  + '\\' + 'z=' + prev_tile_index_z + ',' + next_tile_index_z + '.png'

        optical_flow_backward_tiles_path_wz                 = optical_flow_backward_tiles_path_w                   + '\\' + 'z=' + next_tile_index_z + ',' + prev_tile_index_z
        optical_flow_backward_debug_tiles_flow_path_wz      = optical_flow_backward_debug_tiles_flow_path_w        + '\\' + 'z=' + prev_tile_index_z + ',' + next_tile_index_z
        optical_flow_backward_debug_tiles_alignment_path_wz = optical_flow_backward_debug_tiles_alignment_path_w   + '\\' + 'z=' + prev_tile_index_z + ',' + next_tile_index_z
        
        optical_flow_backward_pyramid_name                  = optical_flow_backward_pyramid_path_w                 + '\\' + 'z=' + next_tile_index_z + ',' + prev_tile_index_z + '.hdf5'
        optical_flow_backward_debug_pyramid_flow_name       = optical_flow_backward_debug_pyramid_flow_path_w      + '\\' + 'z=' + next_tile_index_z + ',' + prev_tile_index_z + '.png'
        optical_flow_backward_debug_pyramid_alignment_name  = optical_flow_backward_debug_pyramid_alignment_path_w + '\\' + 'z=' + next_tile_index_z + ',' + prev_tile_index_z + '.png'

        mkdir_safe( optical_flow_forward_tiles_path_wz )
        mkdir_safe( optical_flow_forward_debug_tiles_flow_path_wz )
        mkdir_safe( optical_flow_forward_debug_tiles_alignment_path_wz )
        mkdir_safe( optical_flow_backward_tiles_path_wz )
        mkdir_safe( optical_flow_backward_debug_tiles_flow_path_wz )
        mkdir_safe( optical_flow_backward_debug_tiles_alignment_path_wz )
        
        prev_image                  = cv.LoadImageM( prev_image_file_full_path, False )
        next_image                  = cv.LoadImageM( next_image_file_full_path, False )
        optical_flow_forward_image  = cv.CreateMat( prev_image.rows, prev_image.cols, cv.CV_32FC2 )
        optical_flow_backward_image = cv.CreateMat( prev_image.rows, prev_image.cols, cv.CV_32FC2 )

        cv.CalcOpticalFlowFarneback(
            prev_image,
            next_image,
            optical_flow_forward_image,
            pyramid_scaling,
            num_pyramid_levels,
            window_size_for_computing_averages,
            num_iterations_per_pyramid_level,
            neighborhood_size,
            gaussian_standard_deviation,
            cv.OPTFLOW_FARNEBACK_GAUSSIAN )

        cv.CalcOpticalFlowFarneback(
            next_image,
            prev_image,
            optical_flow_backward_image,
            pyramid_scaling,
            num_pyramid_levels,
            window_size_for_computing_averages,
            num_iterations_per_pyramid_level,
            neighborhood_size,
            gaussian_standard_deviation,
            cv.OPTFLOW_FARNEBACK_GAUSSIAN )

        save_hdf5( optical_flow_forward_pyramid_name, 'optical_flow_forward', numpy.asarray( optical_flow_forward_image ) )
        save_flow_visualization( optical_flow_forward_debug_pyramid_flow_name, optical_flow_forward_image, optical_flow_color_map )
        save_alignment_visualization( optical_flow_forward_debug_pyramid_alignment_name, prev_image, next_image, optical_flow_forward_image )

        save_hdf5( optical_flow_backward_pyramid_name, 'optical_flow_backward', numpy.asarray( optical_flow_backward_image ) )
        save_flow_visualization( optical_flow_backward_debug_pyramid_flow_name, optical_flow_backward_image, optical_flow_color_map )
        save_alignment_visualization( optical_flow_backward_debug_pyramid_alignment_name, next_image, prev_image, optical_flow_backward_image )

        num_tiles_y = int( math.ceil( float( prev_image.rows ) / tile_num_pixels_y ) )
        num_tiles_x = int( math.ceil( float( prev_image.cols ) / tile_num_pixels_x ) )
        
        for tile_index_y in range( num_tiles_y ):
            for tile_index_x in range( num_tiles_x ):

                y = tile_index_y * tile_num_pixels_y
                x = tile_index_x * tile_num_pixels_x

                optical_flow_forward_tile_name                  = optical_flow_forward_tiles_path_wz                  + '\\' + 'y=' + '%08d' % ( tile_index_y ) + ','  + 'x=' + '%08d' % ( tile_index_x ) + '.hdf5'
                optical_flow_forward_tile_debug_flow_name       = optical_flow_forward_debug_tiles_flow_path_wz       + '\\' + 'y=' + '%08d' % ( tile_index_y ) + ','  + 'x=' + '%08d' % ( tile_index_x ) + '.png'
                optical_flow_forward_tile_debug_alignment_name  = optical_flow_forward_debug_tiles_alignment_path_wz  + '\\' + 'y=' + '%08d' % ( tile_index_y ) + ','  + 'x=' + '%08d' % ( tile_index_x ) + '.png'

                optical_flow_backward_tile_name                 = optical_flow_backward_tiles_path_wz                 + '\\' + 'y=' + '%08d' % ( tile_index_y ) + ','  + 'x=' + '%08d' % ( tile_index_x ) + '.hdf5'
                optical_flow_backward_tile_debug_flow_name      = optical_flow_backward_debug_tiles_flow_path_wz      + '\\' + 'y=' + '%08d' % ( tile_index_y ) + ','  + 'x=' + '%08d' % ( tile_index_x ) + '.png'
                optical_flow_backward_tile_debug_alignment_name = optical_flow_backward_debug_tiles_alignment_path_wz + '\\' + 'y=' + '%08d' % ( tile_index_y ) + ','  + 'x=' + '%08d' % ( tile_index_x ) + '.png'

                prev_tile_image                  = cv.GetSubRect( prev_image,                  ( x, y, tile_num_pixels_x, tile_num_pixels_y ) )
                next_tile_image                  = cv.GetSubRect( next_image,                  ( x, y, tile_num_pixels_x, tile_num_pixels_y ) )
                optical_flow_forward_tile_image  = cv.GetSubRect( optical_flow_forward_image,  ( x, y, tile_num_pixels_x, tile_num_pixels_y ) )
                optical_flow_backward_tile_image = cv.GetSubRect( optical_flow_backward_image, ( x, y, tile_num_pixels_x, tile_num_pixels_y ) )
                
                save_hdf5( optical_flow_forward_tile_name, 'optical_flow_forward', numpy.asarray( optical_flow_forward_tile_image ) )
                save_flow_visualization( optical_flow_forward_tile_debug_flow_name, optical_flow_forward_tile_image, optical_flow_color_map )
                save_alignment_visualization( optical_flow_forward_tile_debug_alignment_name, prev_tile_image, next_tile_image, optical_flow_forward_tile_image )

                save_hdf5( optical_flow_backward_tile_name, 'optical_flow_backward', numpy.asarray( optical_flow_backward_tile_image ) )
                save_flow_visualization( optical_flow_backward_tile_debug_flow_name, optical_flow_backward_tile_image, optical_flow_color_map )
                save_alignment_visualization( optical_flow_backward_tile_debug_alignment_name, next_tile_image, prev_tile_image, optical_flow_backward_tile_image )
