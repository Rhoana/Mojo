using System;
using System.IO;
using System.Linq;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Mojo.Interop;
using SlimDX;
using SlimDX.DXGI;

namespace Mojo
{
    public static class SegmenterImageStackLoader
    {
        private const int NUM_BYTES_PER_OPTICAL_FLOW_VOXEL = 8;

        public static ObservableDictionary<string, VolumeDescription> LoadDataset( SegmenterImageStackLoadDescription segmenterImageStackLoadDescription )
        {
            var volumeDescriptions = new ObservableDictionary< string, VolumeDescription > ();

            segmenterImageStackLoadDescription.DxgiFormat = Format.R8_UNorm;
            segmenterImageStackLoadDescription.NumBytesPerVoxel = 1;
            segmenterImageStackLoadDescription.IsSigned = false;

            LoadImages<Gray, Byte>( segmenterImageStackLoadDescription, "SourceMap", ref volumeDescriptions );
            LoadImages<Gray, Byte>( segmenterImageStackLoadDescription, "FilteredSourceMap", ref volumeDescriptions );

            segmenterImageStackLoadDescription.DxgiFormat = Format.Unknown;
            segmenterImageStackLoadDescription.NumBytesPerVoxel = 0;
            segmenterImageStackLoadDescription.IsSigned = false;

            LoadOpticalFlowForwardFiles<Gray, Byte>( segmenterImageStackLoadDescription, "OpticalFlowForwardMap", "SourceMap", ref volumeDescriptions );
            LoadOpticalFlowBackwardFiles<Gray, Byte>( segmenterImageStackLoadDescription, "OpticalFlowBackwardMap", "SourceMap", ref volumeDescriptions );    

            return volumeDescriptions;
        }

        public static ObservableDictionary<string, VolumeDescription> LoadSegmentation( SegmenterImageStackLoadDescription segmenterImageStackLoadDescription )
        {
            var volumeDescriptions = new ObservableDictionary<string, VolumeDescription> ();

            segmenterImageStackLoadDescription.DxgiFormat = Format.R8G8B8A8_UNorm;
            segmenterImageStackLoadDescription.NumBytesPerVoxel = 4;
            segmenterImageStackLoadDescription.IsSigned = false;

            LoadImages< Rgba, Byte >( segmenterImageStackLoadDescription, "ColorMap", ref volumeDescriptions );

            segmenterImageStackLoadDescription.DxgiFormat = Format.R32_UInt;
            segmenterImageStackLoadDescription.NumBytesPerVoxel = 4;
            segmenterImageStackLoadDescription.IsSigned = false;

            LoadIdMapImages( segmenterImageStackLoadDescription, "IdMap", ref volumeDescriptions );

            return volumeDescriptions;
        }

        public static void SaveSegmentation( SegmenterImageStackSaveDescription segmenterImageStackSaveDescription )
        {
            var colorMapVolumeDescription = segmenterImageStackSaveDescription.VolumeDescriptions.Get( "ColorMap" );
            var colorMapDirectory = segmenterImageStackSaveDescription.Directories.Get( "ColorMap" );

            Directory.CreateDirectory( colorMapDirectory );

            colorMapVolumeDescription.DataStream.Seek( 0, SeekOrigin.Begin );

            for ( var z = 0; z < colorMapVolumeDescription.NumVoxelsZ; z++ )
            {
                var image = new Image<Rgb, Byte>( (int)colorMapVolumeDescription.NumVoxelsX, (int)colorMapVolumeDescription.NumVoxelsY );

                for ( var y = 0; y < colorMapVolumeDescription.NumVoxelsY; y++ )
                {
                    for ( var x = 0; x < colorMapVolumeDescription.NumVoxelsX; x++ )
                    {
                        var colorMapValue = colorMapVolumeDescription.DataStream.Read<int>();
                        var r = ( colorMapValue & ( 0x0000ff << 0 ) ) >> 0;
                        var g = ( colorMapValue & ( 0x0000ff << 8 ) ) >> 8;
                        var b = ( colorMapValue & ( 0x0000ff << 16 ) ) >> 16;

                        image[ y, x ] = new Rgb( r, g, b );
                    }
                }

                image.Save( Path.Combine( colorMapDirectory, String.Format( "{0:0000}.png", z ) ) );
            }

            var idMapVolumeDescription = segmenterImageStackSaveDescription.VolumeDescriptions.Get( "IdMap" );
            var idMapDirectory = segmenterImageStackSaveDescription.Directories.Get( "IdMap" );

            Interop.SegmenterImageStackLoader.SaveIdImages( idMapVolumeDescription, idMapDirectory );
        }

        private static void LoadImages< TEmguImageStructure, TSize >( SegmenterImageStackLoadDescription segmenterImageStackLoadDescription, string volumeName, ref ObservableDictionary<string, VolumeDescription> volumeDescriptions )
            where TEmguImageStructure : struct, IColor
            where TSize : new()
        {
            //
            // load in each source image and copy to client buffer
            //
            var imageDirectory = segmenterImageStackLoadDescription.Directories.Get( volumeName );
            var imageFileNames = from fileInfo in new DirectoryInfo( imageDirectory ).GetFiles( "*.*" ) select fileInfo.Name;
            var imageFilePaths = from imageFileName in imageFileNames select Path.Combine( imageDirectory, imageFileName );

            Release.Assert( imageFilePaths.Any() );

            int width, height, depth;

            using ( var image = new Image< TEmguImageStructure, TSize >( imageFilePaths.First() ) )
            {
                width = image.Width;
                height = image.Height;
                depth = imageFilePaths.Count();
            }

            var volumeDataStream = new DataStream( width * height * depth * segmenterImageStackLoadDescription.NumBytesPerVoxel, true, true );

            foreach ( var imageFilePath in imageFilePaths )
            {
                using ( var image = new Image<TEmguImageStructure, TSize>( imageFilePath ) )
                {
                    volumeDataStream.Write( image.Bytes, 0, image.Bytes.Length );

                    Release.Assert( width == image.Width );
                    Release.Assert( height == image.Height );
                }
            }

            volumeDescriptions.Set( volumeName,
                                    new VolumeDescription
                                    {
                                        DxgiFormat = segmenterImageStackLoadDescription.DxgiFormat,
                                        DataStream = volumeDataStream,
                                        Data = volumeDataStream.DataPointer,
                                        NumBytesPerVoxel = segmenterImageStackLoadDescription.NumBytesPerVoxel,
                                        NumVoxelsX = width,
                                        NumVoxelsY = height,
                                        NumVoxelsZ = depth, 
                                        IsSigned = segmenterImageStackLoadDescription.IsSigned
                                    } );
        }

        private static void LoadIdMapImages( SegmenterImageStackLoadDescription segmenterImageStackLoadDescription, string volumeName, ref ObservableDictionary<string, VolumeDescription> volumeDescriptions )
        {
            //
            // load in each source image and copy to client buffer
            //
            var imageDirectory = segmenterImageStackLoadDescription.Directories.Get( volumeName );
            var imageFileNames = from fileInfo in new DirectoryInfo( imageDirectory ).GetFiles( "*.*" ) select fileInfo.Name;
            var imageFilePaths = from imageFileName in imageFileNames select Path.Combine( imageDirectory, imageFileName );

            Release.Assert( imageFilePaths.Any() );

            int width, height, depth;

            using ( var image = new Image<Gray, UInt16>( imageFilePaths.First() ) )
            {
                width = image.Width;
                height = image.Height;
                depth = imageFilePaths.Count();
            }

            var volumeDataStream = new DataStream( width * height * depth * segmenterImageStackLoadDescription.NumBytesPerVoxel, true, true );

            foreach ( var imageFilePath in imageFilePaths )
            {
                var tmpImagePointer = CvInvoke.cvLoadImage( imageFilePath, LOAD_IMAGE_TYPE.CV_LOAD_IMAGE_ANYDEPTH );
                using ( var tmpImage = new Image<Gray, UInt16>( width, height ) )
                {
                    Release.Assert( width == tmpImage.Width );
                    Release.Assert( height == tmpImage.Height );

                    CvInvoke.cvCopy( tmpImagePointer, tmpImage, IntPtr.Zero );

                    for ( var y = 0; y < height; y++ )
                    {
                        for ( var x = 0; x < width; x++ )
                        {
                            var idMapValue = tmpImage[ y, x ];

                            volumeDataStream.Write( (int)idMapValue.Intensity );
                        }
                    }
                }

                CvInvoke.cvReleaseImage( ref tmpImagePointer );
            }

            volumeDescriptions.Set( volumeName,
                                    new VolumeDescription
                                    {
                                        DxgiFormat = segmenterImageStackLoadDescription.DxgiFormat,
                                        DataStream = volumeDataStream,
                                        Data = volumeDataStream.DataPointer,
                                        NumBytesPerVoxel = segmenterImageStackLoadDescription.NumBytesPerVoxel,
                                        NumVoxelsX = width,
                                        NumVoxelsY = height,
                                        NumVoxelsZ = depth,
                                        IsSigned = segmenterImageStackLoadDescription.IsSigned
                                    } );
        }

        private static void LoadOpticalFlowForwardFiles< TEmguImageStructure, TSize >( SegmenterImageStackLoadDescription segmenterImageStackLoadDescription, string opticalFlowVolumeName, string sourceVolumeName, ref ObservableDictionary<string, VolumeDescription> volumeDescriptions )
            where TEmguImageStructure : struct, IColor
            where TSize : new()
        {
            //
            // load in each source image and copy to client buffer
            //
            var sourceImageDirectory = segmenterImageStackLoadDescription.Directories.Get( sourceVolumeName );
            var sourceImageFileNames = from fileInfo in new DirectoryInfo( sourceImageDirectory ).GetFiles( "*.*" ) select fileInfo.Name;
            var sourceImageFilePaths = from sourceImageFileName in sourceImageFileNames select Path.Combine( sourceImageDirectory, sourceImageFileName );

            if ( sourceImageFilePaths.Count() > 1 )
            {
                int width, height, depth;

                using ( var image = new Image< TEmguImageStructure, TSize >( sourceImageFilePaths.First() ) )
                {
                    width = image.Width;
                    height = image.Height;
                    depth = sourceImageFilePaths.Count() - 1;
                }

                var volumeDataStream = new DataStream( width * height * depth * NUM_BYTES_PER_OPTICAL_FLOW_VOXEL, true, true );

                var fromFileStems = from sourceImageFileName in sourceImageFileNames.Take( sourceImageFileNames.Count() - 1 )
                                    select new string( sourceImageFileName.Remove( sourceImageFileName.LastIndexOf( "." ) ).ToCharArray() );

                var toFileStems = from sourceImageFileName in sourceImageFileNames.Skip( 1 )
                                  select new string( sourceImageFileName.Remove( sourceImageFileName.LastIndexOf( "." ) ).ToCharArray() );

                var opticalFlowFileNames = fromFileStems.Select( ( fromFileStem, i ) => fromFileStem + "-to-" + toFileStems.ElementAt( i ) + ".raw" );
                var opticalFlowFilePaths = from opticalFlowFileName in opticalFlowFileNames
                                           select
                                               Path.Combine( segmenterImageStackLoadDescription.Directories.Get( opticalFlowVolumeName ),
                                                             opticalFlowFileName );

                foreach ( var opticalFlowFilePath in opticalFlowFilePaths )
                {
                    Release.Assert( File.Exists( opticalFlowFilePath ) );

                    using ( var opticalFlowFileStream = new FileStream( opticalFlowFilePath, FileMode.Open ) )
                    {
                        opticalFlowFileStream.CopyTo( volumeDataStream );
                    }
                }

                volumeDescriptions.Set( opticalFlowVolumeName,
                                        new VolumeDescription
                                        {
                                            DxgiFormat = Format.R32G32_Float,
                                            DataStream = volumeDataStream,
                                            Data = volumeDataStream.DataPointer,
                                            NumBytesPerVoxel = NUM_BYTES_PER_OPTICAL_FLOW_VOXEL,
                                            NumVoxelsX = width,
                                            NumVoxelsY = height,
                                            NumVoxelsZ = depth,
                                            IsSigned = true
                                        } );                
            }
        }

        private static void LoadOpticalFlowBackwardFiles< TEmguImageStructure, TSize >( SegmenterImageStackLoadDescription segmenterImageStackLoadDescription, string opticalFlowVolumeName, string sourceVolumeName, ref ObservableDictionary<string, VolumeDescription> volumeDescriptions )
            where TEmguImageStructure : struct, IColor
            where TSize : new()
        {
            //
            // load in each source image and copy to client buffer
            //
            var sourceImageDirectory = segmenterImageStackLoadDescription.Directories.Get( sourceVolumeName );
            var sourceImageFileNames = from fileInfo in new DirectoryInfo( sourceImageDirectory ).GetFiles( "*.*" ) select fileInfo.Name;
            var sourceImageFilePaths = from sourceImageFileName in sourceImageFileNames select Path.Combine( sourceImageDirectory, sourceImageFileName );

            if ( sourceImageFilePaths.Count() > 1 )
            {
                int width, height, depth;

                using ( var image = new Image< TEmguImageStructure, TSize >( sourceImageFilePaths.First() ) )
                {
                    width = image.Width;
                    height = image.Height;
                    depth = sourceImageFilePaths.Count() - 1;
                }

                var volumeDataStream = new DataStream( width * height * depth * NUM_BYTES_PER_OPTICAL_FLOW_VOXEL, true, true );

                var fromFileStems = from sourceImageFileName in sourceImageFileNames.Skip( 1 )
                                    select new string( sourceImageFileName.Remove( sourceImageFileName.LastIndexOf( "." ) ).ToCharArray() );

                var toFileStems = from sourceImageFileName in sourceImageFileNames.Take( sourceImageFileNames.Count() - 1 )
                                  select new string( sourceImageFileName.Remove( sourceImageFileName.LastIndexOf( "." ) ).ToCharArray() );

                var opticalFlowFileNames = fromFileStems.Select( ( fromFileStem, i ) => fromFileStem + "-to-" + toFileStems.ElementAt( i ) + ".raw" ); 
                var opticalFlowFilePaths = from opticalFlowFileName in opticalFlowFileNames
                                           select
                                               Path.Combine( segmenterImageStackLoadDescription.Directories.Get( opticalFlowVolumeName ),
                                                             opticalFlowFileName );

                foreach ( var opticalFlowFilePath in opticalFlowFilePaths )
                {
                    Release.Assert( File.Exists( opticalFlowFilePath ) );

                    using ( var opticalFlowFileStream = new FileStream( opticalFlowFilePath, FileMode.Open ) )
                    {
                        opticalFlowFileStream.CopyTo( volumeDataStream );
                    }
                }

                volumeDescriptions.Set( opticalFlowVolumeName,
                                        new VolumeDescription
                                        {
                                            DxgiFormat = Format.R32G32_Float,
                                            DataStream = volumeDataStream,
                                            Data = volumeDataStream.DataPointer,
                                            NumBytesPerVoxel = NUM_BYTES_PER_OPTICAL_FLOW_VOXEL,
                                            NumVoxelsX = width,
                                            NumVoxelsY = height,
                                            NumVoxelsZ = depth,
                                            IsSigned = true
                                        } );                
            }
        }
    }
}
