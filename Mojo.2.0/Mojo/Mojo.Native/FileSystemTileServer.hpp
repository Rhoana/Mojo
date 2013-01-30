#pragma once

//#include "vld.h"

#include "Mojo.Core/Stl.hpp"
#include <queue>

#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/OpenCV.hpp"
#include "Mojo.Core/Boost.hpp"
#include "Mojo.Core/PrimitiveMap.hpp"
#include "Mojo.Core/VolumeDescription.hpp"
#include "Mojo.Core/Index.hpp"
#include "Mojo.Core/ForEach.hpp"

#include "ITileServer.hpp"
#include "TiledDatasetDescription.hpp"
#include "FileSystemTileCacheEntry.hpp"
#include "FileSystemUndoRedoItem.hpp"
#include "Constants.hpp"

namespace Mojo
{
namespace Native
{

class FileSystemTileServer : public ITileServer
{
public:
    FileSystemTileServer( Core::PrimitiveMap constParameters );
    virtual ~FileSystemTileServer();

    virtual void                                                  LoadTiledDataset( TiledDatasetDescription& tiledDatasetDescription );
    virtual void                                                  UnloadTiledDataset();

    virtual bool                                                  IsTiledDatasetLoaded();

    virtual void                                                  LoadSegmentation( TiledDatasetDescription& tiledDatasetDescription );
    virtual void                                                  UnloadSegmentation();

    virtual bool                                                  IsSegmentationLoaded();

    virtual void                                                  SaveSegmentation();
    virtual void                                                  SaveSegmentationAs( std::string savePath );
    virtual void                                                  AutosaveSegmentation();
    virtual void                                                  DeleteTempFiles();

    virtual int                                                   GetTileCountForId( int segId );

    virtual Core::HashMap< std::string, Core::VolumeDescription > LoadTile( int4 tileIndex );
    virtual void                                                  UnloadTile( int4 tileIndex );

    virtual void                                                  ReplaceSegmentationLabel( int oldId, int newId );
    virtual void                                                  ReplaceSegmentationLabelCurrentSlice( int oldId, int newId, float3 pointTileSpace );
    virtual void                                                  ReplaceSegmentationLabelCurrentConnectedComponent( int oldId, int newId, float3 pointTileSpace );

    virtual void                                                  DrawSplit( float3 pointTileSpace, float radius );
    virtual void                                                  DrawErase( float3 pointTileSpace, float radius );
    virtual void                                                  DrawRegionB( float3 pointTileSpace, float radius );
    virtual void                                                  DrawRegionA( float3 pointTileSpace, float radius );
    virtual void                                                  DrawRegionValue( float3 pointTileSpace, float radius, int value );
    virtual void                                                  AddSplitSource( float3 pointTileSpace );
    virtual void                                                  RemoveSplitSource();
    virtual void                                                  LoadSplitDistances( int segId );
    virtual void                                                  ResetSplitState();
    virtual void                                                  PrepForSplit( int segId, float3 pointTileSpace );
	virtual void                                                  FindBoundaryJoinPoints2D( int segId );
	virtual void                                                  FindBoundaryWithinRegion2D( int segId );
	virtual void                                                  FindBoundaryBetweenRegions2D( int segId );
    virtual int                                                   CompletePointSplit( int segId, float3 pointTileSpace );
    virtual int                                                   CompleteDrawSplit( int segId, float3 pointTileSpace );

	virtual void                                                  UndoChange();
	virtual void                                                  RedoChange();
    virtual void                                                  FlushFileSystemTileCacheChanges();
    virtual void                                                  SaveAndClearFileSystemTileCache();

    virtual marray::Marray< unsigned char >                       GetIdColorMap();

private:
    template < typename TCudaType >
    void                                                          LoadTiledDatasetInternal( TiledDatasetDescription& tiledDatasetDescription );
    void                                                          UnloadTiledDatasetInternal();

    template < typename TCudaType >
	void                                                          LoadSegmentationInternal( TiledDatasetDescription& tiledDatasetDescription );
    void                                                          UnloadSegmentationInternal();

    Core::VolumeDescription                                       LoadTileImage( int4 tileIndex, std::string imageName );
    Core::VolumeDescription                                       LoadTileHdf5( int4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName );

    bool                                                          TryLoadTileImage( int4 tileIndex, std::string imageName, Core::VolumeDescription& volumeDescription );
    bool                                                          TryLoadTileHdf5( int4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, Core::VolumeDescription& volumeDescription );

    void                                                          SaveTile( int4 tileIndex, Core::HashMap< std::string, Core::VolumeDescription >& volumeDescriptions );

    void                                                          SaveTileImage( int4 tileIndex, std::string imageName, const Core::VolumeDescription& volumeDescription );
    void                                                          SaveTileHdf5( int4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, const Core::VolumeDescription& volumeDescription );

    void                                                          UnloadTileImage( Core::VolumeDescription& volumeDescription );                                                                  
    void                                                          UnloadTileHdf5( Core::VolumeDescription& volumeDescription );

    std::string                                                   CreateTileString( int4 tileIndex );
	int4                                                          CreateTileIndex( std::string tileString );
    void                                                          ReduceCacheSize();
    void                                                          ReduceCacheSizeIfNecessary();

    bool                                                          TileContainsId ( int3 numVoxelsPerTile, int3 currentIdNumVoxels, int* currentIdVolume, int segId );

    //
    // tile loading and saving internals
    //
    template < typename TCudaType >                               
    bool                                                          TryLoadTileImageInternal( int4 tileIndex, std::string imageName, Core::VolumeDescription& volumeDescription );

    template < typename TMarrayType >                             
    bool                                                          TryLoadTileHdf5Internal( int4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, Core::VolumeDescription& volumeDescription );

    template < typename TCudaType >
    void                                                          SaveTileImageInternal( int4 tileIndex, std::string imageName, const Core::VolumeDescription& volumeDescription );

    template < typename TMarrayType >                             
    void                                                          SaveTileHdf5Internal( int4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, const Core::VolumeDescription& volumeDescription );

    void                                                          UnloadTileImageInternal( Core::VolumeDescription& volumeDescription );
    void                                                          UnloadTileHdf5Internal( Core::VolumeDescription& volumeDescription );

	void                                                          UpdateSplitTiles();
	void                                                          UpdateSplitTilesBoundingBox( int2 upperLeft, int2 lowerRight );
	void                                                          UpdateSplitTilesHover();
	void                                                          ResetSplitTiles();
	void                                                          PrepForNextUndoRedoChange();

    Core::PrimitiveMap                                            mConstParameters;
    TiledDatasetDescription                                       mTiledDatasetDescription;
    bool                                                          mIsTiledDatasetLoaded;
    bool                                                          mIsSegmentationLoaded;

    Core::HashMap < std::string, FileSystemTileCacheEntry >       mFileSystemTileCache;

    FileSystemIdIndex                                             mIdIndex;

    //
    // simple split variables
    //
    std::vector< int3 >                                           mSplitSourcePoints;
    int                                                           mSplitNPerimiters;
    int3                                                          mSplitWindowStart;
    int3                                                          mSplitWindowNTiles;
    int                                                           mSplitWindowWidth;
    int                                                           mSplitWindowHeight;
    int                                                           mSplitWindowNPix;
	int                                                           mSplitLabelCount;
    int*                                                          mSplitStepDist;
    int*                                                          mSplitResultDist;
    int*                                                          mSplitPrev;
    char*                                                         mSplitBorderTargets;
    char*                                                         mSplitSearchMask;
    char*                                                         mSplitDrawArea;
    unsigned int*                                                 mSplitResultArea;

	FileSystemUndoRedoItem                                        mUndoItem;
	FileSystemUndoRedoItem                                        mRedoItem;
};

template < typename TCudaType >
inline void FileSystemTileServer::LoadTiledDatasetInternal( TiledDatasetDescription& tiledDatasetDescription )
{
    mTiledDatasetDescription = tiledDatasetDescription;
    mIsTiledDatasetLoaded    = true;
}

template < typename TCudaType >
inline void FileSystemTileServer::LoadSegmentationInternal( TiledDatasetDescription& tiledDatasetDescription )
{
    mTiledDatasetDescription = tiledDatasetDescription;

    Core::Printf( "Loading idMaps..." );

    mIdIndex = FileSystemIdIndex( mTiledDatasetDescription.paths.Get( "IdIndex" ) );

    Core::Printf( "Loaded." );

    mTiledDatasetDescription.maxLabelId = mIdIndex.GetMaxId();

    mIsSegmentationLoaded    = true;

    //Core::Printf( "FileSystemTileServer::LoadSegmentationInternal Returning." );
}

template <>
inline bool FileSystemTileServer::TryLoadTileImageInternal< uchar1 >( int4 tileIndex, std::string imageName, Core::VolumeDescription& volumeDescription )
{
    std::string tilePath = Core::ToString(
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).imageDataDirectory, "\\",
        "w=", Core::ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", Core::ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", Core::ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", Core::ToStringZeroPad( tileIndex.x, 8 ), ".",
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).fileExtension );

    if ( boost::filesystem::exists( tilePath ) )
    {
        int3 numVoxelsPerTile              = mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).numVoxelsPerTile;
        int  numBytesPerVoxel              = mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).numBytesPerVoxel;

        volumeDescription.data             = new unsigned char[ numVoxelsPerTile.y * numVoxelsPerTile.x * numBytesPerVoxel ];
        volumeDescription.dxgiFormat       = mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).dxgiFormat;
        volumeDescription.isSigned         = mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).isSigned;
        volumeDescription.numBytesPerVoxel = mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).numBytesPerVoxel;
        volumeDescription.numVoxels        = numVoxelsPerTile;

        int flags = 0; // force greyscale
        cv::Mat tileImage = cv::imread( tilePath, flags );

        RELEASE_ASSERT( tileImage.cols        == numVoxelsPerTile.x );
        RELEASE_ASSERT( tileImage.rows        == numVoxelsPerTile.y );
        RELEASE_ASSERT( tileImage.elemSize()  == 1 );
        RELEASE_ASSERT( tileImage.elemSize1() == 1 );
        RELEASE_ASSERT( tileImage.channels()  == 1 );
        RELEASE_ASSERT( tileImage.type()      == CV_8UC1 );
        RELEASE_ASSERT( tileImage.depth()     == CV_8U );
        RELEASE_ASSERT( tileImage.isContinuous() );

        memcpy( volumeDescription.data, tileImage.ptr(), numVoxelsPerTile.y * numVoxelsPerTile.x * numBytesPerVoxel );

        return true;
    }
    else
    {
        return false;
    }
}

template <>
inline bool FileSystemTileServer::TryLoadTileImageInternal< uchar4 >( int4 tileIndex, std::string imageName, Core::VolumeDescription& volumeDescription )
{
    std::string tilePath = Core::ToString(
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).imageDataDirectory, "\\",
        "w=", Core::ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", Core::ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", Core::ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", Core::ToStringZeroPad( tileIndex.x, 8 ), ".",
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).fileExtension );

    if ( boost::filesystem::exists( tilePath ) )
    {
        int3 numVoxelsPerTile              = mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).numVoxelsPerTile;
        int  numBytesPerVoxel              = mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).numBytesPerVoxel;

        volumeDescription.data             = new unsigned char[ numVoxelsPerTile.y * numVoxelsPerTile.x * numBytesPerVoxel ];
        volumeDescription.dxgiFormat       = mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).dxgiFormat;
        volumeDescription.isSigned         = mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).isSigned;
        volumeDescription.numBytesPerVoxel = mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).numBytesPerVoxel;
        volumeDescription.numVoxels        = numVoxelsPerTile;

        cv::Mat tileImage = cv::imread( tilePath );

        std::vector<cv::Mat> tileImageChannels;
        cv::split( tileImage, tileImageChannels );

        cv::Mat tileImageR;
        cv::Mat tileImageG;
        cv::Mat tileImageB;
        cv::Mat tileImageA;

        if ( mConstParameters.Get< bool >( "SWAP_COLOR_CHANNELS" ) )
        {
            tileImageR = tileImageChannels[ 2 ];
            tileImageG = tileImageChannels[ 1 ];
            tileImageB = tileImageChannels[ 0 ];
            tileImageA = cv::Mat::zeros( tileImageR.rows, tileImageR.cols, CV_8UC1 );
        }
        else
        {
            tileImageR = tileImageChannels[ 0 ];
            tileImageG = tileImageChannels[ 1 ];
            tileImageB = tileImageChannels[ 2 ];
            tileImageA = cv::Mat::zeros( tileImageR.rows, tileImageR.cols, CV_8UC1 );
        }

        tileImageChannels.clear();

        tileImageChannels.push_back( tileImageR );
        tileImageChannels.push_back( tileImageG );
        tileImageChannels.push_back( tileImageB );
        tileImageChannels.push_back( tileImageA );

        cv::merge( tileImageChannels, tileImage );

        RELEASE_ASSERT( tileImage.cols        == numVoxelsPerTile.x );
        RELEASE_ASSERT( tileImage.rows        == numVoxelsPerTile.y );
        RELEASE_ASSERT( tileImage.elemSize()  == 4 );
        RELEASE_ASSERT( tileImage.elemSize1() == 1 );
        RELEASE_ASSERT( tileImage.channels()  == 4 );
        RELEASE_ASSERT( tileImage.type()      == CV_8UC4 );
        RELEASE_ASSERT( tileImage.depth()     == CV_8U );
        RELEASE_ASSERT( tileImage.isContinuous() );

        memcpy( volumeDescription.data, tileImage.ptr(), numVoxelsPerTile.y * numVoxelsPerTile.x * numBytesPerVoxel );

        return true;
    }
    else
    {
        return false;
    }
}

template < typename T >
inline bool FileSystemTileServer::TryLoadTileImageInternal( int4 tileIndex, std::string imageName, Core::VolumeDescription& volumeDescription )
{
    RELEASE_ASSERT( 0 );

    return false;
}

template < typename TMarrayType >
inline bool FileSystemTileServer::TryLoadTileHdf5Internal( int4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, Core::VolumeDescription& volumeDescription )
{
    std::string tilePath = Core::ToString(
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).imageDataDirectory, "\\",
        "w=", Core::ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", Core::ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", Core::ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", Core::ToStringZeroPad( tileIndex.x, 8 ), ".",
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).fileExtension );

    if ( boost::filesystem::exists( tilePath ) )
    {
        int3 numVoxelsPerTile              = mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).numVoxelsPerTile;
        int  numBytesPerVoxel              = mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).numBytesPerVoxel;

        volumeDescription.data             = new unsigned char[ numVoxelsPerTile.y * numVoxelsPerTile.x * numBytesPerVoxel ];
        volumeDescription.dxgiFormat       = mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).dxgiFormat;
        volumeDescription.isSigned         = mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).isSigned;
        volumeDescription.numBytesPerVoxel = mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).numBytesPerVoxel;
        volumeDescription.numVoxels        = numVoxelsPerTile;

		//Core::Printf( "Loading tile ", tilePath, "...");

        hid_t hdf5FileHandle = marray::hdf5::openFile( tilePath );
        marray::Marray< TMarrayType > marray;
        try
        {
            marray::hdf5::load( hdf5FileHandle, hdf5InternalDatasetName, marray );
        }
        catch (...)
        {
            Core::Printf( "Warning - error loading hdf5 tile. Attempting to reduce cache size." );
            ReduceCacheSize();
            marray::hdf5::load( hdf5FileHandle, hdf5InternalDatasetName, marray );
        }
        marray::hdf5::closeFile( hdf5FileHandle );

		//Core::Printf( "Done.");

        RELEASE_ASSERT( marray.dimension() == 2 );
        RELEASE_ASSERT( marray.shape( 0 ) == numVoxelsPerTile.y && marray.shape( 1 ) == numVoxelsPerTile.y );

        memcpy( volumeDescription.data, &marray( 0 ), numVoxelsPerTile.y * numVoxelsPerTile.x * numBytesPerVoxel );

        return true;
    }
    else
    {
        return false;
    }
}

template <>
inline void FileSystemTileServer::SaveTileImageInternal< uchar4 >( int4 tileIndex, std::string imageName, const Core::VolumeDescription& volumeDescription )
{
    std::string tilePath = Core::ToString(
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).imageDataDirectory, "\\",
        "w=", Core::ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", Core::ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", Core::ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", Core::ToStringZeroPad( tileIndex.x, 8 ), ".",
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).fileExtension );

    RELEASE_ASSERT( boost::filesystem::exists( tilePath ) );

    cv::Mat tileImage = cv::Mat::zeros( volumeDescription.numVoxels.y, volumeDescription.numVoxels.x, CV_8UC4 );

    memcpy( tileImage.ptr(), volumeDescription.data, volumeDescription.numVoxels.y * volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel );

    std::vector<cv::Mat> tileImageChannels;
    cv::split( tileImage, tileImageChannels );

    cv::Mat tileImageR;
    cv::Mat tileImageG;
    cv::Mat tileImageB;

    if ( mConstParameters.Get< bool >( "SWAP_COLOR_CHANNELS" ) )
    {
        tileImageR = tileImageChannels[ 2 ];
        tileImageG = tileImageChannels[ 1 ];
        tileImageB = tileImageChannels[ 0 ];
    }
    else
    {
        tileImageR = tileImageChannels[ 0 ];
        tileImageG = tileImageChannels[ 1 ];
        tileImageB = tileImageChannels[ 2 ];
    }

    tileImageChannels.clear();

    tileImageChannels.push_back( tileImageR );
    tileImageChannels.push_back( tileImageG );
    tileImageChannels.push_back( tileImageB );

    cv::merge( tileImageChannels, tileImage );

    RELEASE_ASSERT( tileImage.cols        == volumeDescription.numVoxels.x );
    RELEASE_ASSERT( tileImage.rows        == volumeDescription.numVoxels.y );
    RELEASE_ASSERT( tileImage.elemSize()  == 3 );
    RELEASE_ASSERT( tileImage.elemSize1() == 1 );
    RELEASE_ASSERT( tileImage.channels()  == 3 );
    RELEASE_ASSERT( tileImage.type()      == CV_8UC3 );
    RELEASE_ASSERT( tileImage.depth()     == CV_8U );
    RELEASE_ASSERT( tileImage.isContinuous() );

    cv::imwrite( tilePath, tileImage );
}

template <>
inline void FileSystemTileServer::SaveTileImageInternal< uchar1 >( int4 tileIndex, std::string imageName, const Core::VolumeDescription& volumeDescription )
{
    std::string tilePath = Core::ToString(
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).imageDataDirectory, "\\",
        "w=", Core::ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", Core::ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", Core::ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", Core::ToStringZeroPad( tileIndex.x, 8 ), ".",
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).fileExtension );

    RELEASE_ASSERT( boost::filesystem::exists( tilePath ) );

    cv::Mat tileImage = cv::Mat::zeros( volumeDescription.numVoxels.y, volumeDescription.numVoxels.x, CV_8UC1 );

    memcpy( tileImage.ptr(), volumeDescription.data, volumeDescription.numVoxels.y * volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel );

    RELEASE_ASSERT( tileImage.cols        == volumeDescription.numVoxels.x );
    RELEASE_ASSERT( tileImage.rows        == volumeDescription.numVoxels.y );
    RELEASE_ASSERT( tileImage.elemSize()  == 3 );
    RELEASE_ASSERT( tileImage.elemSize1() == 1 );
    RELEASE_ASSERT( tileImage.channels()  == 3 );
    RELEASE_ASSERT( tileImage.type()      == CV_8UC3 );
    RELEASE_ASSERT( tileImage.depth()     == CV_8U );
    RELEASE_ASSERT( tileImage.isContinuous() );

    cv::imwrite( tilePath, tileImage );
}

template < typename T >
inline void FileSystemTileServer::SaveTileImageInternal( int4 tileIndex, std::string imageName, const Core::VolumeDescription& volumeDescription )
{
    RELEASE_ASSERT( 0 );
}

template < typename TMarrayType >
inline void FileSystemTileServer::SaveTileHdf5Internal( int4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, const Core::VolumeDescription& volumeDescription )
{
    RELEASE_ASSERT( volumeDescription.numBytesPerVoxel == sizeof( TMarrayType ) );

    size_t shape[] = { volumeDescription.numVoxels.y, volumeDescription.numVoxels.x };
    marray::Marray< TMarrayType > marray( shape, shape + 2 );

    memcpy( &marray( 0 ), volumeDescription.data, volumeDescription.numVoxels.y * volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel );

    std::string tilePathString = Core::ToString(
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).imageDataDirectory, "\\",
        "w=", Core::ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", Core::ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", Core::ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", Core::ToStringZeroPad( tileIndex.x, 8 ), ".",
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).fileExtension );

    boost::filesystem::path tilePath = boost::filesystem::path( tilePathString );

    if ( !boost::filesystem::exists( tilePath ) )
    {
        boost::filesystem::create_directories( tilePath.parent_path() );

        hid_t hdf5FileHandle = marray::hdf5::createFile( tilePath.native_file_string() );
        marray::hdf5::save( hdf5FileHandle, hdf5InternalDatasetName, marray );
        marray::hdf5::closeFile( hdf5FileHandle );
    }
    else
    {
        size_t origin[]       = { 0, 0 };
        size_t shape[]        = { marray.shape( 0 ), marray.shape( 1 ) };
        hid_t  hdf5FileHandle = marray::hdf5::openFile( tilePath.native_file_string(), marray::hdf5::READ_WRITE );
        
        marray::hdf5::saveHyperslab( hdf5FileHandle, hdf5InternalDatasetName, origin, origin + 2, shape, marray );
        marray::hdf5::closeFile( hdf5FileHandle );
    }
}

}
}