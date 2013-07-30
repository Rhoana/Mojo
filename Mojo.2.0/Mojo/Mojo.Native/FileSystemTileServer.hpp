#pragma once

#include "Stl.hpp"

#include "Boost.hpp"
#include "PrimitiveMap.hpp"
#include "VolumeDescription.hpp"
#include "Index.hpp"
#include "ForEach.hpp"

#include "ITileServer.hpp"
#include "TiledDatasetDescription.hpp"
#include "FileSystemTileCacheEntry.hpp"
#include "FileSystemUndoRedoItem.hpp"
#include "FileSystemSplitState.hpp"
#include "FileSystemLogger.hpp"
#include "FileSystemTileServerConstants.hpp"

namespace Mojo
{
namespace Native
{

class FileSystemTileServer : public ITileServer
{
public:
    FileSystemTileServer( PrimitiveMap constParameters );
    virtual ~FileSystemTileServer();

    virtual void                                                  LoadSourceImages( TiledDatasetDescription& tiledDatasetDescription );
    virtual void                                                  UnloadSourceImages();

    virtual bool                                                  AreSourceImagesLoaded();

    virtual void                                                  LoadSegmentation( TiledDatasetDescription& tiledDatasetDescription );
    virtual void                                                  UnloadSegmentation();

    virtual bool                                                  IsSegmentationLoaded();

    virtual void                                                  SaveSegmentation();
    virtual void                                                  SaveSegmentationAs( std::string savePath );
    virtual void                                                  AutosaveSegmentation();
    virtual void                                                  DeleteTempFiles();

    virtual int                                                   GetTileCountForId( unsigned int segId );
    virtual Int3                                                  GetSegmentCentralTileLocation( unsigned int segId );
    virtual Int4                                                  GetSegmentZTileBounds( unsigned int segId, int zIndex );

    virtual HashMap< std::string, VolumeDescription >             LoadTile( Int4 tileIndex );
    virtual void                                                  UnloadTile( Int4 tileIndex );

    virtual void                                                  RemapSegmentLabels( std::set< unsigned int > fromSegId, unsigned int toSegId );
    virtual void                                                  ReplaceSegmentationLabel( unsigned int oldId, unsigned int newId );
    virtual void                                                  ReplaceSegmentationLabelCurrentSlice( unsigned int oldId, unsigned int newId, Float3 pointTileSpace );
    virtual void                                                  ReplaceSegmentationLabelCurrentConnectedComponent( unsigned int oldId, unsigned int newId, Float3 pointTileSpace );

    virtual void                                                  DrawSplit( Float3 pointTileSpace, float radius );
    virtual void                                                  DrawErase( Float3 pointTileSpace, float radius );
    virtual void                                                  DrawRegionB( Float3 pointTileSpace, float radius );
    virtual void                                                  DrawRegionA( Float3 pointTileSpace, float radius );
    virtual void                                                  DrawRegionValue( Float3 pointTileSpace, float radius, int value );

    virtual void                                                  AddSplitSource( Float3 pointTileSpace );
    virtual void                                                  RemoveSplitSource();
    virtual void                                                  LoadSplitDistances( unsigned int segId );
    virtual void                                                  ResetSplitState();
    virtual void                                                  PrepForSplit( unsigned int segId, Float3 pointTileSpace );
    virtual void                                                  FindBoundaryJoinPoints2D( unsigned int segId );
    virtual void                                                  FindBoundaryWithinRegion2D( unsigned int segId );
    virtual void                                                  FindBoundaryBetweenRegions2D( unsigned int segId );
    virtual unsigned int                                          CompletePointSplit( unsigned int segId, Float3 pointTileSpace );
    virtual unsigned int                                          CompleteDrawSplit( unsigned int segId, Float3 pointTileSpace, bool join3D, int splitStartZ );

    virtual void                                                  RecordSplitState( unsigned int segId, Float3 pointTileSpace );
    virtual void                                                  PredictSplit( unsigned int segId, Float3 pointTileSpace, float radius );

    virtual void                                                  ResetAdjustState();
    virtual void                                                  PrepForAdjust( unsigned int segId, Float3 pointTileSpace );
    virtual void                                                  CommitAdjustChange( unsigned int segId, Float3 pointTileSpace );

    virtual void                                                  ResetDrawMergeState();
    virtual void                                                  PrepForDrawMerge( Float3 pointTileSpace );
    virtual std::set< unsigned int >                              GetDrawMergeIds( Float3 pointTileSpace );
    virtual std::map< unsigned int, Float3 >                      GetDrawMergeIdsAndPoints( Float3 pointTileSpace );
    virtual unsigned int                                          CommitDrawMerge( std::set< unsigned int > mergeIds, Float3 pointTileSpace );
    virtual unsigned int                                          CommitDrawMergeCurrentSlice( Float3 pointTileSpace );
    virtual unsigned int                                          CommitDrawMergeCurrentConnectedComponent( Float3 pointTileSpace );

    virtual unsigned int                                          GetNewId();
    virtual std::list< unsigned int >                             UndoChange();
    virtual std::list< unsigned int >                             RedoChange();
    virtual void                                                  TempSaveFileSystemTileCacheChanges();
    virtual void                                                  TempSaveAndClearFileSystemTileCache();
    virtual void                                                  ClearFileSystemTileCache();
    virtual float                                                 GetCurrentOperationProgress();

    virtual marray::Marray< unsigned char >*                      GetIdColorMap();
    virtual marray::Marray< unsigned int >*                       GetLabelIdMap();
    virtual marray::Marray< unsigned char >*                      GetIdConfidenceMap();

    virtual void                                                  SortSegmentInfoById( bool reverse );
    virtual void                                                  SortSegmentInfoByName( bool reverse );
    virtual void                                                  SortSegmentInfoBySize( bool reverse );
    virtual void                                                  SortSegmentInfoByConfidence( bool reverse );
    virtual void                                                  LockSegmentLabel( unsigned int segId );
    virtual void                                                  UnlockSegmentLabel( unsigned int segId );
    virtual unsigned int                                          GetSegmentInfoCount();
    virtual unsigned int                                          GetSegmentInfoCurrentListLocation( unsigned int segId );
    virtual std::list< SegmentInfo >                              GetSegmentInfoRange( int begin, int end );
    SegmentInfo                                                   GetSegmentInfo( unsigned int segId );

    virtual FileSystemSegmentInfoManager*                         GetSegmentInfoManager();

private:
    VolumeDescription                                             LoadTileLayerFromImage( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName );
    VolumeDescription                                             LoadTileLayerFromHdf5( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, std::string hdf5InternalDatasetName );

    bool                                                          TryLoadTileLayerFromImage( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, VolumeDescription& volumeDescription );
    bool                                                          TryLoadTileLayerFromHdf5( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, std::string hdf5InternalDatasetName, VolumeDescription& volumeDescription );

    void                                                          UnloadTileLayer( VolumeDescription& volumeDescription );                                                                  

    void                                                          SaveTile( Int4 tileIndex );

    void                                                          SaveTileLayerToImage( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, const VolumeDescription& volumeDescription );
    void                                                          SaveTileLayerToHdf5( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, std::string hdf5InternalDatasetName, const VolumeDescription& volumeDescription );

    std::string                                                   CreateTileString( Int4 tileIndex );
    Int4                                                          CreateTileIndex( std::string tileString );

    void                                                          ReduceCacheSize();
    void                                                          ReduceCacheSizeIfNecessary();

    bool                                                          TileContainsId ( Int3 numVoxelsPerTile, Int3 currentIdNumVoxels, unsigned int* currentIdVolume, unsigned int segId );

    //
    // tile loading and saving internals
    //
    bool                                                          TryLoadTileLayerFromImageInternalUChar1( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, VolumeDescription& volumeDescription );
    bool                                                          TryLoadTileLayerFromImageInternalUChar4( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, VolumeDescription& volumeDescription );

    bool                                                          TryLoadTileLayerFromHdf5Internal( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, std::string hdf5InternalDatasetName, VolumeDescription& volumeDescription );

    void                                                          SaveTileLayerToImageInternalUChar1( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, const VolumeDescription& volumeDescription );
    void                                                          SaveTileLayerToImageInternalUChar4( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, const VolumeDescription& volumeDescription );

    void                                                          SaveTileLayerToHdf5Internal( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, std::string hdf5InternalDatasetName, const VolumeDescription& volumeDescription );

    void                                                          UnloadTileLayerInternal( VolumeDescription& volumeDescription );

    void                                                          UpdateOverlayTiles();
    void                                                          UpdateOverlayTilesBoundingBox( Int2 upperLeft, Int2 lowerRight );
    void                                                          ResetOverlayTiles();
    void                                                          PrepForNextUndoRedoChange();

    void                                                          StrideUpIdTileChange( Int4 numTiles, Int3 numVoxelsPerTile, Int4 tileIndex, unsigned int* data );

    PrimitiveMap                                                  mConstParameters;

    TiledDatasetDescription                                       mSourceImagesTiledDatasetDescription;
    TiledDatasetDescription                                       mSegmentationTiledDatasetDescription;

    bool                                                          mAreSourceImagesLoaded;
    bool                                                          mIsSegmentationLoaded;

    //
    // CODE QUALITY ISSUE:
    // Why is this a hashmap? It seems like this is actually implementing a straightforward page table (like the one in TileManager.cpp).
    // Instead of indexing with the string "X001Y002Z0003W004" or whatever, why don't you just index into an Marray with the index a(1, 2, 3, 4). Then
    // the device and host caching functionality could be refactored into a templetized Cache class that would facilitate code reuse. Moreover,
    // Moreover, since memory fragmentation is an issue, it would be better to have a fixed-size page table and cache allocated at startup.
    //
    HashMap < std::string, FileSystemTileCacheEntry >             mFileSystemTileCache;

    //
    // simple split variables
    //
    std::vector< Int3 >                                           mSplitSourcePoints;
    int                                                           mSplitNPerimiters;
    Int3                                                          mSplitWindowStart;
    Int3                                                          mSplitWindowNTiles;
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

    FileSystemLogger                                              mLogger;
    FileSystemSegmentInfoManager                                  mSegmentInfoManager;

    Float2                                                        mCentroid;
    unsigned int                                                  mPrevSplitId;
    int                                                           mPrevSplitZ;
    std::vector< Float2 >                                         mPrevSplitLine;
    std::vector< std::pair< Float2, int >>                        mPrevSplitCentroids;

    std::map< int, FileSystemSplitState >                         mSplitStates;

    std::deque< FileSystemUndoRedoItem >                          mUndoDeque;
    std::deque< FileSystemUndoRedoItem >                          mRedoDeque;

    FileSystemUndoRedoItem*                                       mNextUndoItem;

    float                                                         mCurrentOperationProgress;

};


}
}