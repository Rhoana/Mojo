#pragma once

//#include "vld.h"

#include "Mojo.Core/Stl.hpp"
#include <queue>

//#include "Mojo.Core/Cuda.hpp"

#include "Mojo.Core/Boost.hpp"
#include "Mojo.Core/PrimitiveMap.hpp"
#include "Mojo.Core/VolumeDescription.hpp"
#include "Mojo.Core/Index.hpp"
#include "Mojo.Core/ForEach.hpp"

#include "ITileServer.hpp"
#include "TiledDatasetDescription.hpp"
#include "FileSystemTileCacheEntry.hpp"
#include "FileSystemUndoRedoItem.hpp"
#include "FileSystemSplitState.hpp"
#include "FileSystemLogger.hpp"
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

    virtual int                                                   GetTileCountForId( unsigned int segId );
    virtual MojoInt3                                              GetSegmentCentralTileLocation( unsigned int segId );
    virtual MojoInt4                                              GetSegmentZTileBounds( unsigned int segId, int zIndex );

    virtual Core::HashMap< std::string, Core::VolumeDescription > LoadTile( MojoInt4 tileIndex );
    virtual void                                                  UnloadTile( MojoInt4 tileIndex );

    virtual void                                                  RemapSegmentLabels( std::set< unsigned int > fromSegId, unsigned int toSegId );
    virtual void                                                  ReplaceSegmentationLabel( unsigned int oldId, unsigned int newId );
    virtual void                                                  ReplaceSegmentationLabelCurrentSlice( unsigned int oldId, unsigned int newId, MojoFloat3 pointTileSpace );
    virtual void                                                  ReplaceSegmentationLabelCurrentConnectedComponent( unsigned int oldId, unsigned int newId, MojoFloat3 pointTileSpace );

    virtual void                                                  DrawSplit( MojoFloat3 pointTileSpace, float radius );
    virtual void                                                  DrawErase( MojoFloat3 pointTileSpace, float radius );
    virtual void                                                  DrawRegionB( MojoFloat3 pointTileSpace, float radius );
    virtual void                                                  DrawRegionA( MojoFloat3 pointTileSpace, float radius );
    virtual void                                                  DrawRegionValue( MojoFloat3 pointTileSpace, float radius, int value );

    virtual void                                                  AddSplitSource( MojoFloat3 pointTileSpace );
    virtual void                                                  RemoveSplitSource();
    virtual void                                                  LoadSplitDistances( unsigned int segId );
    virtual void                                                  ResetSplitState();
    virtual void                                                  PrepForSplit( unsigned int segId, MojoFloat3 pointTileSpace );
	virtual void                                                  FindBoundaryJoinPoints2D( unsigned int segId );
	virtual void                                                  FindBoundaryWithinRegion2D( unsigned int segId );
	virtual void                                                  FindBoundaryBetweenRegions2D( unsigned int segId );
    virtual unsigned int                                          CompletePointSplit( unsigned int segId, MojoFloat3 pointTileSpace );
    virtual unsigned int                                          CompleteDrawSplit( unsigned int segId, MojoFloat3 pointTileSpace, bool join3D, int splitStartZ );

    virtual void                                                  RecordSplitState( unsigned int segId, MojoFloat3 pointTileSpace );
    virtual void                                                  PredictSplit( unsigned int segId, MojoFloat3 pointTileSpace, float radius );

    virtual void                                                  ResetAdjustState();
    virtual void                                                  PrepForAdjust( unsigned int segId, MojoFloat3 pointTileSpace );
    virtual void                                                  CommitAdjustChange( unsigned int segId, MojoFloat3 pointTileSpace );

    virtual void                                                  ResetDrawMergeState();
    virtual void                                                  PrepForDrawMerge( MojoFloat3 pointTileSpace );
	virtual std::set< unsigned int >                              GetDrawMergeIds( MojoFloat3 pointTileSpace );
	virtual std::map< unsigned int, MojoFloat3 >                  GetDrawMergeIdsAndPoints( MojoFloat3 pointTileSpace );
	virtual unsigned int                                          CommitDrawMerge( std::set< unsigned int > mergeIds, MojoFloat3 pointTileSpace );
	virtual unsigned int                                          CommitDrawMergeCurrentSlice( MojoFloat3 pointTileSpace );
	virtual unsigned int                                          CommitDrawMergeCurrentConnectedComponent( MojoFloat3 pointTileSpace );

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
    virtual void                                                  SortSegmentInfoByType( bool reverse );
    virtual void                                                  SortSegmentInfoBySubType( bool reverse );
    virtual void                                                  LockSegmentLabel( unsigned int segId );
    virtual void                                                  UnlockSegmentLabel( unsigned int segId );
    virtual void                                                  SetSegmentType( unsigned int segId, std::string newType );
    virtual void                                                  SetSegmentSubType( unsigned int segId, std::string newSubType );
	virtual unsigned int                                          GetSegmentInfoCount();
	virtual unsigned int                                          GetSegmentInfoCurrentListLocation( unsigned int segId );
    virtual std::list< SegmentInfo >                              GetSegmentInfoRange( int begin, int end );
    SegmentInfo                                                   GetSegmentInfo( unsigned int segId );
    virtual std::set< unsigned int >                              GetRemappedChildren( unsigned int segId );

    virtual FileSystemSegmentInfoManager*                         GetSegmentInfoManager();

private:
    void                                                          LoadTiledDatasetInternal( TiledDatasetDescription& tiledDatasetDescription );
    void                                                          UnloadTiledDatasetInternal();

	void                                                          LoadSegmentationInternal( TiledDatasetDescription& tiledDatasetDescription );
    void                                                          UnloadSegmentationInternal();

    Core::VolumeDescription                                       LoadTileImage( MojoInt4 tileIndex, std::string imageName );
    Core::VolumeDescription                                       LoadTileHdf5( MojoInt4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName );

    bool                                                          TryLoadTileImage( MojoInt4 tileIndex, std::string imageName, Core::VolumeDescription& volumeDescription );
    bool                                                          TryLoadTileHdf5( MojoInt4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, Core::VolumeDescription& volumeDescription );

    void                                                          SaveTile( MojoInt4 tileIndex, Core::HashMap< std::string, Core::VolumeDescription >& volumeDescriptions );

    void                                                          SaveTileImage( MojoInt4 tileIndex, std::string imageName, const Core::VolumeDescription& volumeDescription );
    void                                                          SaveTileHdf5( MojoInt4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, const Core::VolumeDescription& volumeDescription );

    void                                                          UnloadTileImage( Core::VolumeDescription& volumeDescription );                                                                  
    void                                                          UnloadTileHdf5( Core::VolumeDescription& volumeDescription );

    std::string                                                   CreateTileString( MojoInt4 tileIndex );
	MojoInt4                                                      CreateTileIndex( std::string tileString );
    void                                                          ReduceCacheSize();
    void                                                          ReduceCacheSizeIfNecessary();

    bool                                                          TileContainsId ( MojoInt3 numVoxelsPerTile, MojoInt3 currentIdNumVoxels, unsigned int* currentIdVolume, unsigned int segId );

    //
    // tile loading and saving internals
    //
    bool                                                          TryLoadTileImageInternalUChar1( MojoInt4 tileIndex, std::string imageName, Core::VolumeDescription& volumeDescription );
    bool                                                          TryLoadTileImageInternalUChar4( MojoInt4 tileIndex, std::string imageName, Core::VolumeDescription& volumeDescription );

    bool                                                          TryLoadTileHdf5Internal( MojoInt4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, Core::VolumeDescription& volumeDescription );

    void                                                          SaveTileImageInternalUChar1( MojoInt4 tileIndex, std::string imageName, const Core::VolumeDescription& volumeDescription );
    void                                                          SaveTileImageInternalUChar4( MojoInt4 tileIndex, std::string imageName, const Core::VolumeDescription& volumeDescription );

    void                                                          SaveTileHdf5Internal( MojoInt4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, const Core::VolumeDescription& volumeDescription );

    void                                                          UnloadTileImageInternal( Core::VolumeDescription& volumeDescription );
    void                                                          UnloadTileHdf5Internal( Core::VolumeDescription& volumeDescription );

	void                                                          UpdateOverlayTiles();
	void                                                          UpdateOverlayTilesBoundingBox( MojoInt2 upperLeft, MojoInt2 lowerRight );
	void                                                          ResetOverlayTiles();
	void                                                          PrepForNextUndoRedoChange();

	void														  StrideUpIdTileChange( MojoInt4 numTiles, MojoInt3 numVoxelsPerTile, MojoInt4 tileIndex, unsigned int* data );

    Core::PrimitiveMap                                            mConstParameters;
    TiledDatasetDescription                                       mTiledDatasetDescription;
    bool                                                          mIsTiledDatasetLoaded;
    bool                                                          mIsSegmentationLoaded;

    Core::HashMap < std::string, FileSystemTileCacheEntry >       mFileSystemTileCache;

    //
    // simple split variables
    //
    std::vector< MojoInt3 >                                       mSplitSourcePoints;
    int                                                           mSplitNPerimiters;
    MojoInt3                                                      mSplitWindowStart;
    MojoInt3                                                      mSplitWindowNTiles;
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

    MojoFloat2                                                    mCentroid;
    unsigned int                                                  mPrevSplitId;
    int                                                           mPrevSplitZ;
    std::vector< MojoFloat2 >                                     mPrevSplitLine;
    std::vector< std::pair< MojoFloat2, int >>                    mPrevSplitCentroids;

    std::map< int, FileSystemSplitState >                         mSplitStates;

    std::deque< FileSystemUndoRedoItem >                          mUndoDeque;
	std::deque< FileSystemUndoRedoItem >                          mRedoDeque;

    FileSystemUndoRedoItem*                                       mNextUndoItem;

	float                                                         mCurrentOperationProgress;

};


}
}