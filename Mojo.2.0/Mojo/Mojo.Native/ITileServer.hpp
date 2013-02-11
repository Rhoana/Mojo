#pragma once

#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/VolumeDescription.hpp"

#include <marray/marray.hxx>

#include "TiledDatasetDescription.hpp"
#include "SegmentInfo.hpp"

namespace Mojo
{
namespace Native
{

class ITileServer
{
public:
    virtual ~ITileServer() {};

    virtual void                                                  LoadTiledDataset( TiledDatasetDescription& tiledDatasetDescription )                    = 0;
    virtual void                                                  UnloadTiledDataset()                                                                    = 0;

    virtual bool                                                  IsTiledDatasetLoaded()                                                                  = 0;

    virtual void                                                  LoadSegmentation( TiledDatasetDescription& tiledDatasetDescription )                    = 0;
    virtual void                                                  UnloadSegmentation()                                                                    = 0;

    virtual bool                                                  IsSegmentationLoaded()                                                                  = 0;

    virtual void                                                  SaveSegmentation()                                                                      = 0;
    virtual void                                                  SaveSegmentationAs( std::string savePath )                                              = 0;
    virtual void                                                  AutosaveSegmentation()                                                                  = 0;
    virtual void                                                  DeleteTempFiles()                                                                       = 0;

    virtual int                                                   GetTileCountForId( unsigned int segId )                                                          = 0;

    virtual Core::HashMap< std::string, Core::VolumeDescription > LoadTile( int4 tileIndex )                                                              = 0;
    virtual void                                                  UnloadTile( int4 tileIndex )                                                            = 0;

    virtual void                                                  ReplaceSegmentationLabel( unsigned int oldId, unsigned int newId )                                        = 0;
    virtual void                                                  ReplaceSegmentationLabelCurrentSlice( unsigned int oldId, unsigned int newId, float3 pointTileSpace )     = 0;
    virtual void                                                  ReplaceSegmentationLabelCurrentConnectedComponent( unsigned int oldId, unsigned int newId, float3 pointTileSpace ) = 0;

    virtual void                                                  DrawSplit( float3 pointTileSpace, float radius )                                        = 0;
    virtual void                                                  DrawErase( float3 pointTileSpace, float radius )                                        = 0;
    virtual void                                                  DrawRegionB( float3 pointTileSpace, float radius )                                      = 0;
    virtual void                                                  DrawRegionA( float3 pointTileSpace, float radius )                                      = 0;

    virtual void                                                  AddSplitSource( float3 pointTileSpace )                                                 = 0;
    virtual void                                                  RemoveSplitSource()                                                                     = 0;
    virtual void                                                  ResetSplitState()                                                                       = 0;
    virtual void                                                  PrepForSplit( unsigned int segId, float3 pointTileSpace )                                        = 0;
	virtual void                                                  FindBoundaryJoinPoints2D( unsigned int segId )                                                   = 0;
	virtual void                                                  FindBoundaryWithinRegion2D( unsigned int segId )                                                 = 0;
	virtual void                                                  FindBoundaryBetweenRegions2D( unsigned int segId )                                               = 0;
    virtual int                                                   CompletePointSplit( unsigned int segId, float3 pointTileSpace )                                  = 0;
    virtual int                                                   CompleteDrawSplit( unsigned int segId, float3 pointTileSpace, bool join3D, int splitStartZ )     = 0;

    virtual void                                                  RecordSplitState( unsigned int segId, float3 pointTileSpace )                                    = 0;
    virtual void                                                  PredictSplit( unsigned int segId, float3 pointTileSpace, float radius )                          = 0;

    virtual void                                                  ResetAdjustState()                                                                      = 0;
    virtual void                                                  PrepForAdjust( unsigned int segId, float3 pointTileSpace )                                       = 0;
    virtual void                                                  CommitAdjustChange( unsigned int segId, float3 pointTileSpace )                                  = 0;

	virtual void                                                  UndoChange()                                                                            = 0;
	virtual void                                                  RedoChange()                                                                            = 0;
    virtual void                                                  SaveAndClearFileSystemTileCache()                                                       = 0;

    virtual marray::Marray< unsigned char >                       GetIdColorMap()                                                                         = 0;
    virtual marray::Marray< unsigned char >                       GetIdConfidenceMap()                                                                          = 0;

	virtual void                                                  SortSegmentInfoById( bool reverse )			                                          = 0;
    virtual void                                                  SortSegmentInfoByName( bool reverse )		                                              = 0;
    virtual void                                                  SortSegmentInfoBySize( bool reverse )		                                              = 0;
    virtual void                                                  SortSegmentInfoByConfidence( bool reverse )	                                          = 0;
    virtual void                                                  LockSegmentLabel( unsigned int segId )                                                  = 0;
    virtual void                                                  UnlockSegmentLabel( unsigned int segId )                                                = 0;
	virtual unsigned int                                          GetSegmentInfoCount()                                                                   = 0;
    virtual std::list< SegmentInfo >                              GetSegmentInfoRange( int begin, int end ) 	                                          = 0;

};

}
}