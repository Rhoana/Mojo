#pragma once

#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/VolumeDescription.hpp"

#include <marray/marray.hxx>

#include "TiledDatasetDescription.hpp"

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

    virtual int                                                   GetTileCountForId( int segId )                                                          = 0;

    virtual Core::HashMap< std::string, Core::VolumeDescription > LoadTile( int4 tileIndex )                                                              = 0;
    virtual void                                                  UnloadTile( int4 tileIndex )                                                            = 0;

    virtual void                                                  ReplaceSegmentationLabel( int oldId, int newId )                                        = 0;
    virtual void                                                  ReplaceSegmentationLabelCurrentSlice( int oldId, int newId, float3 pointTileSpace )     = 0;
    virtual void                                                  ReplaceSegmentationLabelCurrentConnectedComponent( int oldId, int newId, float3 pointTileSpace )        = 0;

    virtual void                                                  DrawSplit( float3 pointTileSpace, float radius )                                        = 0;
    virtual void                                                  DrawErase( float3 pointTileSpace, float radius )                                        = 0;
    virtual void                                                  DrawRegionB( float3 pointTileSpace, float radius )                                      = 0;
    virtual void                                                  DrawRegionA( float3 pointTileSpace, float radius )                                      = 0;
    virtual void                                                  AddSplitSource( float3 pointTileSpace )                                                 = 0;
    virtual void                                                  RemoveSplitSource()                                                                     = 0;
    virtual void                                                  ResetSplitState()                                                                       = 0;
    virtual void                                                  PrepForSplit( int segId, float3 pointTileSpace )                                        = 0;
	virtual void                                                  FindBoundaryJoinPoints2D( int segId )                                                   = 0;
	virtual void                                                  FindBoundaryWithinRegion2D( int segId )                                                 = 0;
	virtual void                                                  FindBoundaryBetweenRegions2D( int segId )                                               = 0;
    virtual int                                                   CompletePointSplit( int segId, float3 pointTileSpace )                                  = 0;
    virtual int                                                   CompleteDrawSplit( int segId, float3 pointTileSpace )                                   = 0;

	virtual void                                                  UndoChange() = 0;
	virtual void                                                  RedoChange() = 0;
    virtual void                                                  SaveAndClearFileSystemTileCache()                                                       = 0;

    virtual marray::Marray< unsigned char >                       GetIdColorMap()                                                                         = 0;

};

}
}