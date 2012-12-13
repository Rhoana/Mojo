#pragma once

#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/VolumeDescription.hpp"

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

    virtual Core::HashMap< std::string, Core::VolumeDescription > LoadTile( int4 tileIndex )                                                              = 0;
    virtual void                                                  UnloadTile( int4 tileIndex )                                                            = 0;

    virtual void                                                  ReplaceSegmentationLabel( int oldId, int newId )                                        = 0;
    virtual void                                                  ReplaceSegmentationLabelCurrentSlice( int oldId, int newId, float3 pointTileSpace )     = 0;
    virtual void                                                  ReplaceSegmentationLabelCurrentConnectedComponent( int oldId, int newId, float3 pointTileSpace )        = 0;

    virtual void                                                  AddSplitSource( float3 pointTileSpace )                                                 = 0;
    virtual void                                                  RemoveSplitSource()                                                                     = 0;
    virtual void                                                  ResetSplitState()                                                                       = 0;
    virtual void                                                  PrepForSplit( int segId, int zIndex )                                                   = 0;
	virtual void                                                  FindSplitLine2DTemp( int segId )                                                        = 0;
    virtual int                                                   CompleteSplit( int segId )                                                              = 0;

	virtual void                                                  UndoChange() = 0;
	virtual void                                                  RedoChange() = 0;
    virtual void                                                  SaveAndClearFileSystemTileCache()                                                       = 0;
};

}
}