#pragma once

#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/VolumeDescription.hpp"
#include "Mojo.Core/MojoVectors.hpp"

#include <marray/marray.hxx>

#include "TiledDatasetDescription.hpp"
#include "FileSystemSegmentInfoManager.hpp"
#include "SegmentInfo.hpp"

using namespace Mojo::Core;

namespace Mojo
{
namespace Native
{

class ITileServer
{
public:
    virtual ~ITileServer() {};

    virtual void                                                  LoadTiledDataset( TiledDatasetDescription& tiledDatasetDescription )                                                   = 0;
    virtual void                                                  UnloadTiledDataset()                                                                                                   = 0;
																																					                                  
    virtual bool                                                  IsTiledDatasetLoaded()                                                                                                 = 0;
																																					                                  
    virtual void                                                  LoadSegmentation( TiledDatasetDescription& tiledDatasetDescription )                                                   = 0;
    virtual void                                                  UnloadSegmentation()                                                                                                   = 0;
																																					                                  
    virtual bool                                                  IsSegmentationLoaded()                                                                                                 = 0;
																																					                                  
    virtual void                                                  SaveSegmentation()                                                                                                     = 0;
    virtual void                                                  SaveSegmentationAs( std::string savePath )                                                                             = 0;
    virtual void                                                  AutosaveSegmentation()                                                                                                 = 0;
    virtual void                                                  DeleteTempFiles()                                                                                                      = 0;
																																					                                  
    virtual int                                                   GetTileCountForId( unsigned int segId )                                                                                = 0;
    virtual MojoInt3                                              GetSegmentCentralTileLocation( unsigned int segId )                                                                    = 0;
    virtual MojoInt4                                              GetSegmentZTileBounds( unsigned int segId, int zIndex )                                                                = 0;
																																					                                  
																																					                                  
    virtual Core::HashMap< std::string, Core::VolumeDescription > LoadTile( MojoInt4 tileIndex )                                                                                         = 0;
    virtual void                                                  UnloadTile( MojoInt4 tileIndex )                                                                                       = 0;
																																					                                  
    virtual void                                                  RemapSegmentLabels( std::set< unsigned int > fromSegIds, unsigned int toSegId )                                       = 0;
    virtual void                                                  ReplaceSegmentationLabel( unsigned int oldId, unsigned int newId )                                                     = 0;
    virtual void                                                  ReplaceSegmentationLabelCurrentSlice( unsigned int oldId, unsigned int newId, MojoFloat3 pointTileSpace )              = 0;
    virtual void                                                  ReplaceSegmentationLabelCurrentConnectedComponent( unsigned int oldId, unsigned int newId, MojoFloat3 pointTileSpace ) = 0;

    virtual void                                                  DrawSplit( MojoFloat3 pointTileSpace, float radius )                                                                   = 0;
    virtual void                                                  DrawErase( MojoFloat3 pointTileSpace, float radius )                                                                   = 0;
    virtual void                                                  DrawRegionB( MojoFloat3 pointTileSpace, float radius )                                                                 = 0;
    virtual void                                                  DrawRegionA( MojoFloat3 pointTileSpace, float radius )                                                                 = 0;
																																					                                 
    virtual void                                                  AddSplitSource( MojoFloat3 pointTileSpace )                                                                            = 0;
    virtual void                                                  RemoveSplitSource()                                                                                                    = 0;
    virtual void                                                  ResetSplitState()                                                                                                      = 0;
    virtual void                                                  PrepForSplit( unsigned int segId, MojoFloat3 pointTileSpace )                                                          = 0;
	virtual void                                                  FindBoundaryJoinPoints2D( unsigned int segId )                                                                         = 0;
	virtual void                                                  FindBoundaryWithinRegion2D( unsigned int segId )                                                                       = 0;
	virtual void                                                  FindBoundaryBetweenRegions2D( unsigned int segId )                                                                     = 0;
    virtual unsigned int                                          CompletePointSplit( unsigned int segId, MojoFloat3 pointTileSpace )                                                    = 0;
    virtual unsigned int                                          CompleteDrawSplit( unsigned int segId, MojoFloat3 pointTileSpace, bool join3D, int splitStartZ )                       = 0;

    virtual void                                                  RecordSplitState( unsigned int segId, MojoFloat3 pointTileSpace )                                                      = 0;
    virtual void                                                  PredictSplit( unsigned int segId, MojoFloat3 pointTileSpace, float radius )                                            = 0;
																																						                               
    virtual void                                                  ResetAdjustState()                                                                                                     = 0;
    virtual void                                                  PrepForAdjust( unsigned int segId, MojoFloat3 pointTileSpace )                                                         = 0;
    virtual void                                                  CommitAdjustChange( unsigned int segId, MojoFloat3 pointTileSpace )                                                    = 0;

    virtual void                                                  ResetDrawMergeState()                                                                                                  = 0;
    virtual void                                                  PrepForDrawMerge( MojoFloat3 pointTileSpace )                                                                          = 0;
	virtual std::set< unsigned int >                              GetDrawMergeIds( MojoFloat3 pointTileSpace )                                                                           = 0;
	virtual std::map< unsigned int, MojoFloat3 >                  GetDrawMergeIdsAndPoints( MojoFloat3 pointTileSpace )                                                                  = 0;
	virtual unsigned int                                          CommitDrawMerge( std::set< unsigned int > mergeIds, MojoFloat3 pointTileSpace )                                        = 0;
	virtual unsigned int                                          CommitDrawMergeCurrentSlice( MojoFloat3 pointTileSpace )                                                               = 0;
	virtual unsigned int                                          CommitDrawMergeCurrentConnectedComponent( MojoFloat3 pointTileSpace )                                                  = 0;

	virtual std::list< unsigned int >                             UndoChange()                                                                                                           = 0;
	virtual std::list< unsigned int >                             RedoChange()                                                                                                           = 0;
    virtual void                                                  TempSaveAndClearFileSystemTileCache()                                                                                  = 0;
    virtual void                                                  ClearFileSystemTileCache()                                                                                             = 0;
																																						                               
    virtual marray::Marray< unsigned char >*                      GetIdColorMap()                                                                                                        = 0;
    virtual marray::Marray< unsigned int >*                       GetLabelIdMap()                                                                                                        = 0;
    virtual marray::Marray< unsigned char >*                      GetIdConfidenceMap()                                                                                                   = 0;
																																						                               
	virtual void                                                  SortSegmentInfoById( bool reverse )			                                                                         = 0;
    virtual void                                                  SortSegmentInfoByName( bool reverse )		                                                                             = 0;
    virtual void                                                  SortSegmentInfoBySize( bool reverse )		                                                                             = 0;
    virtual void                                                  SortSegmentInfoByConfidence( bool reverse )	                                                                         = 0;
    virtual void                                                  LockSegmentLabel( unsigned int segId )                                                                                 = 0;
    virtual void                                                  UnlockSegmentLabel( unsigned int segId )                                                                               = 0;
	virtual unsigned int                                          GetSegmentInfoCount()                                                                                                  = 0;
	virtual unsigned int                                          GetSegmentInfoCurrentListLocation( unsigned int segId )                                                                = 0;
    virtual std::list< SegmentInfo >                              GetSegmentInfoRange( int begin, int end ) 	                                                                         = 0;

    virtual FileSystemSegmentInfoManager*                         GetSegmentInfoManager()                                                                                                = 0;

};

}
}