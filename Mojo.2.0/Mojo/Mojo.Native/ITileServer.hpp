#pragma once

#include "HashMap.hpp"
#include "VolumeDescription.hpp"
#include "Types.hpp"

#include <marray/marray.hxx>

#include "TiledDatasetDescription.hpp"
#include "FileSystemSegmentInfoManager.hpp"
#include "SegmentInfo.hpp"

namespace Mojo
{
namespace Native
{

//
// CODE QUALITY ISSUE:
// This interface is extremely large and seems to contain a lot of logic that does not pertain to
// serving tiles from a data store (e.g., a hard drive, a web server, etc.). This logic should really
// be implemented in TileManager. Then again, what exactly is a TileManager responsible for? And what is a TileServer
// responsible for? It seems like there is not a clear separation of concerns between these two collaborating
// types in the current implementation. -MR
//
class ITileServer
{
public:
    virtual ~ITileServer() {};

    virtual void                                              LoadSourceImages( TiledDatasetDescription& tiledDatasetDescription )                                               = 0;
    virtual void                                              UnloadSourceImages()                                                                                               = 0;
                                                                                                                                                                                 
    virtual bool                                              AreSourceImagesLoaded()                                                                                            = 0;
                                                                                                                                                                                 
    virtual void                                              LoadSegmentation( TiledDatasetDescription& tiledDatasetDescription )                                               = 0;
    virtual void                                              UnloadSegmentation()                                                                                               = 0;
                                                                                                                                                                                 
    virtual bool                                              IsSegmentationLoaded()                                                                                             = 0;
                                                                                                                                                                                 
    virtual void                                              SaveSegmentation()                                                                                                 = 0;
    virtual void                                              SaveSegmentationAs( std::string savePath )                                                                         = 0;
    virtual void                                              AutosaveSegmentation()                                                                                             = 0;
    virtual void                                              DeleteTempFiles()                                                                                                  = 0;
                                                                                                                                                                                 
    virtual int                                               GetTileCountForId( unsigned int segId )                                                                            = 0;
    virtual Int3                                              GetSegmentCentralTileLocation( unsigned int segId )                                                                = 0;
    virtual Int4                                              GetSegmentZTileBounds( unsigned int segId, int zIndex )                                                            = 0;
                                                                                                                                                                                  
                                                                                                                                                                                  
    virtual HashMap< std::string, VolumeDescription >         LoadTile( Int4 tileIndex )                                                                                         = 0;
    virtual void                                              UnloadTile( Int4 tileIndex )                                                                                       = 0;
                                                                                                                                                                                  
    virtual void                                              RemapSegmentLabels( std::set< unsigned int > fromSegIds, unsigned int toSegId )                                    = 0;
    virtual void                                              ReplaceSegmentationLabel( unsigned int oldId, unsigned int newId )                                                 = 0;
    virtual void                                              ReplaceSegmentationLabelCurrentSlice( unsigned int oldId, unsigned int newId, Float3 pointTileSpace )              = 0;
    virtual void                                              ReplaceSegmentationLabelCurrentConnectedComponent( unsigned int oldId, unsigned int newId, Float3 pointTileSpace ) = 0;

    virtual void                                              DrawSplit( Float3 pointTileSpace, float radius )                                                                   = 0;
    virtual void                                              DrawErase( Float3 pointTileSpace, float radius )                                                                   = 0;
    virtual void                                              DrawRegionB( Float3 pointTileSpace, float radius )                                                                 = 0;
    virtual void                                              DrawRegionA( Float3 pointTileSpace, float radius )                                                                 = 0;
                                                                                                                                                                                 
    virtual void                                              AddSplitSource( Float3 pointTileSpace )                                                                            = 0;
    virtual void                                              RemoveSplitSource()                                                                                                = 0;
    virtual void                                              ResetSplitState()                                                                                                  = 0;
    virtual void                                              PrepForSplit( unsigned int segId, Float3 pointTileSpace )                                                          = 0;
    virtual void                                              FindBoundaryJoinPoints2D( unsigned int segId )                                                                     = 0;
    virtual void                                              FindBoundaryWithinRegion2D( unsigned int segId )                                                                   = 0;
    virtual void                                              FindBoundaryBetweenRegions2D( unsigned int segId )                                                                 = 0;
    virtual unsigned int                                      CompletePointSplit( unsigned int segId, Float3 pointTileSpace )                                                    = 0;
    virtual unsigned int                                      CompleteDrawSplit( unsigned int segId, Float3 pointTileSpace, bool join3D, int splitStartZ )                       = 0;

    virtual void                                              RecordSplitState( unsigned int segId, Float3 pointTileSpace )                                                      = 0;
    virtual void                                              PredictSplit( unsigned int segId, Float3 pointTileSpace, float radius )                                            = 0;
                                                                                                                                                                                   
    virtual void                                              ResetAdjustState()                                                                                                 = 0;
    virtual void                                              PrepForAdjust( unsigned int segId, Float3 pointTileSpace )                                                         = 0;
    virtual void                                              CommitAdjustChange( unsigned int segId, Float3 pointTileSpace )                                                    = 0;

    virtual void                                              ResetDrawMergeState()                                                                                              = 0;
    virtual void                                              PrepForDrawMerge( Float3 pointTileSpace )                                                                          = 0;
    virtual std::set< unsigned int >                          GetDrawMergeIds( Float3 pointTileSpace )                                                                           = 0;
    virtual std::map< unsigned int, Float3 >                  GetDrawMergeIdsAndPoints( Float3 pointTileSpace )                                                                  = 0;
    virtual unsigned int                                      CommitDrawMerge( std::set< unsigned int > mergeIds, Float3 pointTileSpace )                                        = 0;
    virtual unsigned int                                      CommitDrawMergeCurrentSlice( Float3 pointTileSpace )                                                               = 0;
    virtual unsigned int                                      CommitDrawMergeCurrentConnectedComponent( Float3 pointTileSpace )                                                  = 0;

    virtual unsigned int                                      GetNewId()                                                                                                         = 0;
    virtual std::list< unsigned int >                         UndoChange()                                                                                                       = 0;
    virtual std::list< unsigned int >                         RedoChange()                                                                                                       = 0;
    virtual void                                              TempSaveAndClearFileSystemTileCache()                                                                              = 0;
    virtual void                                              ClearFileSystemTileCache()                                                                                         = 0;
    virtual float                                             GetCurrentOperationProgress()                                                                                      = 0;
                                                                                                                                                                                 
    virtual marray::Marray< unsigned char >*                  GetIdColorMap()                                                                                                    = 0;
    virtual marray::Marray< unsigned int >*                   GetLabelIdMap()                                                                                                    = 0;
    virtual marray::Marray< unsigned char >*                  GetIdConfidenceMap()                                                                                               = 0;
                                                                                                                                                                                 
    virtual void                                              SortSegmentInfoById( bool reverse )                                                                                = 0;
    virtual void                                              SortSegmentInfoByName( bool reverse )                                                                              = 0;
    virtual void                                              SortSegmentInfoBySize( bool reverse )                                                                              = 0;
    virtual void                                              SortSegmentInfoByConfidence( bool reverse )                                                                        = 0;
    virtual void                                              LockSegmentLabel( unsigned int segId )                                                                             = 0;
    virtual void                                              UnlockSegmentLabel( unsigned int segId )                                                                           = 0;
    virtual unsigned int                                      GetSegmentInfoCount()                                                                                              = 0;
    virtual unsigned int                                      GetSegmentInfoCurrentListLocation( unsigned int segId )                                                            = 0;
    virtual std::list< SegmentInfo >                          GetSegmentInfoRange( int begin, int end )                                                                          = 0;
    virtual SegmentInfo                                       GetSegmentInfo( unsigned int segId )                                                                               = 0;

    virtual FileSystemSegmentInfoManager*                     GetSegmentInfoManager()                                                                                            = 0;

};

}
}