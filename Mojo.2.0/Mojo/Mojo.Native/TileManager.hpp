#pragma once

#include "Stl.hpp"

#include <marray/marray.hxx>
#include <marray/marray_hdf5.hxx>

#include "Boost.hpp"
#include "D3D11.hpp"
#include "Types.hpp"
#include "ID3D11Texture.hpp"
#include "PrimitiveMap.hpp"
#include "VolumeDescription.hpp"

#include "TiledDatasetDescription.hpp"
#include "TiledDatasetView.hpp"
#include "TileCacheEntry.hpp"
#include "ITileServer.hpp"
#include "SegmentInfo.hpp"
#include "FileSystemTileServerConstants.hpp"

struct ID3D11Device;
struct ID3D11DeviceContext;

namespace Mojo
{
namespace Native
{

class TileManager
{
public:
    TileManager( ID3D11Device* d3d11Device, ID3D11DeviceContext* d3d11DeviceContext, ITileServer* tileServer, PrimitiveMap constParameters );
    ~TileManager();

    void                                                  LoadSourceImages( TiledDatasetDescription& tiledDatasetDescription );
    void                                                  UnloadSourceImages();

    bool                                                  AreSourceImagesLoaded();

    void                                                  LoadSegmentation( TiledDatasetDescription& tiledDatasetDescription );
    void                                                  UnloadSegmentation();

    bool                                                  IsSegmentationLoaded();

    void                                                  SaveSegmentation();
    void                                                  SaveSegmentationAs( std::string savePath );
    void                                                  AutosaveSegmentation();
    void                                                  DeleteTempFiles();

    void                                                  LoadTiles( const TiledDatasetView& tiledDatasetView );

    //
    // CODE QUALITY ISSUE:
    // LoadOverTile is a bad name for this function because it doesn't explain what the function does.
    // I still don't really know what this function does even after reading it. Consider LoadLowResolutionTile(...) instead. -MR
    //
    void                                                  LoadOverTile( const TiledDatasetView& tiledDatasetView );

    std::vector< TileCacheEntry >&                        GetTileCache();
    ID3D11ShaderResourceView*                             GetIdColorMap();
    ID3D11ShaderResourceView*                             GetLabelIdMap();
    ID3D11ShaderResourceView*                             GetIdConfidenceMap();

    void                                                  SortSegmentInfoById( bool reverse );
    void                                                  SortSegmentInfoByName( bool reverse );
    void                                                  SortSegmentInfoBySize( bool reverse );
    void                                                  SortSegmentInfoByConfidence( bool reverse );
    void                                                  RemapSegmentLabel( unsigned int fromSegId, unsigned int toSegId );
    void                                                  LockSegmentLabel( unsigned int segId );
    void                                                  UnlockSegmentLabel( unsigned int segId );
    unsigned int                                          GetSegmentInfoCount();
    unsigned int                                          GetSegmentInfoCurrentListLocation( unsigned int segId );
    std::list< SegmentInfo >                              GetSegmentInfoRange( int begin, int end );
    SegmentInfo                                           GetSegmentInfo( unsigned int segId );

    unsigned int                                          GetSegmentationLabelId( const TiledDatasetView& tiledDatasetView, Float3 pDataSpace );
    Int3                                              GetSegmentationLabelColor( unsigned int segId );
    std::string                                           GetSegmentationLabelColorString( unsigned int segId );
    Int3                                              GetSegmentCentralTileLocation( unsigned int segId );
    Int4                                              GetSegmentZTileBounds( unsigned int segId, int zIndex );

    void                                                  ReplaceSegmentationLabel( unsigned int oldId, unsigned int newId );
    void                                                  ReplaceSegmentationLabelCurrentSlice( unsigned int oldId, unsigned int newId, Float3 pDataSpace );
    void                                                  ReplaceSegmentationLabelCurrentConnectedComponent( unsigned int oldId, unsigned int newId, Float3 pDataSpace );

    void                                                  DrawSplit( Float3 pointTileSpace, float radius );
    void                                                  DrawErase( Float3 pointTileSpace, float radius );
    void                                                  DrawRegionB( Float3 pointTileSpace, float radius );
    void                                                  DrawRegionA( Float3 pointTileSpace, float radius );

    void                                                  AddSplitSource( Float3 pointTileSpace );
    void                                                  RemoveSplitSource();
    void                                                  ResetSplitState( Float3 pointTileSpace );
    void                                                  PrepForSplit( unsigned int segId, Float3 pointTileSpace );
    void                                                  FindBoundaryJoinPoints2D( unsigned int segId, Float3 pointTileSpace );
    void                                                  FindBoundaryWithinRegion2D( unsigned int segId, Float3 pointTileSpace );
    void                                                  FindBoundaryBetweenRegions2D( unsigned int segId, Float3 pointTileSpace );
    int                                                   CompletePointSplit( unsigned int segId, Float3 pointTileSpace );
    int                                                   CompleteDrawSplit( unsigned int segId, Float3 pointTileSpace, bool join3D, int splitStartZ );
    void                                                  RecordSplitState( unsigned int segId, Float3 pointTileSpace );
    void                                                  PredictSplit( unsigned int segId, Float3 pointTileSpace, float radius );

    void                                                  ResetAdjustState( Float3 pointTileSpace );
    void                                                  PrepForAdjust( unsigned int segId, Float3 pointTileSpace );
    void                                                  CommitAdjustChange( unsigned int segId, Float3 pointTileSpace );

    void                                                  ResetDrawMergeState( Float3 pointTileSpace );
    void                                                  PrepForDrawMerge( Float3 pointTileSpace );
    unsigned int                                          CommitDrawMerge( Float3 pointTileSpace );
    unsigned int                                          CommitDrawMergeCurrentSlice( Float3 pointTileSpace );
    unsigned int                                          CommitDrawMergeCurrentConnectedComponent( Float3 pointTileSpace );

    unsigned int                                          GetNewId();
    void                                                  UndoChange();
    void                                                  RedoChange();
    void                                                  TempSaveAndClearFileSystemTileCache();
    void                                                  ClearFileSystemTileCache();
    float                                                 GetCurrentOperationProgress();

    Int3                                              GetZoomLevel( const TiledDatasetView& tiledDatasetView );

private:

    void                                                  LoadSourceImagesInternal( TiledDatasetDescription& tiledDatasetDescription );
    void                                                  UnloadSourceImagesInternal();

    void                                                  LoadSegmentationInternal( TiledDatasetDescription& tiledDatasetDescription );
    void                                                  UnloadSegmentationInternal();

    std::list< Int4 >                                 GetTileIndicesIntersectedByView( const TiledDatasetView& tiledDatasetView );
    Int4                                              GetTileIndexCoveringView( const TiledDatasetView& tiledDatasetView );

    void                                                  GetIndexTileSpace( Int3 zoomLevel, Float3 pointDataSpace, Float4& pointTileSpace, Int4& tileIndex );
    Int3                                              GetIndexVoxelSpace( Float4 pointTileSpace, Int3 numVoxelsPerTile );
    Int3                                              GetOffsetVoxelSpace( Float4 pTileSpace, Int4 pTileIndex, Int3 numVoxelsPerTile );

    void                                                  UpdateLabelIdMap( unsigned int fromSegId );

    void                                                  ReloadTileCache();
    void                                                  ReloadTileCacheOverlayMapOnly( int currentZ );

    ID3D11Device*                                         mD3D11Device;
    ID3D11DeviceContext*                                  mD3D11DeviceContext;

    ID3D11Buffer*                                         mIdColorMapBuffer;
    ID3D11ShaderResourceView*                             mIdColorMapShaderResourceView;

    ID3D11Buffer*                                         mLabelIdMapBuffer;
    ID3D11ShaderResourceView*                             mLabelIdMapShaderResourceView;

    ID3D11Buffer*                                         mIdConfidenceMapBuffer;
    ID3D11ShaderResourceView*                             mIdConfidenceMapShaderResourceView;

    ITileServer*                                          mTileServer;

    PrimitiveMap                                    mConstParameters;
                                                
    TiledDatasetDescription                               mSourceImagesTiledDatasetDescription;
    TiledDatasetDescription                               mSegmentationTiledDatasetDescription;

    std::vector< TileCacheEntry >                         mTileCache;

    int                                                   mTileCacheSearchStart;

    marray::Marray< int >                                 mTileCachePageTable;
    marray::Marray< unsigned char >*                      mIdColorMap;
    marray::Marray< unsigned int >*                       mLabelIdMap;
    marray::Marray< unsigned char >*                      mIdConfidenceMap;

    bool                                                  mAreSourceImagesLoaded;
    bool                                                  mIsSegmentationLoaded;

    int                                                   mDeviceTileCacheSize;


};

}
}