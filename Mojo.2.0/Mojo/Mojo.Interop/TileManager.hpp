#pragma once

//#include "vld.h"

#include "Mojo.Native/ITileServer.hpp"
#include "Mojo.Native/TileManager.hpp"

#include "NotifyPropertyChanged.hpp"
#include "PrimitiveMap.hpp"
#include "TiledDatasetDescription.hpp"
#include "TileCacheEntry.hpp"
#include "TiledDatasetView.hpp"
#include "SegmentInfo.hpp"

#using <SlimDX.dll>

using namespace SlimDX;

namespace Mojo
{
namespace Interop
{

#pragma managed
public ref class TileManager : public NotifyPropertyChanged
{
public:
    TileManager( SlimDX::Direct3D11::Device^ d3d11Device, SlimDX::Direct3D11::DeviceContext^ d3d11DeviceContext, PrimitiveMap^ parameters );
    ~TileManager();

    Native::TileManager*                                    GetTileManager();

    void                                                    LoadTiledDataset( TiledDatasetDescription^ tiledDatasetDescription );
    void                                                    UnloadTiledDataset();

    bool                                                    IsTiledDatasetLoaded();

	void                                                    LoadSegmentation( TiledDatasetDescription^ tiledDatasetDescription );
    void                                                    UnloadSegmentation();

    bool                                                    IsSegmentationLoaded();

    void                                                    SaveSegmentation();
    void                                                    SaveSegmentationAs( String^ savePath );
    void                                                    AutosaveSegmentation();
    void                                                    DeleteTempFiles();

    void                                                    Update();

    void                                                    LoadTiles( TiledDatasetView^ tiledDatasetView );

    Collections::Generic::IList< TileCacheEntry^ >^         GetTileCache();
    SlimDX::Direct3D11::ShaderResourceView^                 GetIdColorMap();

    void                                                    SortSegmentInfoById( bool reverse );
    void                                                    SortSegmentInfoByName( bool reverse );
    void                                                    SortSegmentInfoBySize( bool reverse );
    void                                                    SortSegmentInfoByConfidence( bool reverse );
	void                                                    LockSegmentLabel( unsigned int segId );
	void                                                    UnlockSegmentLabel( unsigned int segId );
    Collections::Generic::IList< SegmentInfo^ >^            GetSegmentInfoRange( int begin, int end );

    unsigned int                                            GetSegmentationLabelId( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace );
    Vector3                                                 GetSegmentationLabelColor( unsigned int segId );

    void                                                    ReplaceSegmentationLabel( unsigned int oldId, unsigned int newId );
    void                                                    ReplaceSegmentationLabelCurrentSlice( unsigned int oldId, unsigned int newId, TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace );
    void                                                    ReplaceSegmentationLabelCurrentConnectedComponent( unsigned int oldId, unsigned int newId, TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace );

    void                                                    DrawSplit( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius );
    void                                                    DrawErase( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius );
    void                                                    DrawRegionB( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius );
    void                                                    DrawRegionA( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius );

    void                                                    AddSplitSource( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace );
    void                                                    RemoveSplitSource();
    void                                                    ResetSplitState();
    void                                                    PrepForSplit( unsigned int segId, Vector3^ pDataSpace );
	void                                                    FindBoundaryJoinPoints2D( unsigned int segId );
	void                                                    FindBoundaryWithinRegion2D( unsigned int segId );
	void                                                    FindBoundaryBetweenRegions2D( unsigned int segId );
    int                                                     CompletePointSplit( unsigned int segId, Vector3^ pDataSpace );
    int                                                     CompleteDrawSplit( unsigned int segId, Vector3^ pDataSpace, bool join3D, int splitStartZ );
    void                                                    RecordSplitState( unsigned int segId, Vector3^ pDataSpace );
    void                                                    PredictSplit( unsigned int segId, Vector3^ pDataSpace, float radius );

    void                                                    ResetAdjustState();
    void                                                    PrepForAdjust( unsigned int segId, Vector3^ pDataSpace );
    void                                                    CommitAdjustChange( unsigned int segId, Vector3^ pDataSpace );

	void                                                    UndoChange();
	void                                                    RedoChange();
    void                                                    SaveAndClearFileSystemTileCache();

private:
    void LoadTileCache();
    void UnloadTileCache();
    void UpdateTileCacheState();

    void LoadIdColorMap();
    void UnloadIdColorMap();

    Native::TileManager*                                    mTileManager;
    Native::ITileServer*                                    mTileServer;

    Collections::Generic::IList< TileCacheEntry^ >^         mTileCache;
    SlimDX::Direct3D11::ShaderResourceView^                 mIdColorMap;

};

}
}