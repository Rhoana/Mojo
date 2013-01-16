#pragma once

//#include "vld.h"

#include "Mojo.Native/ITileServer.hpp"
#include "Mojo.Native/TileManager.hpp"

#include "NotifyPropertyChanged.hpp"
#include "PrimitiveMap.hpp"
#include "TiledDatasetDescription.hpp"
#include "TileCacheEntry.hpp"
#include "TiledDatasetView.hpp"

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

    void                                                    Update();

    void                                                    LoadTiles( TiledDatasetView^ tiledDatasetView );

    Collections::Generic::IList< TileCacheEntry^ >^         GetTileCache();
    SlimDX::Direct3D11::ShaderResourceView^                 GetIdColorMap();

    int                                                     GetSegmentationLabelId( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace );
    Vector3                                                 GetSegmentationLabelColor( int id );

    void                                                    ReplaceSegmentationLabel( int oldId, int newId );
    void                                                    ReplaceSegmentationLabelCurrentSlice( int oldId, int newId, TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace );
    void                                                    ReplaceSegmentationLabelCurrentConnectedComponent( int oldId, int newId, TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace );

    void                                                    DrawSplit( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius );
    void                                                    DrawRegionB( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius );
    void                                                    DrawRegionA( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius );
    void                                                    AddSplitSource( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace );
    void                                                    RemoveSplitSource();
    void                                                    ResetSplitState();
    void                                                    PrepForSplit( int segId, Vector3^ pDataSpace );
	void                                                    FindBoundaryJoinPoints2D( int segId );
	void                                                    FindBoundaryWithinRegion2D( int segId );
	void                                                    FindCutBetweenRegions2D( int segId );
    int                                                     CompleteSplit( int segId );

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