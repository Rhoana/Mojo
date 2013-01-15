#include "TileManager.hpp"

#include <msclr/marshal_cppstd.h>

#include "Mojo.Core/Assert.hpp"
#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/ForEach.hpp"
#include "Mojo.Core/ID3D11CudaTexture.hpp"

#include "Mojo.Native/FileSystemTileServer.hpp"

namespace Mojo
{
namespace Interop
{

TileManager::TileManager( SlimDX::Direct3D11::Device^ d3d11Device, SlimDX::Direct3D11::DeviceContext^ d3d11DeviceContext, PrimitiveMap^ parameters )
{
    mTileServer = new Native::FileSystemTileServer( parameters->ToCore() );

    mTileManager = new Native::TileManager(
        reinterpret_cast< ID3D11Device* >( d3d11Device->ComPointer.ToPointer() ),
        reinterpret_cast< ID3D11DeviceContext* >( d3d11DeviceContext->ComPointer.ToPointer() ),
        mTileServer,
        parameters->ToCore() );

    mTileCache = gcnew Collections::Generic::List< TileCacheEntry^ >();
}

TileManager::~TileManager()
{
    if ( mTileCache != nullptr )
    {
        delete mTileCache;
        mTileCache = nullptr;
    }

    if ( mTileManager != NULL )
    {
        delete mTileManager;
        mTileManager = NULL;
    }

    if ( mTileServer != NULL )
    {
        delete mTileServer;
        mTileServer = NULL;
    }
}

Native::TileManager* TileManager::GetTileManager()
{
    return mTileManager;
}

void TileManager::LoadTiledDataset( TiledDatasetDescription^ tiledDatasetDescription )
{
    mTileManager->LoadTiledDataset( tiledDatasetDescription->ToNative() );

    UnloadTileCache();
    LoadTileCache();

    UnloadIdColorMap();
    LoadIdColorMap();
}

void TileManager::UnloadTiledDataset()
{
    mTileManager->UnloadTiledDataset();

    UnloadTileCache();
    LoadTileCache();

    UnloadIdColorMap();
    LoadIdColorMap();
}

bool TileManager::IsTiledDatasetLoaded()
{
    return mTileManager->IsTiledDatasetLoaded();
}

void TileManager::LoadSegmentation( TiledDatasetDescription^ tiledDatasetDescription )
{
    mTileManager->LoadSegmentation( tiledDatasetDescription->ToNative() );

    UnloadIdColorMap();
    LoadIdColorMap();
}

void TileManager::UnloadSegmentation()
{
    mTileManager->UnloadSegmentation();

    UnloadIdColorMap();
    LoadIdColorMap();
}

bool TileManager::IsSegmentationLoaded()
{
    return mTileManager->IsSegmentationLoaded();
}

void TileManager::Update()
{
    mTileManager->Update();
}

void TileManager::LoadTiles( TiledDatasetView^ tiledDatasetView )
{
    mTileManager->LoadTiles( tiledDatasetView->ToNative() );
}

Collections::Generic::IList< TileCacheEntry^ >^ TileManager::GetTileCache()
{
    UpdateTileCacheState();

    return mTileCache;
}

SlimDX::Direct3D11::ShaderResourceView^ TileManager::GetIdColorMap()
{
    return mIdColorMap;
}

int TileManager::GetSegmentationLabelId( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    return mTileManager->GetSegmentationLabelId( tiledDatasetView->ToNative(), pDataSpaceFloat3 );
}

Vector3 TileManager::GetSegmentationLabelColor( int id )
{
    int4 color = mTileManager->GetSegmentationLabelColor( id );
    return Vector3( (float)color.x, (float)color.y, (float)color.z );
}

void TileManager::ReplaceSegmentationLabel( int oldId, int newId )
{
    mTileManager->ReplaceSegmentationLabel( oldId, newId );
}

void TileManager::ReplaceSegmentationLabelCurrentSlice( int oldId, int newId, TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->ReplaceSegmentationLabelCurrentSlice( oldId, newId, pDataSpaceFloat3  );
}

void TileManager::DrawSplit( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->DrawSplit( pDataSpaceFloat3, radius );
}

void TileManager::AddSplitSource( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->AddSplitSource( pDataSpaceFloat3 );
}

void TileManager::RemoveSplitSource()
{
    mTileManager->RemoveSplitSource();
}

void TileManager::ResetSplitState()
{
    mTileManager->ResetSplitState();
}

void TileManager::PrepForSplit( int segId, Vector3^ pDataSpace )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->PrepForSplit( segId, pDataSpaceFloat3 );
}


void TileManager::FindSplitLine2D( int segId )
{
    mTileManager->FindSplitLine2D( segId );
}

void TileManager::FindSplitLine2DHover( int segId, Vector3^ pDataSpace )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->FindSplitLine2DHover( segId, pDataSpaceFloat3 );
}

int TileManager::CompleteSplit( int segId )
{
    return mTileManager->CompleteSplit( segId );
}

void TileManager::ReplaceSegmentationLabelCurrentConnectedComponent( int oldId, int newId, TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->ReplaceSegmentationLabelCurrentConnectedComponent( oldId, newId, pDataSpaceFloat3  );
}

void TileManager::UndoChange()
{
    mTileManager->UndoChange();
}

void TileManager::RedoChange()
{
    mTileManager->RedoChange();
}

void TileManager::SaveAndClearFileSystemTileCache()
{
	mTileManager->SaveAndClearFileSystemTileCache();
}

void TileManager::LoadTileCache()
{
    MOJO_FOR_EACH( Native::TileCacheEntry nativeTileCacheEntry, mTileManager->GetTileCache() )
    {
        TileCacheEntry^ tileCacheEntry = gcnew TileCacheEntry();

        MOJO_FOR_EACH_KEY_VALUE( std::string key, Core::ID3D11CudaTexture* value, nativeTileCacheEntry.d3d11CudaTextures.GetHashMap() )
        {
            tileCacheEntry->D3D11CudaTextures->Set( msclr::interop::marshal_as< String^ >( key ), ShaderResourceView::FromPointer( IntPtr( value->GetD3D11ShaderResourceView() ) ) );
        }

        tileCacheEntry->KeepState = (Mojo::Interop::TileCacheEntryKeepState)nativeTileCacheEntry.keepState;

        mTileCache->Add( tileCacheEntry );
    }
}

void TileManager::UnloadTileCache()
{
    for each ( TileCacheEntry^ tileCacheEntry in mTileCache )
    {
        for each ( Collections::Generic::KeyValuePair< String^, ShaderResourceView^ > keyValuePair in tileCacheEntry->D3D11CudaTextures )
        {
            delete keyValuePair.Value;
        }

        tileCacheEntry->D3D11CudaTextures->Internal->Clear();
    }

    mTileCache->Clear();
}

void TileManager::UpdateTileCacheState()
{
	boost::array< Native::TileCacheEntry, TILE_CACHE_SIZE >& tileCache = mTileManager->GetTileCache();

    if ( mTileCache->Count == tileCache.size() )
    {
		for ( unsigned int i = 0; i < tileCache.size(); i++ )
		{
            mTileCache[ i ]->KeepState       = (Mojo::Interop::TileCacheEntryKeepState)tileCache[ i ].keepState;
            mTileCache[ i ]->IndexTileSpace  = Vector4( (float)tileCache[ i ].indexTileSpace.x,  (float)tileCache[ i ].indexTileSpace.y,  (float)tileCache[ i ].indexTileSpace.z, (float)tileCache[ i ].indexTileSpace.w );
            mTileCache[ i ]->CenterDataSpace = Vector3( (float)tileCache[ i ].centerDataSpace.x, (float)tileCache[ i ].centerDataSpace.y, (float)tileCache[ i ].centerDataSpace.z );
            mTileCache[ i ]->ExtentDataSpace = Vector3( (float)tileCache[ i ].extentDataSpace.x, (float)tileCache[ i ].extentDataSpace.y, (float)tileCache[ i ].extentDataSpace.z );
            mTileCache[ i ]->Active          = tileCache[ i ].active;
		}
    }
}

void TileManager::LoadIdColorMap()
{
    ID3D11ShaderResourceView* idColorMap =  mTileManager->GetIdColorMap();

    if ( idColorMap != NULL )
    {
        RELEASE_ASSERT( mIdColorMap == nullptr );
        mIdColorMap = ShaderResourceView::FromPointer( IntPtr( idColorMap ) );
    }
}

void TileManager::UnloadIdColorMap()
{
    if ( mIdColorMap != nullptr )
    {
        delete mIdColorMap;
        mIdColorMap = nullptr;
    }
}

}
}