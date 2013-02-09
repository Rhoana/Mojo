#include "TileManager.hpp"

#include <msclr/marshal_cppstd.h>

#include "Mojo.Core/Assert.hpp"
#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/ForEach.hpp"
#include "Mojo.Core/ID3D11CudaTexture.hpp"
#include "Mojo.Core/Stl.hpp"

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

    mTileServer->UnloadSegmentation();

    UnloadIdColorMap();
    LoadIdColorMap();
}

bool TileManager::IsSegmentationLoaded()
{
    return mTileManager->IsSegmentationLoaded();
}

void TileManager::SaveSegmentation()
{
    return mTileManager->SaveSegmentation();
}

void TileManager::SaveSegmentationAs( String^ savePath )
{
    return mTileManager->SaveSegmentationAs( msclr::interop::marshal_as < std::string >( savePath ) );
}

void TileManager::AutosaveSegmentation()
{
    return mTileManager->AutosaveSegmentation();
}

void TileManager::DeleteTempFiles()
{
    return mTileManager->DeleteTempFiles();
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

void TileManager::SortSegmentInfoById( bool reverse )
{
	mTileManager->SortSegmentInfoById( reverse );
}

void TileManager::SortSegmentInfoByName( bool reverse )
{
	mTileManager->SortSegmentInfoByName( reverse );
}

void TileManager::SortSegmentInfoBySize( bool reverse )
{
	mTileManager->SortSegmentInfoBySize( reverse );
}

void TileManager::SortSegmentInfoByConfidence( bool reverse )
{
	mTileManager->SortSegmentInfoByConfidence( reverse );
}

void TileManager::LockSegmentLabel( unsigned int segId )
{
	mTileManager->LockSegmentLabel( segId );
}

void TileManager::UnlockSegmentLabel( unsigned int segId )
{
	mTileManager->UnlockSegmentLabel( segId );
}

Collections::Generic::IList< SegmentInfo^ >^ TileManager::GetSegmentInfoRange( int begin, int end )
{
    std::list< Native::SegmentInfo > segmentInfoPage = mTileManager->GetSegmentInfoRange( begin, end );

    Collections::Generic::List< SegmentInfo^ >^ interopSegmentInfoPage = gcnew Collections::Generic::List< SegmentInfo^ >();

	std::ostringstream colorConverter;
	colorConverter << std::setfill( '0' ) << std::hex;

	for ( std::list< Native::SegmentInfo >::iterator segIt = segmentInfoPage.begin(); segIt != segmentInfoPage.end(); ++segIt )
	{
		int4 color = mTileManager->GetSegmentationLabelColor( segIt->id );

		colorConverter.str("");
		colorConverter << std::setw( 1 ) << "#";
		colorConverter << std::setw( 2 ) << color.x;
		colorConverter << std::setw( 2 ) << color.y;
		colorConverter << std::setw( 2 ) << color.z;

        interopSegmentInfoPage->Add( gcnew SegmentInfo( *segIt, colorConverter.str() ) );
	}

    return interopSegmentInfoPage;
}


SlimDX::Direct3D11::ShaderResourceView^ TileManager::GetIdColorMap()
{
    return mIdColorMap;
}

unsigned int TileManager::GetSegmentationLabelId( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    return mTileManager->GetSegmentationLabelId( tiledDatasetView->ToNative(), pDataSpaceFloat3 );
}

Vector3 TileManager::GetSegmentationLabelColor( unsigned int segId )
{
    int4 color = mTileManager->GetSegmentationLabelColor( segId );
    return Vector3( (float)color.x, (float)color.y, (float)color.z );
}

void TileManager::ReplaceSegmentationLabel( unsigned int oldId, unsigned int newId )
{
    mTileManager->ReplaceSegmentationLabel( oldId, newId );
}

void TileManager::ReplaceSegmentationLabelCurrentSlice( unsigned int oldId, unsigned int newId, TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->ReplaceSegmentationLabelCurrentSlice( oldId, newId, pDataSpaceFloat3  );
}

void TileManager::ReplaceSegmentationLabelCurrentConnectedComponent( unsigned int oldId, unsigned int newId, TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->ReplaceSegmentationLabelCurrentConnectedComponent( oldId, newId, pDataSpaceFloat3  );
}

void TileManager::DrawSplit( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->DrawSplit( pDataSpaceFloat3, radius );
}

void TileManager::DrawErase( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->DrawErase( pDataSpaceFloat3, radius );
}

void TileManager::DrawRegionA( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->DrawRegionA( pDataSpaceFloat3, radius );
}

void TileManager::DrawRegionB( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->DrawRegionB( pDataSpaceFloat3, radius );
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

void TileManager::PrepForSplit( unsigned int segId, Vector3^ pDataSpace )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->PrepForSplit( segId, pDataSpaceFloat3 );
}

void TileManager::FindBoundaryJoinPoints2D( unsigned int segId )
{
    mTileManager->FindBoundaryJoinPoints2D( segId );
}

void TileManager::FindBoundaryWithinRegion2D( unsigned int segId )
{
    mTileManager->FindBoundaryWithinRegion2D( segId );
}

void TileManager::FindBoundaryBetweenRegions2D( unsigned int segId )
{
    mTileManager->FindBoundaryBetweenRegions2D( segId );
}

int TileManager::CompletePointSplit( unsigned int segId, Vector3^ pDataSpace )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    return mTileManager->CompletePointSplit( segId, pDataSpaceFloat3 );
}

int TileManager::CompleteDrawSplit( unsigned int segId, Vector3^ pDataSpace, bool join3D, int splitStartZ )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    return mTileManager->CompleteDrawSplit( segId, pDataSpaceFloat3, join3D, splitStartZ );
}

void TileManager::RecordSplitState( unsigned int segId, Vector3^ pDataSpace )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->RecordSplitState( segId, pDataSpaceFloat3 );
}

void TileManager::PredictSplit( unsigned int segId, Vector3^ pDataSpace, float radius )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->PredictSplit( segId, pDataSpaceFloat3, radius );
}

void TileManager::ResetAdjustState()
{
    mTileManager->ResetAdjustState();
}

void TileManager::PrepForAdjust( unsigned int segId, Vector3^ pDataSpace )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->PrepForAdjust( segId, pDataSpaceFloat3 );
}

void TileManager::CommitAdjustChange( unsigned int segId, Vector3^ pDataSpace )
{
    float3 pDataSpaceFloat3 = make_float3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->CommitAdjustChange( segId, pDataSpaceFloat3 );
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
	boost::array< Native::TileCacheEntry, DEVICE_TILE_CACHE_SIZE >& tileCache = mTileManager->GetTileCache();

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