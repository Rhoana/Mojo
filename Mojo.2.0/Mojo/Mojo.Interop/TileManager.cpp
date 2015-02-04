#include "TileManager.hpp"

#include <msclr/marshal_cppstd.h>

#include "Mojo.Core/Assert.hpp"
//#include "Mojo.Core/Cuda.hpp"
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

    //UnloadIdColorMap();
    //LoadIdColorMap();

    //UnloadLabelIdMap();
    //LoadLabelIdMap();

	//UnloadIdConfidenceMap();
	//LoadIdConfidenceMap();
}

void TileManager::UnloadTiledDataset()
{
    mTileManager->UnloadTiledDataset();

    UnloadTileCache();
    LoadTileCache();

    UnloadIdColorMap();
    //LoadIdColorMap();

    UnloadLabelIdMap();
    //LoadLabelIdMap();

	UnloadIdConfidenceMap();
	//LoadIdConfidenceMap();
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

    UnloadLabelIdMap();
    LoadLabelIdMap();

	UnloadIdConfidenceMap();
	LoadIdConfidenceMap();
}

void TileManager::UnloadSegmentation()
{
    mTileManager->UnloadSegmentation();

    mTileServer->UnloadSegmentation();

    UnloadIdColorMap();
    //LoadIdColorMap();

    UnloadLabelIdMap();
    //LoadLabelIdMap();

	UnloadIdConfidenceMap();
	//LoadIdConfidenceMap();
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

void TileManager::LoadOverTile( TiledDatasetView^ tiledDatasetView )
{
    mTileManager->LoadOverTile( tiledDatasetView->ToNative() );
}

void TileManager::LoadTiles( TiledDatasetView^ tiledDatasetView, int wOffset )
{
    mTileManager->LoadTiles( tiledDatasetView->ToNative(), wOffset );
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

void TileManager::SortSegmentInfoByType( bool reverse )
{
	mTileManager->SortSegmentInfoByType( reverse );
}

void TileManager::SortSegmentInfoBySubType( bool reverse )
{
	mTileManager->SortSegmentInfoBySubType( reverse );
}

void TileManager::RemapSegmentLabel( unsigned int fromSegId, unsigned int toSegId, bool ignoreLocks )
{
	mTileManager->RemapSegmentLabel( fromSegId, toSegId, ignoreLocks );
}

void TileManager::LockSegmentLabel( unsigned int segId )
{
	mTileManager->LockSegmentLabel( segId );
}

void TileManager::UnlockSegmentLabel( unsigned int segId )
{
	mTileManager->UnlockSegmentLabel( segId );
}

void TileManager::SetSegmentType( unsigned int segId, String^ newType )
{
	mTileManager->SetSegmentType( segId, msclr::interop::marshal_as < std::string >( newType ) );
}

void TileManager::SetSegmentSubType( unsigned int segId, String^ newSubType )
{
	mTileManager->SetSegmentSubType( segId, msclr::interop::marshal_as < std::string >( newSubType ) );
}

unsigned int TileManager::GetSegmentInfoCount()
{
	return mTileManager->GetSegmentInfoCount();
}

unsigned int TileManager::GetSegmentInfoCurrentListLocation( unsigned int segId )
{
	return mTileManager->GetSegmentInfoCurrentListLocation( segId );
}

Collections::Generic::IList< SegmentInfo^ >^ TileManager::GetSegmentInfoRange( int begin, int end )
{
    std::list< Native::SegmentInfo > segmentInfoPage = mTileManager->GetSegmentInfoRange( begin, end );

    Collections::Generic::List< SegmentInfo^ >^ interopSegmentInfoPage = gcnew Collections::Generic::List< SegmentInfo^ >();

	for ( std::list< Native::SegmentInfo >::iterator segIt = segmentInfoPage.begin(); segIt != segmentInfoPage.end(); ++segIt )
	{
        interopSegmentInfoPage->Add( gcnew SegmentInfo( *segIt, mTileManager->GetSegmentationLabelColorString( segIt->id ) ) );
	}

    return interopSegmentInfoPage;
}

SegmentInfo^ TileManager::GetSegmentInfo( unsigned int segId )
{
	Native::SegmentInfo segInfo = mTileManager->GetSegmentInfo( segId );
	return gcnew SegmentInfo( segInfo, mTileManager->GetSegmentationLabelColorString( segInfo.id ) );
}

Collections::Generic::IList< unsigned int >^ TileManager::GetRemappedChildren( unsigned int segId )
{
	std::set< unsigned int > remappedChildren = mTileManager->GetRemappedChildren( segId );

	Collections::Generic::List< unsigned int >^ interopRemappedChildren = gcnew Collections::Generic::List< unsigned int >();

	for ( std::set< unsigned int >::iterator childIt = remappedChildren.begin(); childIt != remappedChildren.end(); ++childIt )
	{
		interopRemappedChildren->Add( *childIt );
	}

	return interopRemappedChildren;
}

SlimDX::Direct3D11::ShaderResourceView^ TileManager::GetIdColorMap()
{
    return mIdColorMap;
}

SlimDX::Direct3D11::ShaderResourceView^ TileManager::GetLabelIdMap()
{
    return mLabelIdMap;
}

SlimDX::Direct3D11::ShaderResourceView^ TileManager::GetIdConfidenceMap()
{
    return mIdConfidenceMap;
}

unsigned int TileManager::GetSegmentationLabelId( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    return mTileManager->GetSegmentationLabelId( tiledDatasetView->ToNative(), pDataSpaceFloat3 );
}

Vector3 TileManager::GetSegmentationLabelColor( unsigned int segId )
{
    MojoInt3 labelColor = mTileManager->GetSegmentationLabelColor( segId );
    return Vector3( (float)labelColor.x, (float)labelColor.y, (float)labelColor.z );
}

String^ TileManager::GetSegmentationLabelColorString( unsigned int segId )
{
    return msclr::interop::marshal_as< String^ >( mTileManager->GetSegmentationLabelColorString( segId ) );
}

Vector3 TileManager::GetSegmentCentralTileLocation( unsigned int segId )
{
    Core::MojoInt3 location = mTileManager->GetSegmentCentralTileLocation( segId );
    return Vector3( (float)location.x, (float)location.y, (float)location.z );
}

Vector4 TileManager::GetSegmentZTileBounds( unsigned int segId, int zIndex )
{
    Core::MojoInt4 location = mTileManager->GetSegmentZTileBounds( segId, zIndex );
    return Vector4( (float)location.x, (float)location.y, (float)location.z, (float)location.w );
}

void TileManager::ReplaceSegmentationLabel( unsigned int oldId, unsigned int newId, bool ignoreLocks )
{
    mTileManager->ReplaceSegmentationLabel( oldId, newId, ignoreLocks );
}

void TileManager::ReplaceSegmentationLabelCurrentSlice( unsigned int oldId, unsigned int newId, TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, bool ignoreLocks )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->ReplaceSegmentationLabelCurrentSlice( oldId, newId, pDataSpaceFloat3, ignoreLocks );
}

void TileManager::ReplaceSegmentationLabelCurrentConnectedComponent( unsigned int oldId, unsigned int newId, TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, bool ignoreLocks )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->ReplaceSegmentationLabelCurrentConnectedComponent( oldId, newId, pDataSpaceFloat3, ignoreLocks );
}

void TileManager::DrawSplit( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
	MojoInt3 zoomLevel = mTileManager->GetZoomLevel( tiledDatasetView->ToNative() );
    mTileManager->DrawSplit( pDataSpaceFloat3, radius * (float) pow( 2.0, std::min( zoomLevel.x, zoomLevel.y ) ) );
}

void TileManager::DrawErase( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
	MojoInt3 zoomLevel = mTileManager->GetZoomLevel( tiledDatasetView->ToNative() );
    mTileManager->DrawErase( pDataSpaceFloat3, radius * (float) pow( 2.0, std::min( zoomLevel.x, zoomLevel.y ) ) );
}

void TileManager::DrawRegionA( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
	MojoInt3 zoomLevel = mTileManager->GetZoomLevel( tiledDatasetView->ToNative() );
    mTileManager->DrawRegionA( pDataSpaceFloat3, radius * (float) pow( 2.0, std::min( zoomLevel.x, zoomLevel.y ) ) );
}

void TileManager::DrawRegionB( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace, float radius )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
	MojoInt3 zoomLevel = mTileManager->GetZoomLevel( tiledDatasetView->ToNative() );
    mTileManager->DrawRegionB( pDataSpaceFloat3, radius * (float) pow( 2.0, std::min( zoomLevel.x, zoomLevel.y ) ) );
}

void TileManager::AddSplitSource( TiledDatasetView^ tiledDatasetView, Vector3^ pDataSpace )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->AddSplitSource( pDataSpaceFloat3 );
}

void TileManager::RemoveSplitSource()
{
    mTileManager->RemoveSplitSource();
}

void TileManager::ResetSplitState( Vector3^ pDataSpace )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->ResetSplitState( pDataSpaceFloat3 );
}

void TileManager::PrepForSplit( unsigned int segId, Vector3^ pDataSpace, bool applyBlur )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->PrepForSplit( segId, pDataSpaceFloat3, applyBlur );
}

void TileManager::FindBoundaryJoinPoints2D( unsigned int segId, Vector3^ pDataSpace  )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->FindBoundaryJoinPoints2D( segId, pDataSpaceFloat3 );
}

void TileManager::FindBoundaryWithinRegion2D( unsigned int segId, Vector3^ pDataSpace  )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->FindBoundaryWithinRegion2D( segId, pDataSpaceFloat3 );
}

void TileManager::FindBoundaryBetweenRegions2D( unsigned int segId, Vector3^ pDataSpace  )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->FindBoundaryBetweenRegions2D( segId, pDataSpaceFloat3 );
}

int TileManager::CompletePointSplit( unsigned int segId, Vector3^ pDataSpace, bool applyBlur, bool ignoreLocks )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    return mTileManager->CompletePointSplit( segId, pDataSpaceFloat3, applyBlur, ignoreLocks );
}

int TileManager::CompleteDrawSplit( unsigned int segId, Vector3^ pDataSpace, bool applyBlur, bool join3D, int splitStartZ, bool ignoreLocks )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    return mTileManager->CompleteDrawSplit( segId, pDataSpaceFloat3, applyBlur, join3D, splitStartZ, ignoreLocks );
}

void TileManager::RecordSplitState( unsigned int segId, Vector3^ pDataSpace )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->RecordSplitState( segId, pDataSpaceFloat3 );
}

void TileManager::PredictSplit( unsigned int segId, Vector3^ pDataSpace, float radius )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->PredictSplit( segId, pDataSpaceFloat3, radius );
}

void TileManager::ResetAdjustState( Vector3^ pDataSpace )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->ResetAdjustState( pDataSpaceFloat3 );
}

void TileManager::PrepForAdjust( unsigned int segId, Vector3^ pDataSpace )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->PrepForAdjust( segId, pDataSpaceFloat3 );
}

void TileManager::CommitAdjustChange( unsigned int segId, Vector3^ pDataSpace, bool ignoreLocks )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->CommitAdjustChange( segId, pDataSpaceFloat3, ignoreLocks );
}

void TileManager::ResetDrawMergeState( Vector3^ pDataSpace )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->ResetDrawMergeState( pDataSpaceFloat3 );
}

void TileManager::PrepForDrawMerge( Vector3^ pDataSpace )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    mTileManager->PrepForDrawMerge( pDataSpaceFloat3 );
}

unsigned int TileManager::CommitDrawMerge( Vector3^ pDataSpace, bool ignoreLocks )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    return mTileManager->CommitDrawMerge( pDataSpaceFloat3, ignoreLocks );
}

unsigned int TileManager::CommitDrawMergeCurrentSlice( Vector3^ pDataSpace, bool ignoreLocks )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    return mTileManager->CommitDrawMergeCurrentSlice( pDataSpaceFloat3, ignoreLocks );
}

unsigned int TileManager::CommitDrawMergeCurrentConnectedComponent( Vector3^ pDataSpace, bool ignoreLocks )
{
    MojoFloat3 pDataSpaceFloat3 = MojoFloat3( pDataSpace->X, pDataSpace->Y, pDataSpace->Z );
    return mTileManager->CommitDrawMergeCurrentConnectedComponent( pDataSpaceFloat3, ignoreLocks );
}

unsigned int TileManager::GetNewId()
{
    return mTileManager->GetNewId();
}

float TileManager::GetCurrentOperationProgress()
{
	return mTileManager->GetCurrentOperationProgress();
}

void TileManager::UndoChange()
{
    mTileManager->UndoChange();
}

void TileManager::RedoChange()
{
    mTileManager->RedoChange();
}

void TileManager::TempSaveAndClearFileSystemTileCache()
{
	mTileManager->TempSaveAndClearFileSystemTileCache();
}

void TileManager::ClearFileSystemTileCache()
{
	mTileManager->ClearFileSystemTileCache();
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
	std::vector< Native::TileCacheEntry >& tileCache = mTileManager->GetTileCache();

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

void TileManager::LoadLabelIdMap()
{
    ID3D11ShaderResourceView* labelIdMap =  mTileManager->GetLabelIdMap();

    if ( labelIdMap != NULL )
    {
        RELEASE_ASSERT( mLabelIdMap == nullptr );
        mLabelIdMap = ShaderResourceView::FromPointer( IntPtr( labelIdMap ) );
    }
}

void TileManager::UnloadLabelIdMap()
{
    if ( mLabelIdMap != nullptr )
    {
        delete mLabelIdMap;
        mLabelIdMap = nullptr;
    }
}

void TileManager::LoadIdConfidenceMap()
{
    ID3D11ShaderResourceView* idConfidenceMap =  mTileManager->GetIdConfidenceMap();

    if ( idConfidenceMap != NULL )
    {
        RELEASE_ASSERT( mIdConfidenceMap == nullptr );
        mIdConfidenceMap = ShaderResourceView::FromPointer( IntPtr( idConfidenceMap ) );
    }
}

void TileManager::UnloadIdConfidenceMap()
{
    if ( mIdConfidenceMap != nullptr )
    {
        delete mIdConfidenceMap;
        mIdConfidenceMap = nullptr;
    }
}

}
}