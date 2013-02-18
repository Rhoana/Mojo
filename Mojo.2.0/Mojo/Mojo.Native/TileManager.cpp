#include "TileManager.hpp"

#include "Mojo.Core/Stl.hpp"

//#include "Mojo.Core/OpenCV.hpp"
#include "Mojo.Core/ForEach.hpp"
#include "Mojo.Core/D3D11CudaTexture.hpp"
#include "Mojo.Core/Index.hpp"

#include "TileCacheEntry.hpp"

using namespace Mojo::Core;

namespace Mojo
{
namespace Native
{

TileManager::TileManager( ID3D11Device* d3d11Device, ID3D11DeviceContext* d3d11DeviceContext, ITileServer* tileServer, Core::PrimitiveMap constParameters ) :
    mIdColorMapBuffer            ( NULL ),
    mIdColorMapShaderResourceView( NULL ),
    mLabelIdMapBuffer            ( NULL ),
    mLabelIdMapShaderResourceView( NULL ),
    mIdConfidenceMapBuffer            ( NULL ),
    mIdConfidenceMapShaderResourceView( NULL ),
    mTileServer                  ( tileServer ),
    mConstParameters             ( constParameters ),
    mIsTiledDatasetLoaded        ( false ),
	mIsSegmentationLoaded        ( false )
{
    mD3D11Device = d3d11Device;
    mD3D11Device->AddRef();

    mD3D11DeviceContext = d3d11DeviceContext;
    mD3D11DeviceContext->AddRef();
}

TileManager::~TileManager()
{
    mD3D11DeviceContext->Release();
    mD3D11DeviceContext = NULL;

    mD3D11Device->Release();
    mD3D11Device = NULL;
}

void TileManager::LoadTiledDataset( TiledDatasetDescription& tiledDatasetDescription )
{
    switch ( tiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).dxgiFormat )
    {
        case DXGI_FORMAT_R8_UNORM:
            LoadTiledDatasetInternal< uchar1 >( tiledDatasetDescription );
            break;

        default:
            RELEASE_ASSERT( 0 );
            break;
    }
}

void TileManager::UnloadTiledDataset()
{
    UnloadTiledDatasetInternal();
}

bool TileManager::IsTiledDatasetLoaded()
{
    return mIsTiledDatasetLoaded;
}

void TileManager::LoadSegmentation( TiledDatasetDescription& tiledDatasetDescription )
{
    switch ( tiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).dxgiFormat )
    {
        case DXGI_FORMAT_R8_UNORM:
            LoadSegmentationInternal< uchar1 >( tiledDatasetDescription );
            break;

        default:
            RELEASE_ASSERT( 0 );
            break;
    }
}

void TileManager::UnloadSegmentation()
{
    UnloadSegmentationInternal();
}

bool TileManager::IsSegmentationLoaded()
{
    return mIsSegmentationLoaded;
}

void TileManager::SaveSegmentation()
{
    mTileServer->SaveSegmentation();
}

void TileManager::SaveSegmentationAs( std::string savePath )
{
    mTileServer->SaveSegmentationAs( savePath );
}

void TileManager::AutosaveSegmentation()
{
    mTileServer->AutosaveSegmentation();
}

void TileManager::DeleteTempFiles()
{
    mTileServer->DeleteTempFiles();
}

void TileManager::Update()
{
}

void TileManager::LoadTiles( const TiledDatasetView& tiledDatasetView )
{
    //
    // assume that all cache entries can be discarded unless explicitly marked otherwise
    //
    for ( int cacheIndex = 0; cacheIndex < DEVICE_TILE_CACHE_SIZE; cacheIndex++ )
    {
        mTileCache[ cacheIndex ].keepState = TileCacheEntryKeepState_CanDiscard;
        mTileCache[ cacheIndex ].active    = false;
    }

    std::list< MojoInt4 > tileIndices = GetTileIndicesIntersectedByView( tiledDatasetView );

    //
    // explicitly mark all previously loaded cache entries that intersect the current view as cache entries to keep
    //
    MOJO_FOR_EACH( MojoInt4 tileIndex, tileIndices )
    {
        int cacheIndex = mTileCachePageTable( tileIndex.w, tileIndex.z, tileIndex.y, tileIndex.x );

        if ( cacheIndex != TILE_CACHE_BAD_INDEX )
        {
            mTileCache[ cacheIndex ].keepState = TileCacheEntryKeepState_MustKeep;
            mTileCache[ cacheIndex ].active    = true;
        }
    }

    //
    // for each tile that intersects the current view but is not loaded, load it and overwrite a cache entry that can be discarded
    //
    MOJO_FOR_EACH( MojoInt4 tileIndex, tileIndices )
    {
        int cacheIndex = mTileCachePageTable( tileIndex.w, tileIndex.z, tileIndex.y, tileIndex.x );

        //
        // if the tile is not loaded...
        //
        if ( cacheIndex == TILE_CACHE_BAD_INDEX )
        {
            //
            // find another cache entry to store the tile 
            //
            int newCacheIndex = mTileCacheSearchStart;
            int lastCacheIndex = ( DEVICE_TILE_CACHE_SIZE + mTileCacheSearchStart - 1 ) % DEVICE_TILE_CACHE_SIZE;

            for (; mTileCache[ newCacheIndex ].keepState != TileCacheEntryKeepState_CanDiscard; newCacheIndex = ( newCacheIndex + 1 ) % DEVICE_TILE_CACHE_SIZE )
            {
                //
                // check if we have run out of tiles
                //
                RELEASE_ASSERT ( newCacheIndex != lastCacheIndex );
            }

            mTileCacheSearchStart = ( newCacheIndex + 1 ) % DEVICE_TILE_CACHE_SIZE;

            RELEASE_ASSERT( !mTileCache[ newCacheIndex ].active );

            //Core::Printf( "Replacing tile ", newCacheIndex, " in the device cache.");

            //
            // get the new cache entry's index in tile space
            //
            MojoInt4 indexTileSpace = mTileCache[ newCacheIndex ].indexTileSpace;
            MojoInt4 tempTileIndex = MojoInt4( indexTileSpace.x, indexTileSpace.y, indexTileSpace.z, indexTileSpace.w );

            //
            // if the new cache entry refers to a tile that is already loaded...
            //
            if ( tempTileIndex.x != TILE_CACHE_PAGE_TABLE_BAD_INDEX ||
                 tempTileIndex.y != TILE_CACHE_PAGE_TABLE_BAD_INDEX ||
                 tempTileIndex.z != TILE_CACHE_PAGE_TABLE_BAD_INDEX ||
                 tempTileIndex.w != TILE_CACHE_PAGE_TABLE_BAD_INDEX )
            {
                RELEASE_ASSERT( tempTileIndex.x != TILE_CACHE_PAGE_TABLE_BAD_INDEX &&
                                tempTileIndex.y != TILE_CACHE_PAGE_TABLE_BAD_INDEX &&
                                tempTileIndex.z != TILE_CACHE_PAGE_TABLE_BAD_INDEX &&
                                tempTileIndex.w != TILE_CACHE_PAGE_TABLE_BAD_INDEX );

                //
                // mark the tile as not being loaded any more
                //
                mTileCachePageTable( tempTileIndex.w, tempTileIndex.z, tempTileIndex.y, tempTileIndex.x ) = TILE_CACHE_BAD_INDEX;
            }

            //
            // load image data into host memory
            //
            Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions = mTileServer->LoadTile( tileIndex );

            //
            // load the new data into into device memory for the new cache entry
            //
            mTileCache[ newCacheIndex ].d3d11CudaTextures.Get( "SourceMap" )->Update( volumeDescriptions.Get( "SourceMap" ) );

			if ( IsSegmentationLoaded() )
			{
				if ( volumeDescriptions.GetHashMap().find( "IdMap" ) == volumeDescriptions.GetHashMap().end() )
				{
					Core::Printf( "Warning: Segmentation is loaded, but volume description is missing an IdMap." );
				}
				else
				{
					mTileCache[ newCacheIndex ].d3d11CudaTextures.Get( "IdMap" )->Update( volumeDescriptions.Get( "IdMap" ) );

                    //TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );
					//int idNumVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTileX * tiledVolumeDescription.numVoxelsPerTileY * tiledVolumeDescription.numVoxelsPerTileZ;

					//Core::Thrust::MemcpyHostToDevice( mTileCache[ newCacheIndex ].deviceVectors.Get< int >( "IdMap" ), volumeDescriptions.Get( "IdMap" ).data, idNumVoxelsPerTile );
				}

				if ( volumeDescriptions.GetHashMap().find( "OverlayMap" ) != volumeDescriptions.GetHashMap().end() )
				{
					mTileCache[ newCacheIndex ].d3d11CudaTextures.Get( "OverlayMap" )->Update( volumeDescriptions.Get( "OverlayMap" ) );

                    //TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "OverlayMap" );
					//int idNumVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTileX * tiledVolumeDescription.numVoxelsPerTileY * tiledVolumeDescription.numVoxelsPerTileZ;

					//Core::Thrust::MemcpyHostToDevice( mTileCache[ newCacheIndex ].deviceVectors.Get< int >( "OverlayMap" ), volumeDescriptions.Get( "OverlayMap" ).data, idNumVoxelsPerTile );
				}
			}

            //
            // unload image data from from host memory
            //
            mTileServer->UnloadTile( tileIndex );

            //
            // update tile cache state for the new cache entry
            //
            MojoFloat3 extentDataSpace =
                MojoFloat3(
                    mConstParameters.Get< int >( "TILE_SIZE_X" ) * (float)pow( 2.0, tileIndex.w ),
                    mConstParameters.Get< int >( "TILE_SIZE_Y" ) * (float)pow( 2.0, tileIndex.w ),
                    mConstParameters.Get< int >( "TILE_SIZE_Z" ) * (float)pow( 2.0, tileIndex.w ) );

            MojoFloat3 centerDataSpace =
                MojoFloat3(
                    ( tileIndex.x + 0.5f ) * extentDataSpace.x,
                    ( tileIndex.y + 0.5f ) * extentDataSpace.y,
                    ( tileIndex.z + 0.5f ) * extentDataSpace.z );

            mTileCache[ newCacheIndex ].keepState       = TileCacheEntryKeepState_MustKeep;
            mTileCache[ newCacheIndex ].active          = true;
            mTileCache[ newCacheIndex ].indexTileSpace  = tileIndex;
            mTileCache[ newCacheIndex ].centerDataSpace = centerDataSpace;
            mTileCache[ newCacheIndex ].extentDataSpace = extentDataSpace;

            //
            // mark the new location in tile space as being loaded into the cache
            //
            RELEASE_ASSERT( mTileCachePageTable( tileIndex.w, tileIndex.z, tileIndex.y, tileIndex.x ) == TILE_CACHE_BAD_INDEX );

            mTileCachePageTable( tileIndex.w, tileIndex.z, tileIndex.y, tileIndex.x ) = newCacheIndex;
        }
    }
}

boost::array< TileCacheEntry, DEVICE_TILE_CACHE_SIZE >& TileManager::GetTileCache()
{
    return mTileCache;
}

ID3D11ShaderResourceView* TileManager::GetIdColorMap()
{
    return mIdColorMapShaderResourceView;
}

ID3D11ShaderResourceView* TileManager::GetLabelIdMap()
{
    return mLabelIdMapShaderResourceView;
}

ID3D11ShaderResourceView* TileManager::GetIdConfidenceMap()
{
    return mIdConfidenceMapShaderResourceView;
}

unsigned int TileManager::GetSegmentationLabelId( const TiledDatasetView& tiledDatasetView, MojoFloat3 pDataSpace )
{
	MojoInt3   zoomLevel = GetZoomLevel( tiledDatasetView );
	MojoFloat4 pointTileSpace;
	MojoInt4   tileIndex;
	unsigned int segmentId = 0;

	if ( mIsSegmentationLoaded )
	{

		GetIndexTileSpace( zoomLevel, pDataSpace, pointTileSpace, tileIndex );

        TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

		MojoInt3 numVoxels = MojoInt3( tiledVolumeDescription.numVoxelsX, tiledVolumeDescription.numVoxelsY, tiledVolumeDescription.numVoxelsZ );
        MojoInt3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile();
		MojoInt3 pVoxelSpace = GetIndexVoxelSpace( pointTileSpace, numVoxelsPerTile );

		//
		// Check for overflow
		//
		if ( pVoxelSpace.x >= 0 && pVoxelSpace.x < numVoxels.x &&
			pVoxelSpace.y >= 0 && pVoxelSpace.y < numVoxels.y &&
			pVoxelSpace.z >= 0 && pVoxelSpace.z < numVoxels.z )
		{
			MojoInt3 offsetVoxelSpace   = GetOffsetVoxelSpace( pointTileSpace, tileIndex, numVoxelsPerTile );
			int  offsetVoxelSpace1D = Core::Index3DToIndex1D( offsetVoxelSpace, numVoxelsPerTile );

			Core::HashMap< std::string, Core::VolumeDescription > thisTile = mTileServer->LoadTile( tileIndex );

			segmentId = mTileServer->GetSegmentInfoManager()->GetIdForLabel( ( (int*) thisTile.Get( "IdMap" ).data )[ offsetVoxelSpace1D ] );

			mTileServer->UnloadTile( tileIndex );

		}
	}

	return segmentId;

}

void TileManager::SortSegmentInfoById( bool reverse )
{
	mTileServer->SortSegmentInfoById( reverse );
}

void TileManager::SortSegmentInfoByName( bool reverse )
{
	mTileServer->SortSegmentInfoByName( reverse );
}

void TileManager::SortSegmentInfoBySize( bool reverse )
{
	mTileServer->SortSegmentInfoBySize( reverse );
}

void TileManager::SortSegmentInfoByConfidence( bool reverse )
{
	mTileServer->SortSegmentInfoByConfidence( reverse );
}

void TileManager::RemapSegmentLabel( unsigned int fromSegId, unsigned int toSegId )
{
	Core::Printf( "From ", fromSegId, " before -> ", (*mLabelIdMap)( fromSegId ), "." );	
	Core::Printf( "To ", toSegId, " before -> ", (*mLabelIdMap)( toSegId ), "." );

	std::set< unsigned int > fromSegIds;
	fromSegIds.insert( fromSegId );
	mTileServer->RemapSegmentLabels( fromSegIds, toSegId );

	Core::Printf( "From ", fromSegId, " after -> ", (*mLabelIdMap)( fromSegId ), "." );
	Core::Printf( "To ", toSegId, " after -> ", (*mLabelIdMap)( toSegId ), "." );

	UpdateLabelIdMap( fromSegId );

}

void TileManager::UpdateLabelIdMap( unsigned int fromSegId )
{
    //
    // Update label id map shader buffer
    //
    uint1 labelIdMapEntry = make_uint1( (*mLabelIdMap)( fromSegId ) );

    D3D11_BOX updateBox;
    ZeroMemory( &updateBox, sizeof( D3D11_BOX ) );

    updateBox.left = fromSegId * sizeof( uint1 );
    updateBox.top = 0;
    updateBox.front = 0;
    updateBox.right = ( fromSegId + 1 ) * sizeof( uint1 );
    updateBox.bottom = 1;
    updateBox.back = 1;

    mD3D11DeviceContext->UpdateSubresource(
    mLabelIdMapBuffer,
    0,
    &updateBox,
    &labelIdMapEntry,
    (UINT) mLabelIdMap->shape( 0 ) * sizeof( uint1 ),
    (UINT) mLabelIdMap->shape( 0 ) * sizeof( uint1 ) );
}

void TileManager::LockSegmentLabel( unsigned int segId )
{
	mTileServer->LockSegmentLabel( segId );

    //
    // Update confidence map shader buffer
    //
    uchar1 idConfidenceMapEntry = make_uchar1( (*mIdConfidenceMap)( segId ) );

    D3D11_BOX updateBox;
    ZeroMemory( &updateBox, sizeof( D3D11_BOX ) );

    updateBox.left = segId;
    updateBox.top = 0;
    updateBox.front = 0;
    updateBox.right = segId + 1;
    updateBox.bottom = 1;
    updateBox.back = 1;

    mD3D11DeviceContext->UpdateSubresource(
    mIdConfidenceMapBuffer,
    0,
    &updateBox,
    &idConfidenceMapEntry,
    (UINT) mIdConfidenceMap->shape( 0 ) * sizeof( uchar1 ),
    (UINT) mIdConfidenceMap->shape( 0 ) * sizeof( uchar1 ) );

}

void TileManager::UnlockSegmentLabel( unsigned int segId )
{
	mTileServer->UnlockSegmentLabel( segId );

    //
    // Update confidence map shader buffer
    //
    uchar1 idConfidenceMapEntry = make_uchar1( 0 );

    D3D11_BOX updateBox;
    ZeroMemory( &updateBox, sizeof( D3D11_BOX ) );

    updateBox.left = segId;
    updateBox.top = 0;
    updateBox.front = 0;
    updateBox.right = segId + 1;
    updateBox.bottom = 1;
    updateBox.back = 1;

    mD3D11DeviceContext->UpdateSubresource(
    mIdConfidenceMapBuffer,
    0,
    &updateBox,
    &idConfidenceMapEntry,
    (UINT) mIdConfidenceMap->shape( 0 ) * sizeof( uchar1 ),
    (UINT) mIdConfidenceMap->shape( 0 ) * sizeof( uchar1 ) );

}

unsigned int TileManager::GetSegmentInfoCount()
{
	return mTileServer->GetSegmentInfoCount();
}

unsigned int TileManager::GetSegmentInfoCurrentListLocation( unsigned int segId )
{
	return mTileServer->GetSegmentInfoCurrentListLocation( segId );
}

std::list< SegmentInfo > TileManager::GetSegmentInfoRange( int begin, int end )
{
	return mTileServer->GetSegmentInfoRange( begin, end );
}

MojoInt3 TileManager::GetSegmentationLabelColor( unsigned int segId )
{
	if ( mIdColorMap->size() > 0 )
	{
		int index = segId % mIdColorMap->shape( 0 );
		return MojoInt3( (*mIdColorMap)( index, 0 ), (*mIdColorMap)( index, 1 ), (*mIdColorMap)( index, 2 ) );
	}
    return MojoInt3();
}

std::string TileManager::GetSegmentationLabelColorString( unsigned int segId )
{
	if ( mIdColorMap->size() > 0 )
	{
		int index = segId % mIdColorMap->shape( 0 );

	    std::ostringstream colorConverter;
	    colorConverter << std::setfill( '0' ) << std::hex;
		colorConverter << std::setw( 1 ) << "#";
		colorConverter << std::setw( 2 ) << (int)(*mIdColorMap)( index, 0 );
		colorConverter << std::setw( 2 ) << (int)(*mIdColorMap)( index, 1 );
		colorConverter << std::setw( 2 ) << (int)(*mIdColorMap)( index, 2 );

		return colorConverter.str();
	}
    return "#000000";
}

MojoInt3 TileManager::GetSegmentCentralTileLocation( unsigned int segId )
{
    return mTileServer->GetSegmentCentralTileLocation( segId );
}

MojoInt4 TileManager::GetSegmentZTileBounds( unsigned int segId, int zIndex )
{
    return mTileServer->GetSegmentZTileBounds( segId, zIndex );
}

void TileManager::ReplaceSegmentationLabel( unsigned int oldId, unsigned int newId )
{
    mTileServer->ReplaceSegmentationLabel( oldId, newId );

    ReloadTileCache();
}

void TileManager::ReplaceSegmentationLabelCurrentSlice( unsigned int oldId, unsigned int newId, MojoFloat3 pDataSpace )
{
    mTileServer->ReplaceSegmentationLabelCurrentSlice( oldId, newId, pDataSpace );

    ReloadTileCache();
}

void TileManager::DrawSplit( MojoFloat3 pointTileSpace, float radius )
{
    mTileServer->DrawSplit( pointTileSpace, radius );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::DrawErase( MojoFloat3 pointTileSpace, float radius )
{
    mTileServer->DrawErase( pointTileSpace, radius );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::DrawRegionA( MojoFloat3 pointTileSpace, float radius )
{
    mTileServer->DrawRegionA( pointTileSpace, radius );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::DrawRegionB( MojoFloat3 pointTileSpace, float radius )
{
    mTileServer->DrawRegionB( pointTileSpace, radius );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::AddSplitSource( MojoFloat3 pointTileSpace )
{
    mTileServer->AddSplitSource( pointTileSpace );
}

void TileManager::RemoveSplitSource()
{
    mTileServer->RemoveSplitSource();
}

void TileManager::ResetSplitState( MojoFloat3 pointTileSpace )
{
    mTileServer->ResetSplitState();

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::PrepForSplit( unsigned int segId, MojoFloat3 pointTileSpace )
{
    mTileServer->PrepForSplit( segId, pointTileSpace );

	ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::FindBoundaryJoinPoints2D( unsigned int segId, MojoFloat3 pointTileSpace  )
{
    mTileServer->FindBoundaryJoinPoints2D( segId );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::FindBoundaryWithinRegion2D( unsigned int segId, MojoFloat3 pointTileSpace  )
{
    mTileServer->FindBoundaryWithinRegion2D( segId );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::FindBoundaryBetweenRegions2D( unsigned int segId, MojoFloat3 pointTileSpace  )
{
    mTileServer->FindBoundaryBetweenRegions2D( segId );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

int TileManager::CompletePointSplit( unsigned int segId, MojoFloat3 pointTileSpace )
{
    int newId = mTileServer->CompletePointSplit( segId, pointTileSpace );

    mTileServer->PrepForSplit( segId, pointTileSpace );

    ReloadTileCache();

	return newId;
}

int TileManager::CompleteDrawSplit( unsigned int segId, MojoFloat3 pointTileSpace, bool join3D, int splitStartZ )
{
    int newId = mTileServer->CompleteDrawSplit( segId, pointTileSpace, join3D, splitStartZ );

    mTileServer->PrepForSplit( segId, pointTileSpace );

    ReloadTileCache();

	return newId;
}

void TileManager::RecordSplitState( unsigned int segId, MojoFloat3 pointTileSpace )
{
    mTileServer->RecordSplitState( segId, pointTileSpace );
}

void TileManager::PredictSplit( unsigned int segId, MojoFloat3 pointTileSpace, float radius )
{
    mTileServer->PredictSplit( segId, pointTileSpace, radius );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::ResetAdjustState( MojoFloat3 pointTileSpace )
{
    mTileServer->ResetAdjustState();

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::PrepForAdjust( unsigned int segId, MojoFloat3 pointTileSpace )
{
    mTileServer->PrepForAdjust( segId, pointTileSpace );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::CommitAdjustChange( unsigned int segId, MojoFloat3 pointTileSpace )
{
    mTileServer->CommitAdjustChange( segId, pointTileSpace );

    mTileServer->PrepForAdjust( segId, pointTileSpace );

    ReloadTileCache();
}

void TileManager::ResetDrawMergeState( MojoFloat3 pointTileSpace )
{
    mTileServer->ResetDrawMergeState();

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::PrepForDrawMerge( MojoFloat3 pointTileSpace )
{
    mTileServer->PrepForDrawMerge( pointTileSpace );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

unsigned int TileManager::CommitDrawMerge( MojoFloat3 pointTileSpace )
{
	std::set< unsigned int > remapIds = mTileServer->GetDrawMergeIds( pointTileSpace );

	unsigned int newId = mTileServer->CommitDrawMerge( remapIds, pointTileSpace );

	for ( std::set< unsigned int >::iterator updateIt = remapIds.begin(); updateIt != remapIds.end(); ++updateIt )
	{
		UpdateLabelIdMap( *updateIt );
	}

    mTileServer->PrepForDrawMerge( pointTileSpace );

	ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );

	return newId;

}

void TileManager::ReplaceSegmentationLabelCurrentConnectedComponent( unsigned int oldId, unsigned int newId, MojoFloat3 pDataSpace )
{
    mTileServer->ReplaceSegmentationLabelCurrentConnectedComponent( oldId, newId, pDataSpace );

    ReloadTileCache();
}

void TileManager::UndoChange()
{
	std::list< unsigned int > remappedIds = mTileServer->UndoChange();

	for ( std::list< unsigned int >::iterator updateIt = remappedIds.begin(); updateIt != remappedIds.end(); ++updateIt )
	{
		UpdateLabelIdMap( *updateIt );
	}

    ReloadTileCache();
}

void TileManager::RedoChange()
{
    std::list< unsigned int > remappedIds = mTileServer->RedoChange();

	for ( std::list< unsigned int >::iterator updateIt = remappedIds.begin(); updateIt != remappedIds.end(); ++updateIt )
	{
		UpdateLabelIdMap( *updateIt );
	}

    ReloadTileCache();
}


void TileManager::SaveAndClearFileSystemTileCache()
{
    mTileServer->SaveAndClearFileSystemTileCache();
}

void TileManager::UnloadTiledDatasetInternal()
{
	if ( mIsSegmentationLoaded )
	{
		UnloadSegmentationInternal();
	}
    if ( mIsTiledDatasetLoaded )
    {
        //
        // reset all state
        //
        mIsTiledDatasetLoaded    = false;
        mTiledDatasetDescription = TiledDatasetDescription();

        //
        // reset page table
        //
        mTileCachePageTable = marray::Marray< int >( 0 );

        //
        // output memory stats to the console
        //
        //size_t freeMemory, totalMemory;
        //CUresult     memInfoResult;
        //memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
        //RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );

        IDXGIDevice * pDXGIDevice;
        mD3D11Device->QueryInterface(__uuidof(IDXGIDevice), (void **)&pDXGIDevice);
        IDXGIAdapter * pDXGIAdapter;
        pDXGIDevice->GetAdapter(&pDXGIAdapter);
        DXGI_ADAPTER_DESC adapterDesc;
        pDXGIAdapter->GetDesc(&adapterDesc);

        Core::Printf( "\nUnloading tiled dataset...\n" );
        Core::Printf( "\n    Before freeing GPU memory:\n",
            "        Free memory:  ", (unsigned int) adapterDesc.DedicatedVideoMemory  / ( 1024 * 1024 ), " MBytes.\n" );
            //"        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );

        //
        // delete textures in the tile cache
        //
        for ( int i = 0; i < DEVICE_TILE_CACHE_SIZE; i++ )
        {
            //mTileCache[ i ].deviceVectors.Clear();

            delete mTileCache[ i ].d3d11CudaTextures.Get( "SourceMap" );
            delete mTileCache[ i ].d3d11CudaTextures.Get( "IdMap" );
            delete mTileCache[ i ].d3d11CudaTextures.Get( "OverlayMap" );

            mTileCache[ i ].d3d11CudaTextures.GetHashMap().clear();
        }

        //
        // output memory stats to the console
        //
        //memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
        //RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );

        pDXGIAdapter->GetDesc(&adapterDesc);

        Core::Printf( "    After freeing GPU memory:\n",
            "        Free memory:  ", (unsigned int) adapterDesc.DedicatedVideoMemory  / ( 1024 * 1024 ), " MBytes.\n" );
            //"        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );
    }
}

void TileManager::UnloadSegmentationInternal()
{
    if ( mIsSegmentationLoaded )
    {
        //
        // reset segmentation state
        //
        mIsSegmentationLoaded    = false;

        //
        // output memory stats to the console
        //
        //size_t freeMemory, totalMemory;
        //CUresult     memInfoResult;
        //memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
        //RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );

        IDXGIDevice * pDXGIDevice;
        mD3D11Device->QueryInterface(__uuidof(IDXGIDevice), (void **)&pDXGIDevice);
        IDXGIAdapter * pDXGIAdapter;
        pDXGIDevice->GetAdapter(&pDXGIAdapter);
        DXGI_ADAPTER_DESC adapterDesc;
        pDXGIAdapter->GetDesc(&adapterDesc);

        Core::Printf( "\nUnloading segmentation...\n" );
        Core::Printf( "\n    Before freeing GPU memory:\n",
            "        Free memory:  ", (unsigned int) adapterDesc.DedicatedVideoMemory  / ( 1024 * 1024 ), " MBytes.\n" );
            //"        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );

        //
        // release id color map
        //
        mIdColorMapShaderResourceView->Release();
        mIdColorMapShaderResourceView = NULL;

        mIdColorMapBuffer->Release();
        mIdColorMapBuffer = NULL;

        //
        // release label id map
        //
        mLabelIdMapShaderResourceView->Release();
        mLabelIdMapShaderResourceView = NULL;

        mLabelIdMapBuffer->Release();
        mLabelIdMapBuffer = NULL;

        //
        // release id lock map
        //
        mIdConfidenceMapShaderResourceView->Release();
        mIdConfidenceMapShaderResourceView = NULL;

        mIdConfidenceMapBuffer->Release();
        mIdConfidenceMapBuffer = NULL;

        //
        // output memory stats to the console
        //
        //memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
        //RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );

        pDXGIAdapter->GetDesc(&adapterDesc);

        Core::Printf( "    After freeing GPU memory:\n",
            "        Free memory:  ", (unsigned int) adapterDesc.DedicatedVideoMemory  / ( 1024 * 1024 ), " MBytes.\n" );
            //"        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );
    }
}

MojoInt3 TileManager::GetZoomLevel( const TiledDatasetView& tiledDatasetView )
{
    //
    // figure out what the current zoom level is
    //
    TiledVolumeDescription tiledvolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" );

    double expZoomLevelXY   = ( tiledDatasetView.extentDataSpace.x * tiledvolumeDescription.numVoxelsPerTileX ) / ( tiledDatasetView.widthNumPixels );

    int    zoomLevelXY      = (int)floor( std::min( (double)( tiledvolumeDescription.numTilesW - 1 ), std::max( 0.0, ( log( expZoomLevelXY ) / log( 2.0 ) ) ) ) );
    int    zoomLevelZ       = 0;

    return MojoInt3( zoomLevelXY, zoomLevelXY, zoomLevelZ );
}

std::list< MojoInt4 > TileManager::GetTileIndicesIntersectedByView( const TiledDatasetView& tiledDatasetView )
{
    std::list< MojoInt4 > tilesIntersectedByCamera;

    //
    // figure out what the current zoom level is
    //
    MojoInt3 zoomLevel   = GetZoomLevel( tiledDatasetView );
    int  zoomLevelXY = std::min( zoomLevel.x, zoomLevel.y );
    int  zoomLevelZ  = zoomLevel.z;

    //
    // figure out how many tiles there are at the current zoom level
    //
    MojoInt4 numTiles              = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles();
    int  numTilesForZoomLevelX = (int)ceil( numTiles.x / pow( 2.0, zoomLevelXY ) );
    int  numTilesForZoomLevelY = (int)ceil( numTiles.y / pow( 2.0, zoomLevelXY ) );
    int  numTilesForZoomLevelZ = (int)ceil( numTiles.z / pow( 2.0, zoomLevelZ ) );

    //
    // figure out the tile size (in data space) at the current zoom level
    //
    int tileSizeDataSpaceX = (int)( mConstParameters.Get< int >( "TILE_SIZE_X" ) * pow( 2.0, zoomLevelXY ) );
    int tileSizeDataSpaceY = (int)( mConstParameters.Get< int >( "TILE_SIZE_Y" ) * pow( 2.0, zoomLevelXY ) );
    int tileSizeDataSpaceZ = (int)( mConstParameters.Get< int >( "TILE_SIZE_Z" ) * pow( 2.0, zoomLevelZ ) );

    //
    // compute the top-left-front and bottom-right-back points (in data space) that are currently in view
    //
    MojoFloat3 topLeftFrontDataSpace =
        MojoFloat3(
            tiledDatasetView.centerDataSpace.x - ( tiledDatasetView.extentDataSpace.x / 2 ),
            tiledDatasetView.centerDataSpace.y - ( tiledDatasetView.extentDataSpace.y / 2 ),
            tiledDatasetView.centerDataSpace.z - ( tiledDatasetView.extentDataSpace.z / 2 ) );

    MojoFloat3 bottomRightBackDataSpace =
        MojoFloat3(
            tiledDatasetView.centerDataSpace.x + ( tiledDatasetView.extentDataSpace.x / 2 ),
            tiledDatasetView.centerDataSpace.y + ( tiledDatasetView.extentDataSpace.y / 2 ),
            tiledDatasetView.centerDataSpace.z + ( tiledDatasetView.extentDataSpace.z / 2 ) );

    //
    // compute the tile space indices for the top-left-front and bottom-right-back points
    //
    MojoInt3 topLeftFrontTileIndex =
        MojoInt3(
            (int)floor( topLeftFrontDataSpace.x / tileSizeDataSpaceX ),
            (int)floor( topLeftFrontDataSpace.y / tileSizeDataSpaceY ),
            (int)floor( topLeftFrontDataSpace.z / tileSizeDataSpaceZ ) );

    MojoInt3 bottomRightBackTileIndex =
        MojoInt3(
            (int)floor( bottomRightBackDataSpace.x / tileSizeDataSpaceX ),
            (int)floor( bottomRightBackDataSpace.y / tileSizeDataSpaceY ),
            (int)floor( bottomRightBackDataSpace.z / tileSizeDataSpaceZ ) );

    //
    // clip the tiles to the appropriate tile space borders
    //
    int minX = std::max( 0,                         topLeftFrontTileIndex.x );
    int maxX = std::min( numTilesForZoomLevelX - 1, bottomRightBackTileIndex.x );
    int minY = std::max( 0,                         topLeftFrontTileIndex.y );
    int maxY = std::min( numTilesForZoomLevelY - 1, bottomRightBackTileIndex.y );
    int minZ = std::max( 0,                         topLeftFrontTileIndex.z );
    int maxZ = std::min( numTilesForZoomLevelZ - 1, bottomRightBackTileIndex.z );

    for ( int z = minZ; z <= maxZ; z++ )
        for ( int y = minY; y <= maxY; y++ )
            for ( int x = minX; x <= maxX; x++ )
                tilesIntersectedByCamera.push_back( MojoInt4( x, y, z, zoomLevelXY ) );

    return tilesIntersectedByCamera;
}

void TileManager::GetIndexTileSpace( MojoInt3 zoomLevel, MojoFloat3 pointDataSpace, MojoFloat4& pointTileSpace, MojoInt4& tileIndex )
{
    int zoomLevelXY = std::min( zoomLevel.x, zoomLevel.y );
    int zoomLevelZ  = zoomLevel.z;

    //
    // figure out the tile size (in data space) at the current zoom level
    //
    int tileSizeDataSpaceX = (int)( mConstParameters.Get< int >( "TILE_SIZE_X" ) * pow( 2.0, zoomLevelXY ) );
    int tileSizeDataSpaceY = (int)( mConstParameters.Get< int >( "TILE_SIZE_Y" ) * pow( 2.0, zoomLevelXY ) );
    int tileSizeDataSpaceZ = (int)( mConstParameters.Get< int >( "TILE_SIZE_Z" ) * pow( 2.0, zoomLevelZ ) );

    //
    // compute the tile space indices for the requested point
    //
    pointTileSpace =
        MojoFloat4(
            (float)pointDataSpace.x / tileSizeDataSpaceX,
            (float)pointDataSpace.y / tileSizeDataSpaceY,
            (float)pointDataSpace.z / tileSizeDataSpaceZ,
            (float)zoomLevelXY );

    tileIndex =
        MojoInt4(
            (int)floor( pointTileSpace.x ),
            (int)floor( pointTileSpace.y ),
            (int)floor( pointTileSpace.z ),
            (int)floor( pointTileSpace.w ) );
}

MojoInt3 TileManager::GetIndexVoxelSpace( MojoFloat4 pointTileSpace, MojoInt3 numVoxelsPerTile )
{
    //
    // compute the index in voxels within the full volume
    //
    MojoInt3 indexVoxelSpace = 
        MojoInt3(
            (int)floor( pointTileSpace.x * numVoxelsPerTile.x * pow( 2.0, (int)pointTileSpace.w ) ),
            (int)floor( pointTileSpace.y * numVoxelsPerTile.y * pow( 2.0, (int)pointTileSpace.w ) ),
            (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

    return indexVoxelSpace;
}

MojoInt3 TileManager::GetOffsetVoxelSpace( MojoFloat4 pointTileSpace, MojoInt4 tileIndex, MojoInt3 numVoxelsPerTile )
{
    //
    // compute the offset (in data space) within the current tile 
    //
    MojoFloat4 tileIndexMojoFloat4 =
        MojoFloat4(
            (float)tileIndex.x,
            (float)tileIndex.y,
            (float)tileIndex.z,
            (float)tileIndex.w );

    //
    // compute the offset in voxels within the current tile 
    //
    MojoInt3 offsetVoxelSpace = 
        MojoInt3(
            (int)floor( ( pointTileSpace.x - tileIndexMojoFloat4.x ) * numVoxelsPerTile.x ),
            (int)floor( ( pointTileSpace.y - tileIndexMojoFloat4.y ) * numVoxelsPerTile.y ),
            (int)floor( ( pointTileSpace.z - tileIndexMojoFloat4.z ) * numVoxelsPerTile.z ) );

    return offsetVoxelSpace;
}

void TileManager::ReloadTileCache()
{
    for ( int cacheIndex = 0; cacheIndex < DEVICE_TILE_CACHE_SIZE; cacheIndex++ )
    {
        if ( mTileCache[ cacheIndex ].indexTileSpace.x != TILE_CACHE_PAGE_TABLE_BAD_INDEX &&
             mTileCache[ cacheIndex ].indexTileSpace.y != TILE_CACHE_PAGE_TABLE_BAD_INDEX &&
             mTileCache[ cacheIndex ].indexTileSpace.z != TILE_CACHE_PAGE_TABLE_BAD_INDEX &&
             mTileCache[ cacheIndex ].indexTileSpace.w != TILE_CACHE_PAGE_TABLE_BAD_INDEX )
        {
            //
            // load image data into host memory
            //
            MojoInt4 tileIndex = MojoInt4 ( mTileCache[ cacheIndex ].indexTileSpace );
            Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions = mTileServer->LoadTile( tileIndex );

            //
            // load the new data into into device memory for the new cache entry
            //
            mTileCache[ cacheIndex ].d3d11CudaTextures.Get( "SourceMap" )->Update( volumeDescriptions.Get( "SourceMap" ) );

			if ( mIsSegmentationLoaded )
			{
				if ( volumeDescriptions.GetHashMap().find( "IdMap" ) != volumeDescriptions.GetHashMap().end() )
				{
					mTileCache[ cacheIndex ].d3d11CudaTextures.Get( "IdMap"     )->Update( volumeDescriptions.Get( "IdMap"     ) );

                    //MojoInt3 volShape = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile();
                    //int idNumVoxelsPerTile = volShape.x * volShape.y * volShape.z;

                    //Core::Thrust::MemcpyHostToDevice( mTileCache[ cacheIndex ].deviceVectors.Get< int >( "IdMap" ), volumeDescriptions.Get( "IdMap" ).data, idNumVoxelsPerTile );
				}

				if ( volumeDescriptions.GetHashMap().find( "OverlayMap" ) != volumeDescriptions.GetHashMap().end() )
				{
					mTileCache[ cacheIndex ].d3d11CudaTextures.Get( "OverlayMap" )->Update( volumeDescriptions.Get( "OverlayMap" ) );

					//MojoInt3 volShape = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "OverlayMap" ).numVoxelsPerTile();
					//int idNumVoxelsPerTile = volShape.x * volShape.y * volShape.z;

					//Core::Thrust::MemcpyHostToDevice( mTileCache[ cacheIndex ].deviceVectors.Get< int >( "OverlayMap" ), volumeDescriptions.Get( "OverlayMap" ).data, idNumVoxelsPerTile );
				}
			}

            //
            // unload image data from from host memory
            //
            mTileServer->UnloadTile( tileIndex );
        }
    }
}

void TileManager::ReloadTileCacheOverlayMapOnly( int currentZ )
{
    for ( int cacheIndex = 0; cacheIndex < DEVICE_TILE_CACHE_SIZE; cacheIndex++ )
    {
        if ( mTileCache[ cacheIndex ].indexTileSpace.x != TILE_CACHE_PAGE_TABLE_BAD_INDEX &&
             mTileCache[ cacheIndex ].indexTileSpace.y != TILE_CACHE_PAGE_TABLE_BAD_INDEX &&
             mTileCache[ cacheIndex ].indexTileSpace.z != TILE_CACHE_PAGE_TABLE_BAD_INDEX &&
             mTileCache[ cacheIndex ].indexTileSpace.w != TILE_CACHE_PAGE_TABLE_BAD_INDEX &&
             mTileCache[ cacheIndex ].indexTileSpace.z == currentZ )
        {
            //
            // load image data into host memory
            //
            MojoInt4 tileIndex = MojoInt4 ( mTileCache[ cacheIndex ].indexTileSpace );
            Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions = mTileServer->LoadTile( tileIndex );

            //
            // load the new data into into device memory for the new cache entry
            //

			if ( mIsSegmentationLoaded )
			{
				if ( volumeDescriptions.GetHashMap().find( "OverlayMap" ) != volumeDescriptions.GetHashMap().end() )
				{
					mTileCache[ cacheIndex ].d3d11CudaTextures.Get( "OverlayMap" )->Update( volumeDescriptions.Get( "OverlayMap" ) );

					//MojoInt3 volShape = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "OverlayMap" ).numVoxelsPerTile();
					//int idNumVoxelsPerTile = volShape.x * volShape.y * volShape.z;

					//Core::Thrust::MemcpyHostToDevice( mTileCache[ cacheIndex ].deviceVectors.Get< int >( "OverlayMap" ), volumeDescriptions.Get( "OverlayMap" ).data, idNumVoxelsPerTile );
				}
			}

            //
            // unload image data from from host memory
            //
            mTileServer->UnloadTile( tileIndex );
        }
    }
}

}
}