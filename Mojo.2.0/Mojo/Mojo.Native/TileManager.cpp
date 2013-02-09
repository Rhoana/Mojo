#include "TileManager.hpp"

#include "Mojo.Core/Stl.hpp"

#include "Mojo.Core/OpenCV.hpp"
#include "Mojo.Core/ForEach.hpp"
#include "Mojo.Core/D3D11CudaTexture.hpp"
#include "Mojo.Core/Index.hpp"

#include "TileCacheEntry.hpp"

namespace Mojo
{
namespace Native
{

TileManager::TileManager( ID3D11Device* d3d11Device, ID3D11DeviceContext* d3d11DeviceContext, ITileServer* tileServer, Core::PrimitiveMap constParameters ) :
    mIdColorMapBuffer            ( NULL ),
    mIdColorMapShaderResourceView( NULL ),
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

    std::list< int4 > tileIndices = GetTileIndicesIntersectedByView( tiledDatasetView );

    //
    // explicitly mark all previously loaded cache entries that intersect the current view as cache entries to keep
    //
    MOJO_FOR_EACH( int4 tileIndex, tileIndices )
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
    MOJO_FOR_EACH( int4 tileIndex, tileIndices )
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
            int newCacheIndex = 0;

            for (; mTileCache[ newCacheIndex ].keepState != TileCacheEntryKeepState_CanDiscard; newCacheIndex++ );

            RELEASE_ASSERT( !mTileCache[ newCacheIndex ].active );

            //
            // get the new cache entry's index in tile space
            //
            int4 tempTileIndex = mTileCache[ newCacheIndex ].indexTileSpace;

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

					int3 volShape = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
					int idNumVoxelsPerTile = volShape.x * volShape.y * volShape.z;

					Core::Thrust::MemcpyHostToDevice( mTileCache[ newCacheIndex ].deviceVectors.Get< int >( "IdMap" ), volumeDescriptions.Get( "IdMap" ).data, idNumVoxelsPerTile );
				}

				if ( volumeDescriptions.GetHashMap().find( "OverlayMap" ) != volumeDescriptions.GetHashMap().end() )
				{
					mTileCache[ newCacheIndex ].d3d11CudaTextures.Get( "OverlayMap" )->Update( volumeDescriptions.Get( "OverlayMap" ) );

					int3 volShape = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "OverlayMap" ).numVoxelsPerTile;
					int idNumVoxelsPerTile = volShape.x * volShape.y * volShape.z;

					Core::Thrust::MemcpyHostToDevice( mTileCache[ newCacheIndex ].deviceVectors.Get< int >( "OverlayMap" ), volumeDescriptions.Get( "OverlayMap" ).data, idNumVoxelsPerTile );
				}
			}

            //
            // unload image data from from host memory
            //
            mTileServer->UnloadTile( tileIndex );

            //
            // update tile cache state for the new cache entry
            //
            float3 extentDataSpace =
                make_float3(
                    mConstParameters.Get< int >( "TILE_SIZE_X" ) * (float)pow( 2.0, tileIndex.w ),
                    mConstParameters.Get< int >( "TILE_SIZE_Y" ) * (float)pow( 2.0, tileIndex.w ),
                    mConstParameters.Get< int >( "TILE_SIZE_Z" ) * (float)pow( 2.0, tileIndex.w ) );

            float3 centerDataSpace =
                make_float3(
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

unsigned int TileManager::GetSegmentationLabelId( const TiledDatasetView& tiledDatasetView, float3 pDataSpace )
{
	int3   zoomLevel = GetZoomLevel( tiledDatasetView );
	float4 pointTileSpace;
	int4   tileIndex;
	int segmentId = -1;

	if ( mIsSegmentationLoaded )
	{

		GetIndexTileSpace( zoomLevel, pDataSpace, pointTileSpace, tileIndex );

		int3 numVoxels = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxels;
		int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
		int3 pVoxelSpace = GetIndexVoxelSpace( pointTileSpace, numVoxelsPerTile );

		//
		// Check for overflow
		//
		if ( pVoxelSpace.x >= 0 && pVoxelSpace.x < numVoxels.x &&
			pVoxelSpace.y >= 0 && pVoxelSpace.y < numVoxels.y &&
			pVoxelSpace.z >= 0 && pVoxelSpace.z < numVoxels.z )
		{
			int3 offsetVoxelSpace   = GetOffsetVoxelSpace( pointTileSpace, tileIndex, numVoxelsPerTile );
			int  offsetVoxelSpace1D = Core::Index3DToIndex1D( offsetVoxelSpace, numVoxelsPerTile );

			Core::HashMap< std::string, Core::VolumeDescription > thisTile = mTileServer->LoadTile( tileIndex );
			segmentId = ( (int*) thisTile.Get( "IdMap" ).data )[ offsetVoxelSpace1D ];
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

void TileManager::LockSegmentLabel( unsigned int segId )
{
	mTileServer->LockSegmentLabel( segId );
}

void TileManager::UnlockSegmentLabel( unsigned int segId )
{
	mTileServer->UnlockSegmentLabel( segId );
}

std::list< SegmentInfo > TileManager::GetSegmentInfoRange( int begin, int end )
{
	return mTileServer->GetSegmentInfoRange( begin, end );
}

int4 TileManager::GetSegmentationLabelColor( unsigned int segId )
{
    int index = segId % mIdColorMap.shape( 0 );
    return make_int4( mIdColorMap( index, 0 ), mIdColorMap( index, 1 ), mIdColorMap( index, 2 ), 255 );
}

void TileManager::ReplaceSegmentationLabel( unsigned int oldId, unsigned int newId )
{
    mTileServer->ReplaceSegmentationLabel( oldId, newId );

    ReloadTileCache();
}

void TileManager::ReplaceSegmentationLabelCurrentSlice( unsigned int oldId, unsigned int newId, float3 pDataSpace )
{
    mTileServer->ReplaceSegmentationLabelCurrentSlice( oldId, newId, pDataSpace );

    ReloadTileCache();
}

void TileManager::DrawSplit( float3 pointTileSpace, float radius )
{
    mTileServer->DrawSplit( pointTileSpace, radius );

    ReloadTileCacheOverlayMapOnly();
}

void TileManager::DrawErase( float3 pointTileSpace, float radius )
{
    mTileServer->DrawErase( pointTileSpace, radius );

    ReloadTileCacheOverlayMapOnly();
}

void TileManager::DrawRegionA( float3 pointTileSpace, float radius )
{
    mTileServer->DrawRegionA( pointTileSpace, radius );

    ReloadTileCacheOverlayMapOnly();
}

void TileManager::DrawRegionB( float3 pointTileSpace, float radius )
{
    mTileServer->DrawRegionB( pointTileSpace, radius );

    ReloadTileCacheOverlayMapOnly();
}

void TileManager::AddSplitSource( float3 pointTileSpace )
{
    mTileServer->AddSplitSource( pointTileSpace );
}

void TileManager::RemoveSplitSource()
{
    mTileServer->RemoveSplitSource();
}

void TileManager::ResetSplitState()
{
    mTileServer->ResetSplitState();

    ReloadTileCacheOverlayMapOnly();
}

void TileManager::PrepForSplit( unsigned int segId, float3 pointTileSpace )
{
    mTileServer->PrepForSplit( segId, pointTileSpace );

	ReloadTileCacheOverlayMapOnly();
}

void TileManager::FindBoundaryJoinPoints2D( unsigned int segId )
{
    mTileServer->FindBoundaryJoinPoints2D( segId );

    ReloadTileCacheOverlayMapOnly();
}

void TileManager::FindBoundaryWithinRegion2D( unsigned int segId )
{
    mTileServer->FindBoundaryWithinRegion2D( segId );

    ReloadTileCacheOverlayMapOnly();
}

void TileManager::FindBoundaryBetweenRegions2D( unsigned int segId )
{
    mTileServer->FindBoundaryBetweenRegions2D( segId );

    ReloadTileCacheOverlayMapOnly();
}

int TileManager::CompletePointSplit( unsigned int segId, float3 pointTileSpace )
{
    int newId = mTileServer->CompletePointSplit( segId, pointTileSpace );

    mTileServer->PrepForSplit( segId, pointTileSpace );

    ReloadTileCache();

	return newId;
}

int TileManager::CompleteDrawSplit( unsigned int segId, float3 pointTileSpace, bool join3D, int splitStartZ )
{
    int newId = mTileServer->CompleteDrawSplit( segId, pointTileSpace, join3D, splitStartZ );

    mTileServer->PrepForSplit( segId, pointTileSpace );

    ReloadTileCache();

	return newId;
}

void TileManager::RecordSplitState( unsigned int segId, float3 pointTileSpace )
{
    mTileServer->RecordSplitState( segId, pointTileSpace );
}

void TileManager::PredictSplit( unsigned int segId, float3 pointTileSpace, float radius )
{
    mTileServer->PredictSplit( segId, pointTileSpace, radius );

    ReloadTileCacheOverlayMapOnly();
}

void TileManager::ResetAdjustState()
{
    mTileServer->ResetAdjustState();

    ReloadTileCacheOverlayMapOnly();
}

void TileManager::PrepForAdjust( unsigned int segId, float3 pointTileSpace )
{
    mTileServer->PrepForAdjust( segId, pointTileSpace );

    ReloadTileCacheOverlayMapOnly();
}

void TileManager::CommitAdjustChange( unsigned int segId, float3 pointTileSpace )
{
    mTileServer->CommitAdjustChange( segId, pointTileSpace );

    mTileServer->PrepForAdjust( segId, pointTileSpace );

    ReloadTileCache();
}

void TileManager::ReplaceSegmentationLabelCurrentConnectedComponent( unsigned int oldId, unsigned int newId, float3 pDataSpace )
{
    mTileServer->ReplaceSegmentationLabelCurrentConnectedComponent( oldId, newId, pDataSpace );

    ReloadTileCache();
}

void TileManager::UndoChange()
{
	mTileServer->UndoChange();

    ReloadTileCache();
}

void TileManager::RedoChange()
{
    mTileServer->RedoChange();

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
        // release id color map
        //
        //mIdColorMapShaderResourceView->Release();
        //mIdColorMapShaderResourceView = NULL;

        //mIdColorMapBuffer->Release();
        //mIdColorMapBuffer = NULL;

        //mIdColorMap = marray::Marray< unsigned char >();

        //
        // reset page table
        //
        mTileCachePageTable = marray::Marray< int >( 0 );

        //
        // output memory stats to the console
        //
        size_t freeMemory, totalMemory;
        CUresult     memInfoResult;
        memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
        RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );
        Core::Printf( "\nUnloading tiled dataset...\n" );
        Core::Printf( "\n    Before freeing GPU memory:\n",
                      "        Free memory:  ", (unsigned int) freeMemory  / ( 1024 * 1024 ), " MBytes.\n",
                      "        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );

        //
        // delete textures in the tile cache
        //
        for ( int i = 0; i < DEVICE_TILE_CACHE_SIZE; i++ )
        {
            mTileCache[ i ].deviceVectors.Clear();

            delete mTileCache[ i ].d3d11CudaTextures.Get( "SourceMap" );
            delete mTileCache[ i ].d3d11CudaTextures.Get( "IdMap" );
            delete mTileCache[ i ].d3d11CudaTextures.Get( "OverlayMap" );

            mTileCache[ i ].d3d11CudaTextures.GetHashMap().clear();
        }

        //
        // output memory stats to the console
        //
        memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
        RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );
        Core::Printf( "    After freeing GPU memory:\n",
                      "        Free memory:  ", (unsigned int) freeMemory  / ( 1024 * 1024 ), " MBytes.\n",
                      "        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );
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
        size_t freeMemory, totalMemory;
        CUresult     memInfoResult;
        memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
        RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );
        Core::Printf( "\nUnloading segmentation...\n" );
        Core::Printf( "\n    Before freeing GPU memory:\n",
                      "        Free memory:  ", (unsigned int) freeMemory  / ( 1024 * 1024 ), " MBytes.\n",
                      "        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );

        //
        // release id color map
        //
        mIdColorMapShaderResourceView->Release();
        mIdColorMapShaderResourceView = NULL;

        mIdColorMapBuffer->Release();
        mIdColorMapBuffer = NULL;

        mIdColorMap = marray::Marray< unsigned char >();

        //
        // output memory stats to the console
        //
        memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
        RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );
        Core::Printf( "    After freeing GPU memory:\n",
                      "        Free memory:  ", (unsigned int) freeMemory  / ( 1024 * 1024 ), " MBytes.\n",
                      "        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );
    }
}

int3 TileManager::GetZoomLevel( const TiledDatasetView& tiledDatasetView )
{
    //
    // figure out what the current zoom level is
    //
    int4   numTiles         = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles;
    int3   numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numVoxelsPerTile;

    double expZoomLevelXY   = ( tiledDatasetView.extentDataSpace.x * numVoxelsPerTile.x ) / ( tiledDatasetView.widthNumPixels );

    int    zoomLevelXY      = (int)floor( std::min( (double)( numTiles.w - 1 ), std::max( 0.0, ( log( expZoomLevelXY ) / log( 2.0 ) ) ) ) );
    int    zoomLevelZ       = 0;

    return make_int3( zoomLevelXY, zoomLevelXY, zoomLevelZ );
}

std::list< int4 > TileManager::GetTileIndicesIntersectedByView( const TiledDatasetView& tiledDatasetView )
{
    std::list< int4 > tilesIntersectedByCamera;

    //
    // figure out what the current zoom level is
    //
    int3 zoomLevel   = GetZoomLevel( tiledDatasetView );
    int  zoomLevelXY = std::min( zoomLevel.x, zoomLevel.y );
    int  zoomLevelZ  = zoomLevel.z;

    //
    // figure out how many tiles there are at the current zoom level
    //
    int4 numTiles              = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles;
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
    float3 topLeftFrontDataSpace =
        make_float3(
            tiledDatasetView.centerDataSpace.x - ( tiledDatasetView.extentDataSpace.x / 2 ),
            tiledDatasetView.centerDataSpace.y - ( tiledDatasetView.extentDataSpace.y / 2 ),
            tiledDatasetView.centerDataSpace.z - ( tiledDatasetView.extentDataSpace.z / 2 ) );

    float3 bottomRightBackDataSpace =
        make_float3(
            tiledDatasetView.centerDataSpace.x + ( tiledDatasetView.extentDataSpace.x / 2 ),
            tiledDatasetView.centerDataSpace.y + ( tiledDatasetView.extentDataSpace.y / 2 ),
            tiledDatasetView.centerDataSpace.z + ( tiledDatasetView.extentDataSpace.z / 2 ) );

    //
    // compute the tile space indices for the top-left-front and bottom-right-back points
    //
    int3 topLeftFrontTileIndex =
        make_int3(
            (int)floor( topLeftFrontDataSpace.x / tileSizeDataSpaceX ),
            (int)floor( topLeftFrontDataSpace.y / tileSizeDataSpaceY ),
            (int)floor( topLeftFrontDataSpace.z / tileSizeDataSpaceZ ) );

    int3 bottomRightBackTileIndex =
        make_int3(
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
                tilesIntersectedByCamera.push_back( make_int4( x, y, z, zoomLevelXY ) );

    return tilesIntersectedByCamera;
}

void TileManager::GetIndexTileSpace( int3 zoomLevel, float3 pointDataSpace, float4& pointTileSpace, int4& tileIndex )
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
        make_float4(
            (float)pointDataSpace.x / tileSizeDataSpaceX,
            (float)pointDataSpace.y / tileSizeDataSpaceY,
            (float)pointDataSpace.z / tileSizeDataSpaceZ,
            (float)zoomLevelXY );

    tileIndex =
        make_int4(
            (int)floor( pointTileSpace.x ),
            (int)floor( pointTileSpace.y ),
            (int)floor( pointTileSpace.z ),
            (int)floor( pointTileSpace.w ) );
}

int3 TileManager::GetIndexVoxelSpace( float4 pointTileSpace, int3 numVoxelsPerTile )
{
    //
    // compute the index in voxels within the full volume
    //
    int3 indexVoxelSpace = 
        make_int3(
            (int)floor( pointTileSpace.x * numVoxelsPerTile.x * pow( 2.0, (int)pointTileSpace.w ) ),
            (int)floor( pointTileSpace.y * numVoxelsPerTile.y * pow( 2.0, (int)pointTileSpace.w ) ),
            (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

    return indexVoxelSpace;
}

int3 TileManager::GetOffsetVoxelSpace( float4 pointTileSpace, int4 tileIndex, int3 numVoxelsPerTile )
{
    //
    // compute the offset (in data space) within the current tile 
    //
    float4 tileIndexFloat4 =
        make_float4(
            (float)tileIndex.x,
            (float)tileIndex.y,
            (float)tileIndex.z,
            (float)tileIndex.w );

    float4 offsetTileSpace = pointTileSpace - tileIndexFloat4;

    //
    // compute the offset in voxels within the current tile 
    //
    int3 offsetVoxelSpace = 
        make_int3(
            (int)floor( offsetTileSpace.x * numVoxelsPerTile.x ),
            (int)floor( offsetTileSpace.y * numVoxelsPerTile.y ),
            (int)floor( offsetTileSpace.z * numVoxelsPerTile.z ) );

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
            int4 tileIndex = mTileCache[ cacheIndex ].indexTileSpace;
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

					int3 volShape = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
					int idNumVoxelsPerTile = volShape.x * volShape.y * volShape.z;

					Core::Thrust::MemcpyHostToDevice( mTileCache[ cacheIndex ].deviceVectors.Get< int >( "IdMap" ), volumeDescriptions.Get( "IdMap" ).data, idNumVoxelsPerTile );
				}

				if ( volumeDescriptions.GetHashMap().find( "OverlayMap" ) != volumeDescriptions.GetHashMap().end() )
				{
					mTileCache[ cacheIndex ].d3d11CudaTextures.Get( "OverlayMap" )->Update( volumeDescriptions.Get( "OverlayMap" ) );

					int3 volShape = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "OverlayMap" ).numVoxelsPerTile;
					int idNumVoxelsPerTile = volShape.x * volShape.y * volShape.z;

					Core::Thrust::MemcpyHostToDevice( mTileCache[ cacheIndex ].deviceVectors.Get< int >( "OverlayMap" ), volumeDescriptions.Get( "OverlayMap" ).data, idNumVoxelsPerTile );
				}
			}

            //
            // unload image data from from host memory
            //
            mTileServer->UnloadTile( tileIndex );
        }
    }
}

void TileManager::ReloadTileCacheOverlayMapOnly()
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
            int4 tileIndex = mTileCache[ cacheIndex ].indexTileSpace;
            Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions = mTileServer->LoadTile( tileIndex );

            //
            // load the new data into into device memory for the new cache entry
            //

			if ( mIsSegmentationLoaded )
			{
				if ( volumeDescriptions.GetHashMap().find( "OverlayMap" ) != volumeDescriptions.GetHashMap().end() )
				{
					mTileCache[ cacheIndex ].d3d11CudaTextures.Get( "OverlayMap" )->Update( volumeDescriptions.Get( "OverlayMap" ) );

					int3 volShape = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "OverlayMap" ).numVoxelsPerTile;
					int idNumVoxelsPerTile = volShape.x * volShape.y * volShape.z;

					Core::Thrust::MemcpyHostToDevice( mTileCache[ cacheIndex ].deviceVectors.Get< int >( "OverlayMap" ), volumeDescriptions.Get( "OverlayMap" ).data, idNumVoxelsPerTile );
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