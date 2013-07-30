#include "TileManager.hpp"

#include "Stl.hpp"
#include "ForEach.hpp"
#include "D3D11Texture.hpp"
#include "Index.hpp"
#include "TileCacheEntry.hpp"

namespace Mojo
{
namespace Native
{

//
// CODE QUALITY ISSUE:
// Not all state is initialized in constructor. For example, ints like mTileCacheSearchStart should be initialized to -1. -MR
//
TileManager::TileManager( ID3D11Device* d3d11Device, ID3D11DeviceContext* d3d11DeviceContext, ITileServer* tileServer, PrimitiveMap constParameters ) :
    mIdColorMapBuffer                 ( NULL ),
    mIdColorMapShaderResourceView     ( NULL ),
    mLabelIdMapBuffer                 ( NULL ),
    mLabelIdMapShaderResourceView     ( NULL ),
    mIdConfidenceMapBuffer            ( NULL ),
    mIdConfidenceMapShaderResourceView( NULL ),
    mTileServer                       ( tileServer ),
    mConstParameters                  ( constParameters ),
    mAreSourceImagesLoaded            ( false ),
    mIsSegmentationLoaded             ( false )
{
    mD3D11Device = d3d11Device;
    mD3D11Device->AddRef();

    mD3D11DeviceContext = d3d11DeviceContext;
    mD3D11DeviceContext->AddRef();
}

TileManager::~TileManager()
{
    if ( mIsSegmentationLoaded )
    {
       UnloadSegmentation();
    }

    if ( mAreSourceImagesLoaded )
    {
        UnloadSourceImages();
    }

    mD3D11DeviceContext->Release();
    mD3D11DeviceContext = NULL;

    mD3D11Device->Release();
    mD3D11Device = NULL;
}

void TileManager::LoadSourceImages( TiledDatasetDescription& tiledDatasetDescription )
{
    RELEASE_ASSERT( tiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).dxgiFormat == DXGI_FORMAT_R8_UNORM );

    IDXGIDevice * pDXGIDevice;
    mD3D11Device->QueryInterface(__uuidof(IDXGIDevice), (void **)&pDXGIDevice);
    IDXGIAdapter * pDXGIAdapter;
    pDXGIDevice->GetAdapter(&pDXGIAdapter);
    DXGI_ADAPTER_DESC adapterDesc;
    pDXGIAdapter->GetDesc(&adapterDesc);

    //
    // inform the tile server
    //
    mTileServer->LoadSourceImages( tiledDatasetDescription );

    RELEASE_ASSERT( mTileServer->AreSourceImagesLoaded() );

    //
    // We assume that the source, id, and overlay tiles are the same shape
    //
    mSourceImagesTiledDatasetDescription = tiledDatasetDescription;
    TiledVolumeDescription t             = mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" );
    int bytesPerTile                     = t.numVoxelsPerTile.x * t.numVoxelsPerTile.y * t.numVoxelsPerTile.z * t.numBytesPerVoxel;
        bytesPerTile                    += t.numVoxelsPerTile.x * t.numVoxelsPerTile.y * t.numVoxelsPerTile.z * mConstParameters.Get< int >( "ID_MAP_NUM_BYTES_PER_VOXEL" );
        bytesPerTile                    += t.numVoxelsPerTile.x * t.numVoxelsPerTile.y * t.numVoxelsPerTile.z * mConstParameters.Get< int >( "OVERLAY_MAP_NUM_BYTES_PER_VOXEL" );

    //
    // CODE QUALITY ISSUE:
    // Move "1024 * 1024 * 100" into a ConstParameter. Also, what is the unit for MAX_DEVICE_TILE_CACHE_SIZE? Is it in bytes? Number of tiles? Consider renaming.
    // Finally, mDeviceTileCacheSize should be computed in closed form. -MR
    //

    //
    // Try to leave at least 100MB video memory free
    //
    mDeviceTileCacheSize = MAX_DEVICE_TILE_CACHE_SIZE;
    while ( mDeviceTileCacheSize * bytesPerTile > (int)adapterDesc.DedicatedVideoMemory - ( 1024 * 1024 * 100 ) )
    {
        mDeviceTileCacheSize /= 2;
    }

    mTileCache.reserve( mDeviceTileCacheSize );

    Printf( "\nLoading tiled dataset...\n" );
    Printf( "        Total memory:    ", (unsigned int) adapterDesc.DedicatedVideoMemory  / ( 1024 * 1024 ), " MB\n" );
    Printf( "        Bytes per tile:  ", bytesPerTile, "\n" );
    Printf( "        Tile cache size: ", mDeviceTileCacheSize, " (", mDeviceTileCacheSize * bytesPerTile / ( 1024 * 1024 ), " MB)\n" );

    //
    // initialize the tile cache
    //
    for ( int i = 0; i < mDeviceTileCacheSize; i++ )
    {
        TileCacheEntry tileCacheEntry;

        //
        // We assume that the source, id, and overlay tiles are the same shape
        //
        int numVoxelsPerTile = t.numVoxelsPerTile.x * t.numVoxelsPerTile.y * t.numVoxelsPerTile.z;

        D3D11_TEXTURE3D_DESC textureDesc3D;
        ZeroMemory( &textureDesc3D, sizeof( D3D11_TEXTURE3D_DESC ) );

        textureDesc3D.Width     = t.numVoxelsPerTile.x;
        textureDesc3D.Height    = t.numVoxelsPerTile.y;
        textureDesc3D.Depth     = t.numVoxelsPerTile.z;
        textureDesc3D.MipLevels = 1;
        textureDesc3D.Usage     = D3D11_USAGE_DEFAULT;
        textureDesc3D.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        textureDesc3D.Format    = t.dxgiFormat;
        
        tileCacheEntry.d3d11Textures.Set(
            "SourceMap",
            new D3D11Texture< ID3D11Texture3D >(
                mD3D11Device,
                mD3D11DeviceContext,
                textureDesc3D ) );

        //
        // Allocate memory for the IdMap and OverlayMap tiles here
        // (data will be loaded in LoadSegmentation)
        //
        ZeroMemory( &textureDesc3D, sizeof( D3D11_TEXTURE3D_DESC ) );

        textureDesc3D.Width     = t.numVoxelsPerTile.x;
        textureDesc3D.Height    = t.numVoxelsPerTile.y;
        textureDesc3D.Depth     = t.numVoxelsPerTile.z;
        textureDesc3D.MipLevels = 1;
        textureDesc3D.Usage     = D3D11_USAGE_DEFAULT;
        textureDesc3D.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        textureDesc3D.Format    = DXGI_FORMAT_R32_UINT;

        tileCacheEntry.d3d11Textures.Set(
            "IdMap",
            new D3D11Texture< ID3D11Texture3D >(
                mD3D11Device,
                mD3D11DeviceContext,
                textureDesc3D ) );
        
        ZeroMemory( &textureDesc3D, sizeof( D3D11_TEXTURE3D_DESC ) );

        textureDesc3D.Width     = t.numVoxelsPerTile.x;
        textureDesc3D.Height    = t.numVoxelsPerTile.y;
        textureDesc3D.Depth     = t.numVoxelsPerTile.z;
        textureDesc3D.MipLevels = 1;
        textureDesc3D.Usage     = D3D11_USAGE_DEFAULT;
        textureDesc3D.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        textureDesc3D.Format    = DXGI_FORMAT_R32_UINT;

        tileCacheEntry.d3d11Textures.Set(
            "OverlayMap",
            new D3D11Texture< ID3D11Texture3D >(
                mD3D11Device,
                mD3D11DeviceContext,
                textureDesc3D ) );

        mTileCache.push_back( tileCacheEntry );
    }

    mTileCacheSearchStart = 0;

    //
    // initialize the page table
    //
    Int4 numTiles = t.numTiles;

    size_t shape[] = { numTiles.w, numTiles.z, numTiles.y, numTiles.x };
    mTileCachePageTable = marray::Marray< int >( shape, shape + 4 );

    for ( int w = 0; w < numTiles.w; w++ )
        for ( int z = 0; z < numTiles.z; z++ )
            for ( int y = 0; y < numTiles.y; y++ )
                for ( int x = 0; x < numTiles.x; x++ )
                    mTileCachePageTable( w, z, y, x ) = TILE_CACHE_BAD_INDEX; 

    //
    // CODE QUALITY ISSUE:
    // The name FileSystemCache is misleading, since the cache actually resides in host memory. Consider HostMemoryCache.
    // Also, the fact that the FileSystemTileServer even maintains a host memory cache should not be in its public interface. -MR
    //

    mAreSourceImagesLoaded = true;
}

void TileManager::UnloadSourceImages()
{
    if ( mIsSegmentationLoaded )
    {
        UnloadSegmentation();
    }

    if ( mAreSourceImagesLoaded )
    {
        mAreSourceImagesLoaded = false;
        mTileCachePageTable    = marray::Marray< int >( 0 );
        mTileCacheSearchStart  = -1;

        for ( int i = 0; i < mDeviceTileCacheSize; i++ )
        {
            delete mTileCache[ i ].d3d11Textures.Get( "SourceMap" );
            delete mTileCache[ i ].d3d11Textures.Get( "IdMap" );
            delete mTileCache[ i ].d3d11Textures.Get( "OverlayMap" );

            mTileCache[ i ].d3d11Textures.GetHashMap().clear();
        }

        mTileCache.clear();

        mDeviceTileCacheSize                 = -1;
        mSourceImagesTiledDatasetDescription = TiledDatasetDescription();

        mTileServer->UnloadSourceImages();

        RELEASE_ASSERT( !mTileServer->AreSourceImagesLoaded() );
    }
}

bool TileManager::AreSourceImagesLoaded()
{
    return mAreSourceImagesLoaded;
}

void TileManager::LoadSegmentation( TiledDatasetDescription& tiledDatasetDescription )
{
    Printf( "\nLoading segmentation...\n" );

    RELEASE_ASSERT( tiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).dxgiFormat         == DXGI_FORMAT_R32_UINT );
    RELEASE_ASSERT( tiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxels.x        == mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numVoxels.x );
    RELEASE_ASSERT( tiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxels.y        == mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numVoxels.y );
    RELEASE_ASSERT( tiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxels.z        == mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numVoxels.z );
    RELEASE_ASSERT( tiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles.x         == mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles.x );
    RELEASE_ASSERT( tiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles.y         == mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles.y );
    RELEASE_ASSERT( tiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles.z         == mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles.z );
    RELEASE_ASSERT( tiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles.w         == mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles.w );
    RELEASE_ASSERT( tiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile.x == mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numVoxelsPerTile.x );
    RELEASE_ASSERT( tiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile.y == mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numVoxelsPerTile.y );
    RELEASE_ASSERT( tiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile.z == mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numVoxelsPerTile.z );

    //
    // inform the tile server
    //
    mTileServer->LoadSegmentation( tiledDatasetDescription );

    RELEASE_ASSERT( mTileServer->IsSegmentationLoaded() );

    //
    // store a copy of the tiled dataset description
    //
    mSegmentationTiledDatasetDescription = tiledDatasetDescription;

    //
    // reinitialize the page table
    //
    Int4 numTiles = mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles;

    for ( int w = 0; w < numTiles.w; w++ )
        for ( int z = 0; z < numTiles.z; z++ )
            for ( int y = 0; y < numTiles.y; y++ )
                for ( int x = 0; x < numTiles.x; x++ )
                    mTileCachePageTable( w, z, y, x ) = TILE_CACHE_BAD_INDEX; 

    for ( int i = 0; i < mDeviceTileCacheSize; i++ )
        mTileCache[ i ].active = false; 

    D3D11_BUFFER_DESC bufferDesc;
    D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;

    //
    // load the id color map
    //
    mIdColorMap = mTileServer->GetIdColorMap();

    unsigned char* idColorMap = new unsigned char[ mIdColorMap->shape( 0 ) * 4 ];

    for ( unsigned int i = 0; i < mIdColorMap->shape( 0 ); i++ )
    {
        idColorMap[ i * 4 + 0 ] = (*mIdColorMap)( i, 0 );
        idColorMap[ i * 4 + 1 ] = (*mIdColorMap)( i, 1 );
        idColorMap[ i * 4 + 2 ] = (*mIdColorMap)( i, 2 );
        idColorMap[ i * 4 + 3 ] = 255;
    }

    ZeroMemory( &bufferDesc, sizeof( D3D11_BUFFER_DESC ) );

    bufferDesc.ByteWidth           = (UINT) mIdColorMap->shape( 0 ) * sizeof( unsigned char ) * 4;
    bufferDesc.Usage               = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags           = (UINT) D3D11_BIND_SHADER_RESOURCE;
    bufferDesc.StructureByteStride = (UINT) sizeof( unsigned char ) * 4;
    
    ZeroMemory( &shaderResourceViewDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );

    shaderResourceViewDesc.Format               = DXGI_FORMAT_R8G8B8A8_UNORM;
    shaderResourceViewDesc.ViewDimension        = D3D11_SRV_DIMENSION_BUFFER;
    shaderResourceViewDesc.Buffer.FirstElement  = (UINT) 0;
    shaderResourceViewDesc.Buffer.NumElements   = (UINT) mIdColorMap->shape( 0 );

    MOJO_D3D_SAFE( mD3D11Device->CreateBuffer( &bufferDesc, NULL, &mIdColorMapBuffer ) );
    MOJO_D3D_SAFE( mD3D11Device->CreateShaderResourceView( mIdColorMapBuffer, &shaderResourceViewDesc, &mIdColorMapShaderResourceView ) );

    mD3D11DeviceContext->UpdateSubresource(
        mIdColorMapBuffer,
        0,
        NULL,
        idColorMap,
        (UINT) mIdColorMap->shape( 0 ) * sizeof( unsigned char ) * 4,
        (UINT) mIdColorMap->shape( 0 ) * sizeof( unsigned char ) * 4 );

    delete[] idColorMap;

    //
    // load the label id map
    //
    mLabelIdMap = mTileServer->GetLabelIdMap();

    size_t nSegmentsToAllocate = mLabelIdMap->shape( 0 );

    unsigned int* labelIdMap = new unsigned int [ nSegmentsToAllocate ];

    for ( unsigned int i = 0; i < mLabelIdMap->shape( 0 ); ++i )
    {
        labelIdMap[ i ] = ( (*mLabelIdMap)( i ) );
    }

    ZeroMemory( &bufferDesc, sizeof( D3D11_BUFFER_DESC ) );

    bufferDesc.ByteWidth           = (UINT) nSegmentsToAllocate * sizeof( unsigned int );
    bufferDesc.Usage               = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags           = (UINT) D3D11_BIND_SHADER_RESOURCE;
    bufferDesc.StructureByteStride = (UINT) sizeof( unsigned int );
    
    ZeroMemory( &shaderResourceViewDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );

    shaderResourceViewDesc.Format               = DXGI_FORMAT_R32_UINT;
    shaderResourceViewDesc.ViewDimension        = D3D11_SRV_DIMENSION_BUFFER;
    shaderResourceViewDesc.Buffer.FirstElement  = (UINT) 0;
    shaderResourceViewDesc.Buffer.NumElements   = (UINT) nSegmentsToAllocate;

    MOJO_D3D_SAFE( mD3D11Device->CreateBuffer( &bufferDesc, NULL, &mLabelIdMapBuffer ) );
    MOJO_D3D_SAFE( mD3D11Device->CreateShaderResourceView( mLabelIdMapBuffer, &shaderResourceViewDesc, &mLabelIdMapShaderResourceView ) );

    mD3D11DeviceContext->UpdateSubresource(
        mLabelIdMapBuffer,
        0,
        NULL,
        labelIdMap,
        (UINT) nSegmentsToAllocate * sizeof( unsigned int ),
        (UINT) nSegmentsToAllocate * sizeof( unsigned int ) );

    delete[] labelIdMap;

    //
    // load the id lock map
    //

    //
    // CODE QUALITY ISSUE:
    // Is the lock map the same as the confidence map? This snippet is quite hard to read.
    // Also, why the for loops? Can't you just upload the Marrays directly to the GPU? -MR
    //
    mIdConfidenceMap = mTileServer->GetIdConfidenceMap();

    unsigned char* idConfidenceMap = new unsigned char[ mIdConfidenceMap->shape( 0 ) ];

    for ( unsigned int i = 0; i < mIdConfidenceMap->shape( 0 ); ++i )
    {
        idConfidenceMap[ i ] = (*mIdConfidenceMap)( i );
    }

    ZeroMemory( &bufferDesc, sizeof( D3D11_BUFFER_DESC ) );

    bufferDesc.ByteWidth           = (UINT) nSegmentsToAllocate * sizeof( unsigned char );
    bufferDesc.Usage               = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags           = (UINT) D3D11_BIND_SHADER_RESOURCE;
    bufferDesc.StructureByteStride = (UINT) sizeof( unsigned char );
    
    ZeroMemory( &shaderResourceViewDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );

    shaderResourceViewDesc.Format               = DXGI_FORMAT_R8_UNORM;
    shaderResourceViewDesc.ViewDimension        = D3D11_SRV_DIMENSION_BUFFER;
    shaderResourceViewDesc.Buffer.FirstElement  = (UINT) 0;
    shaderResourceViewDesc.Buffer.NumElements   = (UINT) nSegmentsToAllocate;

    MOJO_D3D_SAFE( mD3D11Device->CreateBuffer( &bufferDesc, NULL, &mIdConfidenceMapBuffer ) );
    MOJO_D3D_SAFE( mD3D11Device->CreateShaderResourceView( mIdConfidenceMapBuffer, &shaderResourceViewDesc, &mIdConfidenceMapShaderResourceView ) );

    mD3D11DeviceContext->UpdateSubresource(
        mIdConfidenceMapBuffer,
        0,
        NULL,
        idConfidenceMap,
        (UINT) nSegmentsToAllocate * sizeof( unsigned char ),
        (UINT) nSegmentsToAllocate * sizeof( unsigned char ) );

    delete[] idConfidenceMap;

    mIsSegmentationLoaded = true;
}

void TileManager::UnloadSegmentation()
{
    if ( mIsSegmentationLoaded )
    {
        mIsSegmentationLoaded = false;

        mIdConfidenceMapShaderResourceView->Release();
        mIdConfidenceMapShaderResourceView = NULL;

        mIdConfidenceMapBuffer->Release();
        mIdConfidenceMapBuffer = NULL;

        mLabelIdMapShaderResourceView->Release();
        mLabelIdMapShaderResourceView = NULL;

        mLabelIdMapBuffer->Release();
        mLabelIdMapBuffer = NULL;

        mIdColorMapShaderResourceView->Release();
        mIdColorMapShaderResourceView = NULL;

        mIdColorMapBuffer->Release();
        mIdColorMapBuffer = NULL;

        mIdColorMap = NULL;

        mTileServer->UnloadSegmentation();

        RELEASE_ASSERT( !mTileServer->IsSegmentationLoaded() );
    }
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

void TileManager::LoadOverTile( const TiledDatasetView& tiledDatasetView )
{
    //
    // Find a single tile that will completely cover the current view
    //
    Int4 tileIndex = GetTileIndexCoveringView( tiledDatasetView );

    int cacheIndex = mTileCachePageTable( tileIndex.w, tileIndex.z, tileIndex.y, tileIndex.x );

    //
    // all cache entries can be discarded unless they are in the current z window
    //
    for ( int tmpCacheIndex = 0; tmpCacheIndex < mDeviceTileCacheSize; tmpCacheIndex++ )
    {
        if ( mTileCache[ tmpCacheIndex ].indexTileSpace.z != tileIndex.z )
        {
            mTileCache[ tmpCacheIndex ].active = false;
        }
    }

    //
    // if the tile is already loaded, mark it as active.
    //
    if ( cacheIndex != TILE_CACHE_BAD_INDEX )
    {
        mTileCache[ cacheIndex ].active = true;
    }

    //
    // if the tile is not loaded...
    //
    if ( cacheIndex == TILE_CACHE_BAD_INDEX )
    {

        //Printf( "Loading single tile at: w=", tileIndex.w, ", z=", tileIndex.z, ", y=", tileIndex.y, ", x=", tileIndex.x, "." );

        //
        // Overwrite the tile at mTileCacheSearchStart
        //
        int newCacheIndex = mTileCacheSearchStart;

        mTileCacheSearchStart = ( newCacheIndex + 1 ) % mDeviceTileCacheSize;

        //RELEASE_ASSERT( !mTileCache[ newCacheIndex ]->active );

        //Printf( "Replacing tile ", newCacheIndex, " in the device cache.");

        //
        // get the new cache entry's index in tile space
        //
        Int4 indexTileSpace = mTileCache[ newCacheIndex ].indexTileSpace;
        Int4 tempTileIndex = Int4( indexTileSpace.x, indexTileSpace.y, indexTileSpace.z, indexTileSpace.w );

        //
        // if the new cache entry refers to a tile that is already loaded...
        //
        if ( tempTileIndex.x != TILE_CACHE_PAGE_TABLE_BAD_INDEX ||
             tempTileIndex.y != TILE_CACHE_PAGE_TABLE_BAD_INDEX ||
             tempTileIndex.z != TILE_CACHE_PAGE_TABLE_BAD_INDEX ||
             tempTileIndex.w != TILE_CACHE_PAGE_TABLE_BAD_INDEX )
        {
            //
            // mark the tile as not being loaded any more
            //
            mTileCachePageTable( tempTileIndex.w, tempTileIndex.z, tempTileIndex.y, tempTileIndex.x ) = TILE_CACHE_BAD_INDEX;
        }

        //
        // load image data into host memory
        //
        HashMap< std::string, VolumeDescription > volumeDescriptions = mTileServer->LoadTile( tileIndex );

        //
        // load the new data into into device memory for the new cache entry
        //
        mTileCache[ newCacheIndex ].d3d11Textures.Get( "SourceMap" )->Update( volumeDescriptions.Get( "SourceMap" ) );

        if ( IsSegmentationLoaded() )
        {
            mTileCache[ newCacheIndex ].d3d11Textures.Get( "IdMap"      )->Update( volumeDescriptions.Get( "IdMap"      ) );
            mTileCache[ newCacheIndex ].d3d11Textures.Get( "OverlayMap" )->Update( volumeDescriptions.Get( "OverlayMap" ) );
        }

        //
        // unload image data from from host memory
        //
        mTileServer->UnloadTile( tileIndex );

        //
        // update tile cache state for the new cache entry
        //
        Float3 extentDataSpace =
            Float3(
                mConstParameters.Get< int >( "TILE_SIZE_X" ) * (float)pow( 2.0, tileIndex.w ),
                mConstParameters.Get< int >( "TILE_SIZE_Y" ) * (float)pow( 2.0, tileIndex.w ),
                mConstParameters.Get< int >( "TILE_SIZE_Z" ) * (float)pow( 2.0, tileIndex.w ) );

        Float3 centerDataSpace =
            Float3(
                ( tileIndex.x + 0.5f ) * extentDataSpace.x,
                ( tileIndex.y + 0.5f ) * extentDataSpace.y,
                ( tileIndex.z + 0.5f ) * extentDataSpace.z );

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

void TileManager::LoadTiles( const TiledDatasetView& tiledDatasetView )
{
    //
    // assume that all cache entries can be discarded unless explicitly marked otherwise
    //
    for ( int cacheIndex = 0; cacheIndex < mDeviceTileCacheSize; cacheIndex++ )
    {
        mTileCache[ cacheIndex ].active = false;
    }

    std::list< Int4 > tileIndices = GetTileIndicesIntersectedByView( tiledDatasetView );

    //
    // explicitly mark all previously loaded cache entries that intersect the current view as cache entries to keep
    //
    MOJO_FOR_EACH( Int4 tileIndex, tileIndices )
    {
        int cacheIndex = mTileCachePageTable( tileIndex.w, tileIndex.z, tileIndex.y, tileIndex.x );

        if ( cacheIndex != TILE_CACHE_BAD_INDEX )
        {
            mTileCache[ cacheIndex ].active = true;
        }
    }

    //
    // for each tile that intersects the current view but is not loaded, load it and overwrite a cache entry that can be discarded
    //
    MOJO_FOR_EACH( Int4 tileIndex, tileIndices )
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
            int lastCacheIndex = ( mDeviceTileCacheSize + mTileCacheSearchStart - 1 ) % mDeviceTileCacheSize;

            for (; mTileCache[ newCacheIndex ].active == true; newCacheIndex = ( newCacheIndex + 1 ) % mDeviceTileCacheSize )
            {
                RELEASE_ASSERT ( newCacheIndex != lastCacheIndex );
            }

            mTileCacheSearchStart = ( newCacheIndex + 1 ) % mDeviceTileCacheSize;

            RELEASE_ASSERT( !mTileCache[ newCacheIndex ].active );

            //
            // get the new cache entry's index in tile space
            //
            Int4 indexTileSpace = mTileCache[ newCacheIndex ].indexTileSpace;
            Int4 tempTileIndex = Int4( indexTileSpace.x, indexTileSpace.y, indexTileSpace.z, indexTileSpace.w );

            //
            // if the new cache entry refers to a tile that is already loaded...
            //
            if ( tempTileIndex.x != TILE_CACHE_PAGE_TABLE_BAD_INDEX ||
                 tempTileIndex.y != TILE_CACHE_PAGE_TABLE_BAD_INDEX ||
                 tempTileIndex.z != TILE_CACHE_PAGE_TABLE_BAD_INDEX ||
                 tempTileIndex.w != TILE_CACHE_PAGE_TABLE_BAD_INDEX )
            {
                //
                // mark the tile as not being loaded any more
                //
                mTileCachePageTable( tempTileIndex.w, tempTileIndex.z, tempTileIndex.y, tempTileIndex.x ) = TILE_CACHE_BAD_INDEX;
            }

            //
            // load image data into host memory
            //
            HashMap< std::string, VolumeDescription > volumeDescriptions = mTileServer->LoadTile( tileIndex );

            //
            // load the new data into into device memory for the new cache entry
            //
            mTileCache[ newCacheIndex ].d3d11Textures.Get( "SourceMap" )->Update( volumeDescriptions.Get( "SourceMap" ) );

            if ( IsSegmentationLoaded() )
            {
                mTileCache[ newCacheIndex ].d3d11Textures.Get( "IdMap" )->Update( volumeDescriptions.Get( "IdMap" ) );
                mTileCache[ newCacheIndex ].d3d11Textures.Get( "OverlayMap" )->Update( volumeDescriptions.Get( "OverlayMap" ) );
            }

            //
            // unload image data from from host memory
            //
            mTileServer->UnloadTile( tileIndex );

            //
            // update tile cache state for the new cache entry
            //
            Float3 extentDataSpace =
                Float3(
                    mConstParameters.Get< int >( "TILE_SIZE_X" ) * (float)pow( 2.0, tileIndex.w ),
                    mConstParameters.Get< int >( "TILE_SIZE_Y" ) * (float)pow( 2.0, tileIndex.w ),
                    mConstParameters.Get< int >( "TILE_SIZE_Z" ) * (float)pow( 2.0, tileIndex.w ) );

            Float3 centerDataSpace =
                Float3(
                    ( tileIndex.x + 0.5f ) * extentDataSpace.x,
                    ( tileIndex.y + 0.5f ) * extentDataSpace.y,
                    ( tileIndex.z + 0.5f ) * extentDataSpace.z );

            mTileCache[ newCacheIndex ].active          = true;
            mTileCache[ newCacheIndex ].indexTileSpace  = tileIndex;
            mTileCache[ newCacheIndex ].centerDataSpace = centerDataSpace;
            mTileCache[ newCacheIndex ].extentDataSpace = extentDataSpace;

            //
            // mark the new location in tile space as being loaded into the cache
            //
            RELEASE_ASSERT( mTileCachePageTable( tileIndex.w, tileIndex.z, tileIndex.y, tileIndex.x ) == TILE_CACHE_BAD_INDEX );
            RELEASE_ASSERT( newCacheIndex >= 0 && newCacheIndex < mDeviceTileCacheSize );

            mTileCachePageTable( tileIndex.w, tileIndex.z, tileIndex.y, tileIndex.x ) = newCacheIndex;
        }
    }
}

std::vector< TileCacheEntry >& TileManager::GetTileCache()
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

unsigned int TileManager::GetSegmentationLabelId( const TiledDatasetView& tiledDatasetView, Float3 pDataSpace )
{
    Int3   zoomLevel = GetZoomLevel( tiledDatasetView );
    Float4 pointTileSpace;
    Int4   tileIndex;
    unsigned int segmentId = 0;

    if ( mIsSegmentationLoaded )
    {
        GetIndexTileSpace( zoomLevel, pDataSpace, pointTileSpace, tileIndex );

        TiledVolumeDescription tiledVolumeDescription = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

        Int3 numVoxels        = tiledVolumeDescription.numVoxels;
        Int3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile;
        Int3 pVoxelSpace = GetIndexVoxelSpace( pointTileSpace, numVoxelsPerTile );

        //
        // Check for overflow
        //
        if ( pVoxelSpace.x >= 0 && pVoxelSpace.x < numVoxels.x &&
             pVoxelSpace.y >= 0 && pVoxelSpace.y < numVoxels.y &&
             pVoxelSpace.z >= 0 && pVoxelSpace.z < numVoxels.z )
        {
            Int3 offsetVoxelSpace   = GetOffsetVoxelSpace( pointTileSpace, tileIndex, numVoxelsPerTile );
            int  offsetVoxelSpace1D = Index3DToIndex1D( offsetVoxelSpace, numVoxelsPerTile );

            HashMap< std::string, VolumeDescription > thisTile = mTileServer->LoadTile( tileIndex );

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
    Printf( "From ", fromSegId, " before -> ", (*mLabelIdMap)( fromSegId ), "." );    
    Printf( "To ", toSegId, " before -> ", (*mLabelIdMap)( toSegId ), "." );

    std::set< unsigned int > fromSegIds;
    fromSegIds.insert( fromSegId );
    mTileServer->RemapSegmentLabels( fromSegIds, toSegId );

    Printf( "From ", fromSegId, " after -> ", (*mLabelIdMap)( fromSegId ), "." );
    Printf( "To ", toSegId, " after -> ", (*mLabelIdMap)( toSegId ), "." );

    UpdateLabelIdMap( fromSegId );

}

void TileManager::UpdateLabelIdMap( unsigned int fromSegId )
{
    //
    // Update label id map shader buffer
    //
    unsigned int labelIdMapEntry = ( (*mLabelIdMap)( fromSegId ) );

    D3D11_BOX updateBox;
    ZeroMemory( &updateBox, sizeof( D3D11_BOX ) );

    updateBox.left = fromSegId * sizeof( unsigned int );
    updateBox.top = 0;
    updateBox.front = 0;
    updateBox.right = ( fromSegId + 1 ) * sizeof( unsigned int );
    updateBox.bottom = 1;
    updateBox.back = 1;

    mD3D11DeviceContext->UpdateSubresource(
        mLabelIdMapBuffer,
        0,
        &updateBox,
        &labelIdMapEntry,
        (UINT) mLabelIdMap->shape( 0 ) * sizeof( unsigned int ),
        (UINT) mLabelIdMap->shape( 0 ) * sizeof( unsigned int ) );
}

void TileManager::LockSegmentLabel( unsigned int segId )
{
    mTileServer->LockSegmentLabel( segId );

    //
    // Update confidence map shader buffer
    //
    unsigned char idConfidenceMapEntry = ( (*mIdConfidenceMap)( segId ) );

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
        (UINT) mIdConfidenceMap->shape( 0 ) * sizeof( unsigned char ),
        (UINT) mIdConfidenceMap->shape( 0 ) * sizeof( unsigned char ) );

}

void TileManager::UnlockSegmentLabel( unsigned int segId )
{
    mTileServer->UnlockSegmentLabel( segId );

    //
    // Update confidence map shader buffer
    //
    unsigned char idConfidenceMapEntry = 0;

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
        (UINT) mIdConfidenceMap->shape( 0 ) * sizeof( unsigned char ),
        (UINT) mIdConfidenceMap->shape( 0 ) * sizeof( unsigned char ) );

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

SegmentInfo TileManager::GetSegmentInfo( unsigned int segId )
{
    return mTileServer->GetSegmentInfo( segId );
}

Int3 TileManager::GetSegmentationLabelColor( unsigned int segId )
{
    if ( mIdColorMap->size() > 0 )
    {
        int index = segId % mIdColorMap->shape( 0 );
        return Int3( (*mIdColorMap)( index, 0 ), (*mIdColorMap)( index, 1 ), (*mIdColorMap)( index, 2 ) );
    }
    return Int3();
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

Int3 TileManager::GetSegmentCentralTileLocation( unsigned int segId )
{
    return mTileServer->GetSegmentCentralTileLocation( segId );
}

Int4 TileManager::GetSegmentZTileBounds( unsigned int segId, int zIndex )
{
    return mTileServer->GetSegmentZTileBounds( segId, zIndex );
}

void TileManager::ReplaceSegmentationLabel( unsigned int oldId, unsigned int newId )
{
    mTileServer->ReplaceSegmentationLabel( oldId, newId );

    ReloadTileCache();
}

void TileManager::ReplaceSegmentationLabelCurrentSlice( unsigned int oldId, unsigned int newId, Float3 pDataSpace )
{
    mTileServer->ReplaceSegmentationLabelCurrentSlice( oldId, newId, pDataSpace );

    ReloadTileCache();
}

void TileManager::DrawSplit( Float3 pointTileSpace, float radius )
{
    mTileServer->DrawSplit( pointTileSpace, radius );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::DrawErase( Float3 pointTileSpace, float radius )
{
    mTileServer->DrawErase( pointTileSpace, radius );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::DrawRegionA( Float3 pointTileSpace, float radius )
{
    mTileServer->DrawRegionA( pointTileSpace, radius );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::DrawRegionB( Float3 pointTileSpace, float radius )
{
    mTileServer->DrawRegionB( pointTileSpace, radius );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::AddSplitSource( Float3 pointTileSpace )
{
    mTileServer->AddSplitSource( pointTileSpace );
}

void TileManager::RemoveSplitSource()
{
    mTileServer->RemoveSplitSource();
}

void TileManager::ResetSplitState( Float3 pointTileSpace )
{
    mTileServer->ResetSplitState();

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::PrepForSplit( unsigned int segId, Float3 pointTileSpace )
{
    mTileServer->PrepForSplit( segId, pointTileSpace );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::FindBoundaryJoinPoints2D( unsigned int segId, Float3 pointTileSpace  )
{
    mTileServer->FindBoundaryJoinPoints2D( segId );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::FindBoundaryWithinRegion2D( unsigned int segId, Float3 pointTileSpace  )
{
    mTileServer->FindBoundaryWithinRegion2D( segId );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::FindBoundaryBetweenRegions2D( unsigned int segId, Float3 pointTileSpace  )
{
    mTileServer->FindBoundaryBetweenRegions2D( segId );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

int TileManager::CompletePointSplit( unsigned int segId, Float3 pointTileSpace )
{
    int newId = mTileServer->CompletePointSplit( segId, pointTileSpace );

    mTileServer->PrepForSplit( segId, pointTileSpace );

    ReloadTileCache();

    return newId;
}

int TileManager::CompleteDrawSplit( unsigned int segId, Float3 pointTileSpace, bool join3D, int splitStartZ )
{
    int newId = mTileServer->CompleteDrawSplit( segId, pointTileSpace, join3D, splitStartZ );

    mTileServer->PrepForSplit( segId, pointTileSpace );

    ReloadTileCache();

    return newId;
}

void TileManager::RecordSplitState( unsigned int segId, Float3 pointTileSpace )
{
    mTileServer->RecordSplitState( segId, pointTileSpace );
}

void TileManager::PredictSplit( unsigned int segId, Float3 pointTileSpace, float radius )
{
    mTileServer->PredictSplit( segId, pointTileSpace, radius );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::ResetAdjustState( Float3 pointTileSpace )
{
    mTileServer->ResetAdjustState();

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::PrepForAdjust( unsigned int segId, Float3 pointTileSpace )
{
    mTileServer->PrepForAdjust( segId, pointTileSpace );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::CommitAdjustChange( unsigned int segId, Float3 pointTileSpace )
{
    mTileServer->CommitAdjustChange( segId, pointTileSpace );

    mTileServer->PrepForAdjust( segId, pointTileSpace );

    ReloadTileCache();
}

void TileManager::ResetDrawMergeState( Float3 pointTileSpace )
{
    mTileServer->ResetDrawMergeState();

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

void TileManager::PrepForDrawMerge( Float3 pointTileSpace )
{
    mTileServer->PrepForDrawMerge( pointTileSpace );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );
}

unsigned int TileManager::CommitDrawMerge( Float3 pointTileSpace )
{
    std::set< unsigned int > remapIds = mTileServer->GetDrawMergeIds( pointTileSpace );

    unsigned int newId = 0;

    if ( remapIds.size() == 1 )
    {
        newId = *remapIds.begin();
        mTileServer->ResetDrawMergeState();
    }
    else if ( remapIds.size() > 1 )
    {
        newId = mTileServer->CommitDrawMerge( remapIds, pointTileSpace );

        for ( std::set< unsigned int >::iterator updateIt = remapIds.begin(); updateIt != remapIds.end(); ++updateIt )
        {
            UpdateLabelIdMap( *updateIt );
        }
    }

    mTileServer->PrepForDrawMerge( pointTileSpace );

    ReloadTileCacheOverlayMapOnly( (int)pointTileSpace.z );

    return newId;

}

unsigned int TileManager::CommitDrawMergeCurrentSlice( Float3 pointTileSpace )
{

    unsigned int newId = mTileServer->CommitDrawMergeCurrentSlice( pointTileSpace );

    mTileServer->PrepForDrawMerge( pointTileSpace );

    ReloadTileCache();

    return newId;

}

unsigned int TileManager::CommitDrawMergeCurrentConnectedComponent( Float3 pointTileSpace )
{

    unsigned int newId = mTileServer->CommitDrawMergeCurrentConnectedComponent( pointTileSpace );

    mTileServer->PrepForDrawMerge( pointTileSpace );

    ReloadTileCache();

    return newId;

}

void TileManager::ReplaceSegmentationLabelCurrentConnectedComponent( unsigned int oldId, unsigned int newId, Float3 pDataSpace )
{
    mTileServer->ReplaceSegmentationLabelCurrentConnectedComponent( oldId, newId, pDataSpace );

    ReloadTileCache();
}

unsigned int TileManager::GetNewId()
{
    return mTileServer->GetNewId();
}

float TileManager::GetCurrentOperationProgress()
{
    return mTileServer->GetCurrentOperationProgress();
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


void TileManager::TempSaveAndClearFileSystemTileCache()
{
    mTileServer->TempSaveAndClearFileSystemTileCache();
}

void TileManager::ClearFileSystemTileCache()
{
    mTileServer->ClearFileSystemTileCache();
}

Int3 TileManager::GetZoomLevel( const TiledDatasetView& tiledDatasetView )
{
    //
    // figure out what the current zoom level is
    //
    TiledVolumeDescription tiledvolumeDescription = mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" );

    double expZoomLevelXY   = ( tiledDatasetView.extentDataSpace.x * tiledvolumeDescription.numVoxelsPerTile.x ) / ( tiledDatasetView.widthNumPixels );

    int    zoomLevelXY      = (int)floor( std::min( (double)( tiledvolumeDescription.numTiles.w - 1 ), std::max( 0.0, ( log( expZoomLevelXY ) / log( 2.0 ) ) ) ) );
    int    zoomLevelZ       = 0;

    return Int3( zoomLevelXY, zoomLevelXY, zoomLevelZ );
}

std::list< Int4 > TileManager::GetTileIndicesIntersectedByView( const TiledDatasetView& tiledDatasetView )
{
    std::list< Int4 > tilesIntersectedByCamera;

    //
    // figure out what the current zoom level is
    //
    Int3 zoomLevel = GetZoomLevel( tiledDatasetView );
    int  zoomLevelXY = std::min( zoomLevel.x, zoomLevel.y );
    int  zoomLevelZ  = zoomLevel.z;

    //
    // figure out how many tiles there are at the current zoom level
    //
    Int4 numTiles          = mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles;
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
    Float3 topLeftFrontDataSpace =
        Float3(
            tiledDatasetView.centerDataSpace.x - ( tiledDatasetView.extentDataSpace.x / 2 ),
            tiledDatasetView.centerDataSpace.y - ( tiledDatasetView.extentDataSpace.y / 2 ),
            tiledDatasetView.centerDataSpace.z - ( tiledDatasetView.extentDataSpace.z / 2 ) );

    Float3 bottomRightBackDataSpace =
        Float3(
            tiledDatasetView.centerDataSpace.x + ( tiledDatasetView.extentDataSpace.x / 2 ),
            tiledDatasetView.centerDataSpace.y + ( tiledDatasetView.extentDataSpace.y / 2 ),
            tiledDatasetView.centerDataSpace.z + ( tiledDatasetView.extentDataSpace.z / 2 ) );

    //
    // compute the tile space indices for the top-left-front and bottom-right-back points
    //
    Int3 topLeftFrontTileIndex =
        Int3(
            (int)floor( topLeftFrontDataSpace.x / tileSizeDataSpaceX ),
            (int)floor( topLeftFrontDataSpace.y / tileSizeDataSpaceY ),
            (int)floor( topLeftFrontDataSpace.z / tileSizeDataSpaceZ ) );

    Int3 bottomRightBackTileIndex =
        Int3(
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
                tilesIntersectedByCamera.push_back( Int4( x, y, z, zoomLevelXY ) );

    return tilesIntersectedByCamera;
}

Int4 TileManager::GetTileIndexCoveringView( const TiledDatasetView& tiledDatasetView )
{
    Int4 tileCoveringView;

    //
    // figure out what the current zoom level is
    //
    Int3 zoomLevel = GetZoomLevel( tiledDatasetView );
    int  nextZoomLevelXY = std::min( zoomLevel.x, zoomLevel.y );
    int  zoomLevelZ  = zoomLevel.z;

    int minX = 0;
    int maxX = 1;
    int minY = 0;
    int maxY = 1;
    int minZ = 0;
    int maxZ = 1;

    int zoomLevelXY = 0;

    Int4 numTiles = mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles;

    //
    // Zoom out until there is only one tile visible
    //
    while ( minX < maxX || minY < maxY || minZ < maxZ )
    {
        zoomLevelXY = nextZoomLevelXY;

        if ( zoomLevelXY >= numTiles.w - 1 )
        {
            zoomLevelXY = numTiles.w - 1;
            minX = 0;
            minY = 0;
            minZ = (int)tiledDatasetView.centerDataSpace.z;
            break;
        }

        //
        // figure out how many tiles there are at the current zoom level
        //
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
        Float3 topLeftFrontDataSpace =
            Float3(
                tiledDatasetView.centerDataSpace.x - ( tiledDatasetView.extentDataSpace.x / 2 ),
                tiledDatasetView.centerDataSpace.y - ( tiledDatasetView.extentDataSpace.y / 2 ),
                tiledDatasetView.centerDataSpace.z - ( tiledDatasetView.extentDataSpace.z / 2 ) );

        Float3 bottomRightBackDataSpace =
            Float3(
                tiledDatasetView.centerDataSpace.x + ( tiledDatasetView.extentDataSpace.x / 2 ),
                tiledDatasetView.centerDataSpace.y + ( tiledDatasetView.extentDataSpace.y / 2 ),
                tiledDatasetView.centerDataSpace.z + ( tiledDatasetView.extentDataSpace.z / 2 ) );

        //
        // compute the tile space indices for the top-left-front and bottom-right-back points
        //
        Int3 topLeftFrontTileIndex =
            Int3(
                (int)floor( topLeftFrontDataSpace.x / tileSizeDataSpaceX ),
                (int)floor( topLeftFrontDataSpace.y / tileSizeDataSpaceY ),
                (int)floor( topLeftFrontDataSpace.z / tileSizeDataSpaceZ ) );

        Int3 bottomRightBackTileIndex =
            Int3(
                (int)floor( bottomRightBackDataSpace.x / tileSizeDataSpaceX ),
                (int)floor( bottomRightBackDataSpace.y / tileSizeDataSpaceY ),
                (int)floor( bottomRightBackDataSpace.z / tileSizeDataSpaceZ ) );

        //
        // clip the tiles to the appropriate tile space borders
        //
        minX = std::max( 0,                         topLeftFrontTileIndex.x );
        maxX = std::min( numTilesForZoomLevelX - 1, bottomRightBackTileIndex.x );
        minY = std::max( 0,                         topLeftFrontTileIndex.y );
        maxY = std::min( numTilesForZoomLevelY - 1, bottomRightBackTileIndex.y );
        minZ = std::max( 0,                         topLeftFrontTileIndex.z );
        maxZ = std::min( numTilesForZoomLevelZ - 1, bottomRightBackTileIndex.z );

        int maxTilesXY = std::max(maxX - minX + 1, maxY - minY + 1);

        if ( maxTilesXY > 2 )
        {
            nextZoomLevelXY = zoomLevelXY + (int) ceil ( log( (double)maxTilesXY ) / log( 2.0 ) );
        }
        else
        {
            nextZoomLevelXY = zoomLevelXY + 1;
        }

    }

    return Int4(minX, minY, minZ, zoomLevelXY);

}

void TileManager::GetIndexTileSpace( Int3 zoomLevel, Float3 pointDataSpace, Float4& pointTileSpace, Int4& tileIndex )
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
        Float4(
            (float)pointDataSpace.x / tileSizeDataSpaceX,
            (float)pointDataSpace.y / tileSizeDataSpaceY,
            (float)pointDataSpace.z / tileSizeDataSpaceZ,
            (float)zoomLevelXY );

    tileIndex =
        Int4(
            (int)floor( pointTileSpace.x ),
            (int)floor( pointTileSpace.y ),
            (int)floor( pointTileSpace.z ),
            (int)floor( pointTileSpace.w ) );
}

Int3 TileManager::GetIndexVoxelSpace( Float4 pointTileSpace, Int3 numVoxelsPerTile )
{
    //
    // compute the index in voxels within the full volume
    //
    Int3 indexVoxelSpace = 
        Int3(
            (int)floor( pointTileSpace.x * numVoxelsPerTile.x * pow( 2.0, (int)pointTileSpace.w ) ),
            (int)floor( pointTileSpace.y * numVoxelsPerTile.y * pow( 2.0, (int)pointTileSpace.w ) ),
            (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

    return indexVoxelSpace;
}

Int3 TileManager::GetOffsetVoxelSpace( Float4 pointTileSpace, Int4 tileIndex, Int3 numVoxelsPerTile )
{
    //
    // compute the offset (in data space) within the current tile 
    //
    Float4 tileIndexFloat4 =
        Float4(
            (float)tileIndex.x,
            (float)tileIndex.y,
            (float)tileIndex.z,
            (float)tileIndex.w );

    //
    // compute the offset in voxels within the current tile 
    //
    Int3 offsetVoxelSpace = 
        Int3(
            (int)floor( ( pointTileSpace.x - tileIndexFloat4.x ) * numVoxelsPerTile.x ),
            (int)floor( ( pointTileSpace.y - tileIndexFloat4.y ) * numVoxelsPerTile.y ),
            (int)floor( ( pointTileSpace.z - tileIndexFloat4.z ) * numVoxelsPerTile.z ) );

    return offsetVoxelSpace;
}

//
// CODE QUALITY ISSUE:
// The name of this function is confusing because there are two different caches: a host memory cache and a device memory cache.
// Which one is being reloaded? -MR
//
void TileManager::ReloadTileCache()
{
    for ( int cacheIndex = 0; cacheIndex < mDeviceTileCacheSize; cacheIndex++ )
    {
        if ( mTileCache[ cacheIndex ].indexTileSpace.x != TILE_CACHE_PAGE_TABLE_BAD_INDEX &&
             mTileCache[ cacheIndex ].indexTileSpace.y != TILE_CACHE_PAGE_TABLE_BAD_INDEX &&
             mTileCache[ cacheIndex ].indexTileSpace.z != TILE_CACHE_PAGE_TABLE_BAD_INDEX &&
             mTileCache[ cacheIndex ].indexTileSpace.w != TILE_CACHE_PAGE_TABLE_BAD_INDEX )
        {
            //
            // load image data into host memory
            //
            Int4 tileIndex = Int4 ( mTileCache[ cacheIndex ].indexTileSpace );
            HashMap< std::string, VolumeDescription > volumeDescriptions = mTileServer->LoadTile( tileIndex );

            //
            // load the new data into into device memory for the new cache entry
            //
            mTileCache[ cacheIndex ].d3d11Textures.Get( "SourceMap" )->Update( volumeDescriptions.Get( "SourceMap" ) );

            if ( mIsSegmentationLoaded )
            {
                mTileCache[ cacheIndex ].d3d11Textures.Get( "IdMap"      )->Update( volumeDescriptions.Get( "IdMap"      ) );
                mTileCache[ cacheIndex ].d3d11Textures.Get( "OverlayMap" )->Update( volumeDescriptions.Get( "OverlayMap" ) );
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
    for ( int cacheIndex = 0; cacheIndex < mDeviceTileCacheSize; cacheIndex++ )
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
            Int4 tileIndex = Int4 ( mTileCache[ cacheIndex ].indexTileSpace );
            HashMap< std::string, VolumeDescription > volumeDescriptions = mTileServer->LoadTile( tileIndex );

            //
            // load the new data into into device memory for the new cache entry
            //

            if ( mIsSegmentationLoaded )
            {
                mTileCache[ cacheIndex ].d3d11Textures.Get( "OverlayMap" )->Update( volumeDescriptions.Get( "OverlayMap" ) );
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