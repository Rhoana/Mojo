#pragma once

#include "Mojo.Core/Stl.hpp"

#include <marray/marray.hxx>
#include <marray/marray_hdf5.hxx>

#include "Mojo.Core/Boost.hpp"
#include "Mojo.Core/OpenCV.hpp"
#include "Mojo.Core/D3D11.hpp"
#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/ID3D11CudaTexture.hpp"
#include "Mojo.Core/PrimitiveMap.hpp"
#include "Mojo.Core/D3D11CudaTextureMap.hpp"
#include "Mojo.Core/VolumeDescription.hpp"

#include "TiledDatasetDescription.hpp"
#include "TiledDatasetView.hpp"
#include "TileCacheEntry.hpp"
#include "ITileServer.hpp"

struct ID3D11Device;
struct ID3D11DeviceContext;

const int DEVICE_TILE_CACHE_SIZE = 128;

namespace Mojo
{
namespace Native
{

class TileManager
{
public:
    TileManager( ID3D11Device* d3d11Device, ID3D11DeviceContext* d3d11DeviceContext, ITileServer* tileServer, Core::PrimitiveMap constParameters );
    ~TileManager();

    void                                                  LoadTiledDataset( TiledDatasetDescription& tiledDatasetDescription );
    void                                                  UnloadTiledDataset();

    bool                                                  IsTiledDatasetLoaded();

    void                                                  LoadSegmentation( TiledDatasetDescription& tiledDatasetDescription );
    void                                                  UnloadSegmentation();

    bool                                                  IsSegmentationLoaded();

    void                                                  SaveSegmentation();
    void                                                  SaveSegmentationAs( std::string savePath );

    void                                                  Update();

    void                                                  LoadTiles( const TiledDatasetView& tiledDatasetView );

    boost::array< TileCacheEntry,
        DEVICE_TILE_CACHE_SIZE >&                         GetTileCache();
    ID3D11ShaderResourceView*                             GetIdColorMap();

    int                                                   GetSegmentationLabelId( const TiledDatasetView& tiledDatasetView, float3 pDataSpace );
    int4                                                  GetSegmentationLabelColor( int id );

    void                                                  ReplaceSegmentationLabel( int oldId, int newId );
    void                                                  ReplaceSegmentationLabelCurrentSlice( int oldId, int newId, float3 pDataSpace );
    void                                                  ReplaceSegmentationLabelCurrentConnectedComponent( int oldId, int newId, float3 pDataSpace );

    void                                                  DrawSplit( float3 pointTileSpace, float radius );
    void                                                  DrawErase( float3 pointTileSpace, float radius );
    void                                                  DrawRegionB( float3 pointTileSpace, float radius );
    void                                                  DrawRegionA( float3 pointTileSpace, float radius );
    void                                                  AddSplitSource( float3 pointTileSpace );
    void                                                  RemoveSplitSource();
    void                                                  ResetSplitState();
    void                                                  PrepForSplit( int segId, float3 pointTileSpace );
	void                                                  FindBoundaryJoinPoints2D( int segId );
	void                                                  FindBoundaryWithinRegion2D( int segId );
	void                                                  FindBoundaryBetweenRegions2D( int segId );
    int                                                   CompletePointSplit( int segId );
    int                                                   CompleteDrawSplit( int segId );

	void                                                  UndoChange();
	void                                                  RedoChange();
    void                                                  SaveAndClearFileSystemTileCache();

private:
    template < typename TCudaType >
    void                                                  LoadTiledDatasetInternal( TiledDatasetDescription& tiledDatasetDescription );
    void                                                  UnloadTiledDatasetInternal();

    template < typename TCudaType >
    void                                                  LoadSegmentationInternal( TiledDatasetDescription& tiledDatasetDescription );
    void                                                  UnloadSegmentationInternal();

    int3                                                  GetZoomLevel( const TiledDatasetView& tiledDatasetView );
    std::list< int4 >                                     GetTileIndicesIntersectedByView( const TiledDatasetView& tiledDatasetView );

    void                                                  GetIndexTileSpace( int3 zoomLevel, float3 pointDataSpace, float4& pointTileSpace, int4& tileIndex );
    int3                                                  GetIndexVoxelSpace( float4 pointTileSpace, int3 numVoxelsPerTile );
    int3                                                  GetOffsetVoxelSpace( float4 pTileSpace, int4 pTileIndex, int3 numVoxelsPerTile );

    void                                                  ReloadTileCache();
    void                                                  ReloadTileCacheOverlayMapOnly();

    ID3D11Device*                                         mD3D11Device;
    ID3D11DeviceContext*                                  mD3D11DeviceContext;
    ID3D11Buffer*                                         mIdColorMapBuffer;
    ID3D11ShaderResourceView*                             mIdColorMapShaderResourceView;

    ITileServer*                                          mTileServer;

    Core::PrimitiveMap                                    mConstParameters;                                                
    TiledDatasetDescription                               mTiledDatasetDescription;

    boost::array< TileCacheEntry,
        DEVICE_TILE_CACHE_SIZE >                          mTileCache;
    marray::Marray< int >                                 mTileCachePageTable;
    marray::Marray< unsigned char >                       mIdColorMap;

    bool                                                  mIsTiledDatasetLoaded;
    bool                                                  mIsSegmentationLoaded;
};

template < typename TCudaType >
inline void TileManager::LoadTiledDatasetInternal( TiledDatasetDescription& tiledDatasetDescription )
{
    //
    // output memory stats to the console
    //
    size_t freeMemory, totalMemory;
    CUresult     memInfoResult;
    memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
    RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );
    Core::Printf( "\nLoading tiled dataset...\n" );
    Core::Printf( "\n    Before allocating GPU memory:\n",
                  "        Free memory:  ", (unsigned int) freeMemory  / ( 1024 * 1024 ), " MBytes.\n",
                  "        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );

    //
    // inform the tile server
    //
    mTileServer->LoadTiledDataset( tiledDatasetDescription );

    RELEASE_ASSERT( mTileServer->IsTiledDatasetLoaded() );

    //
    // store a refernce to the tiled dataset description
    //
    mTiledDatasetDescription = tiledDatasetDescription;

    //
    // initialize the tile cache
    //
    for ( int i = 0; i < DEVICE_TILE_CACHE_SIZE; i++ )
    {
        TileCacheEntry tileCacheEntry;


		//
		// Assume tile sizes are the same
		// 
        int numVoxelsPerTile =
            mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile.x *
            mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile.y *
            mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile.z;

        tileCacheEntry.deviceVectors.Set( "IdMap", thrust::device_vector< int >( numVoxelsPerTile, mConstParameters.Get< int >( "ID_MAP_INITIAL_VALUE" ) ) );
        tileCacheEntry.deviceVectors.Set( "OverlayMap", thrust::device_vector< int >( numVoxelsPerTile, mConstParameters.Get< int >( "ID_MAP_INITIAL_VALUE" ) ) );


        D3D11_TEXTURE3D_DESC textureDesc3D;
        ZeroMemory( &textureDesc3D, sizeof( D3D11_TEXTURE3D_DESC ) );

        textureDesc3D.Width     = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numVoxelsPerTile.x;
        textureDesc3D.Height    = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numVoxelsPerTile.y;
        textureDesc3D.Depth     = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numVoxelsPerTile.z;
        textureDesc3D.MipLevels = 1;
        textureDesc3D.Usage     = D3D11_USAGE_DEFAULT;
        textureDesc3D.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        textureDesc3D.Format    = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).dxgiFormat;
        
        tileCacheEntry.d3d11CudaTextures.Set(
            "SourceMap",
            new Core::D3D11CudaTexture< ID3D11Texture3D, TCudaType >(
                mD3D11Device,
                mD3D11DeviceContext,
                textureDesc3D ) );


		//
		// Allocate memory for the IdMap and OverlayMap tiles here
		// (data will be loaded in LoadSegmentationInternal)
		//
        ZeroMemory( &textureDesc3D, sizeof( D3D11_TEXTURE3D_DESC ) );

        textureDesc3D.Width     = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile.x;
        textureDesc3D.Height    = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile.y;
        textureDesc3D.Depth     = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile.z;
        textureDesc3D.MipLevels = 1;
        textureDesc3D.Usage     = D3D11_USAGE_DEFAULT;
        textureDesc3D.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        textureDesc3D.Format    = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).dxgiFormat;
        
        tileCacheEntry.d3d11CudaTextures.Set(
            "IdMap",
            new Core::D3D11CudaTexture< ID3D11Texture3D, int >(
                mD3D11Device,
                mD3D11DeviceContext,
                textureDesc3D,
                mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile,
                tileCacheEntry.deviceVectors.Get< int >( "IdMap" ) ) );
        
        tileCacheEntry.d3d11CudaTextures.Set(
            "OverlayMap",
            new Core::D3D11CudaTexture< ID3D11Texture3D, int >(
                mD3D11Device,
                mD3D11DeviceContext,
                textureDesc3D,
                mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile,
                tileCacheEntry.deviceVectors.Get< int >( "OverlayMap" ) ) );

        mTileCache[ i ] = tileCacheEntry;
    }

    //
    // initialize the page table
    //
    int4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles;

    size_t shape[] = { numTiles.w, numTiles.z, numTiles.y, numTiles.x };
    mTileCachePageTable = marray::Marray< int >( shape, shape + 4 );

    for ( int w = 0; w < numTiles.w; w++ )
        for ( int z = 0; z < numTiles.z; z++ )
            for ( int y = 0; y < numTiles.y; y++ )
                for ( int x = 0; x < numTiles.x; x++ )
                    mTileCachePageTable( w, z, y, x ) = TILE_CACHE_BAD_INDEX; 

    //
    // load the id color map
    //
    //hid_t hdf5FileHandle = marray::hdf5::openFile( mTiledDatasetDescription.paths.Get( "IdColorMap" ) );
    //marray::hdf5::load( hdf5FileHandle, "IdColorMap", mIdColorMap );
    //marray::hdf5::closeFile( hdf5FileHandle );

    //uchar4* idColorMap = new uchar4[ mIdColorMap.shape( 0 ) ];

    //for ( unsigned int i = 0; i < mIdColorMap.shape( 0 ); i++ )
    //    idColorMap[ i ] = make_uchar4( mIdColorMap( i, 0 ), mIdColorMap( i, 1 ), mIdColorMap( i, 2 ), 255 );

    //D3D11_BUFFER_DESC bufferDesc;
    //ZeroMemory( &bufferDesc, sizeof( D3D11_BUFFER_DESC ) );

    //bufferDesc.ByteWidth           = mIdColorMap.shape( 0 ) * sizeof( uchar4 );
    //bufferDesc.Usage               = D3D11_USAGE_DEFAULT;
    //bufferDesc.BindFlags           = D3D11_BIND_SHADER_RESOURCE;
    //bufferDesc.StructureByteStride = sizeof( uchar4 );
    //
    //D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
    //ZeroMemory( &shaderResourceViewDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );

    //shaderResourceViewDesc.Format               = DXGI_FORMAT_R8G8B8A8_UNORM;
    //shaderResourceViewDesc.ViewDimension        = D3D11_SRV_DIMENSION_BUFFER;
    //shaderResourceViewDesc.Buffer.FirstElement  = 0;
    //shaderResourceViewDesc.Buffer.NumElements   = mIdColorMap.shape( 0 );

    //MOJO_D3D_SAFE( mD3D11Device->CreateBuffer( &bufferDesc, NULL, &mIdColorMapBuffer ) );
    //MOJO_D3D_SAFE( mD3D11Device->CreateShaderResourceView( mIdColorMapBuffer, &shaderResourceViewDesc, &mIdColorMapShaderResourceView ) );

    //mD3D11DeviceContext->UpdateSubresource(
    //    mIdColorMapBuffer,
    //    0,
    //    NULL,
    //    idColorMap,
    //    mIdColorMap.shape( 0 ) * sizeof( uchar4 ),
    //    mIdColorMap.shape( 0 ) * sizeof( uchar4 ) );

    //
    // initialize all state
    //
    mIsTiledDatasetLoaded = true;

    //
    // load tiles into the cache
    //
    Update();

    //
    // output memory stats to the console
    //
    memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
    RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );
    Core::Printf( "    After allocating GPU memory:\n",
                  "        Free memory:  ", (unsigned int) freeMemory  / ( 1024 * 1024 ), " MBytes.\n",
                  "        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );
}

template < typename TCudaType >
inline void TileManager::LoadSegmentationInternal( TiledDatasetDescription& tiledDatasetDescription )
{

    //
    // output memory stats to the console
    //
    size_t freeMemory, totalMemory;
    CUresult     memInfoResult;
    memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
    RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );
    Core::Printf( "\nLoading segmentation...\n" );
    Core::Printf( "\n    Before allocating GPU memory:\n",
                  "        Free memory:  ", (unsigned int) freeMemory  / ( 1024 * 1024 ), " MBytes.\n",
                  "        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );

    //
    // inform the tile server
    //
    mTileServer->LoadSegmentation( tiledDatasetDescription );

    RELEASE_ASSERT( mTileServer->IsSegmentationLoaded() );

    //
    // store a copy of the tiled dataset description
    //
    mTiledDatasetDescription = tiledDatasetDescription;

    //
    // initialize the tile cache
	// (Overrides initial allocation done in LoadTiledDatasetInternal)
    //
    //for ( int i = 0; i < DEVICE_TILE_CACHE_SIZE; i++ )
    //{

    //    int numVoxelsPerTile =
    //        mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile.x *
    //        mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile.y *
    //        mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile.z;

    //    mTileCache[ i ].deviceVectors.Set( "IdMap", thrust::device_vector< int >( numVoxelsPerTile, mConstParameters.Get< int >( "ID_MAP_INITIAL_VALUE" ) ) );


    //    D3D11_TEXTURE3D_DESC textureDesc3D;
    //    ZeroMemory( &textureDesc3D, sizeof( D3D11_TEXTURE3D_DESC ) );

    //    textureDesc3D.Width     = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile.x;
    //    textureDesc3D.Height    = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile.y;
    //    textureDesc3D.Depth     = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile.z;
    //    textureDesc3D.MipLevels = 1;
    //    textureDesc3D.Usage     = D3D11_USAGE_DEFAULT;
    //    textureDesc3D.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    //    textureDesc3D.Format    = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).dxgiFormat;
    //    
    //    mTileCache[ i ].d3d11CudaTextures.Set(
    //        "IdMap",
    //        new Core::D3D11CudaTexture< ID3D11Texture3D, int >(
    //            mD3D11Device,
    //            mD3D11DeviceContext,
    //            textureDesc3D,
    //            mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile,
    //            mTileCache[ i ].deviceVectors.Get< int >( "IdMap" ) ) );

    //}

    //
    // reinitialize the page table
    //
    int4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles;

    //size_t shape[] = { numTiles.w, numTiles.z, numTiles.y, numTiles.x };
    //mTileCachePageTable = marray::Marray< int >( shape, shape + 4 );

    for ( int w = 0; w < numTiles.w; w++ )
        for ( int z = 0; z < numTiles.z; z++ )
            for ( int y = 0; y < numTiles.y; y++ )
                for ( int x = 0; x < numTiles.x; x++ )
                    mTileCachePageTable( w, z, y, x ) = TILE_CACHE_BAD_INDEX; 

	//
    // load the id color map
    //
    hid_t hdf5FileHandle = marray::hdf5::openFile( mTiledDatasetDescription.paths.Get( "IdColorMap" ) );
    marray::hdf5::load( hdf5FileHandle, "IdColorMap", mIdColorMap );
    marray::hdf5::closeFile( hdf5FileHandle );

    uchar4* idColorMap = new uchar4[ mIdColorMap.shape( 0 ) ];

    unsigned int i;
    for ( i = 0; i < mIdColorMap.shape( 0 ); i++ )
        idColorMap[ i ] = make_uchar4( mIdColorMap( i, 0 ), mIdColorMap( i, 1 ), mIdColorMap( i, 2 ), 255 );

    D3D11_BUFFER_DESC bufferDesc;
    ZeroMemory( &bufferDesc, sizeof( D3D11_BUFFER_DESC ) );

    bufferDesc.ByteWidth           = (UINT) mIdColorMap.shape( 0 ) * sizeof( uchar4 );
    bufferDesc.Usage               = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags           = (UINT) D3D11_BIND_SHADER_RESOURCE;
    bufferDesc.StructureByteStride = (UINT) sizeof( uchar4 );
    
    D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
    ZeroMemory( &shaderResourceViewDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );

    shaderResourceViewDesc.Format               = DXGI_FORMAT_R8G8B8A8_UNORM;
    shaderResourceViewDesc.ViewDimension        = D3D11_SRV_DIMENSION_BUFFER;
    shaderResourceViewDesc.Buffer.FirstElement  = (UINT) 0;
    shaderResourceViewDesc.Buffer.NumElements   = (UINT) mIdColorMap.shape( 0 );

    MOJO_D3D_SAFE( mD3D11Device->CreateBuffer( &bufferDesc, NULL, &mIdColorMapBuffer ) );
    MOJO_D3D_SAFE( mD3D11Device->CreateShaderResourceView( mIdColorMapBuffer, &shaderResourceViewDesc, &mIdColorMapShaderResourceView ) );

    mD3D11DeviceContext->UpdateSubresource(
        mIdColorMapBuffer,
        0,
        NULL,
        idColorMap,
        (UINT) mIdColorMap.shape( 0 ) * sizeof( uchar4 ),
        (UINT) mIdColorMap.shape( 0 ) * sizeof( uchar4 ) );

    //
    // initialize all state
    //
    mIsSegmentationLoaded = true;

	//
	// Clear file system tile cache
	//
	mTileServer->SaveAndClearFileSystemTileCache();

	//
	// Reload the device cache
	//
	ReloadTileCache();

	//
    // load tiles into the cache
    //
    Update();

    //
    // output memory stats to the console
    //
    memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
    RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );
    Core::Printf( "    After allocating GPU memory:\n",
                  "        Free memory:  ", (unsigned int) freeMemory  / ( 1024 * 1024 ), " MBytes.\n",
                  "        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );
}

}
}