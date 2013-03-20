#pragma once

#define NOMINMAX
#include "Mojo.Core/Stl.hpp"
#include "Mojo.Core/MojoVectors.hpp"

#include <marray/marray.hxx>
#include <marray/marray_hdf5.hxx>

#include "Mojo.Core/Boost.hpp"
//#include "Mojo.Core/OpenCV.hpp"
#include "Mojo.Core/D3D11.hpp"
//#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/ID3D11CudaTexture.hpp"
#include "Mojo.Core/PrimitiveMap.hpp"
#include "Mojo.Core/D3D11CudaTextureMap.hpp"
#include "Mojo.Core/VolumeDescription.hpp"

#include "TiledDatasetDescription.hpp"
#include "TiledDatasetView.hpp"
#include "TileCacheEntry.hpp"
#include "ITileServer.hpp"
#include "SegmentInfo.hpp"
#include "Constants.hpp"

struct ID3D11Device;
struct ID3D11DeviceContext;

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
    void                                                  AutosaveSegmentation();
    void                                                  DeleteTempFiles();

    void                                                  Update();

    void                                                  LoadTiles( const TiledDatasetView& tiledDatasetView );

    boost::array< TileCacheEntry,
        DEVICE_TILE_CACHE_SIZE >&                         GetTileCache();
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

    unsigned int                                          GetSegmentationLabelId( const TiledDatasetView& tiledDatasetView, MojoFloat3 pDataSpace );
    MojoInt3                                              GetSegmentationLabelColor( unsigned int segId );
    std::string                                           GetSegmentationLabelColorString( unsigned int segId );
    MojoInt3                                              GetSegmentCentralTileLocation( unsigned int segId );
    MojoInt4                                              GetSegmentZTileBounds( unsigned int segId, int zIndex );


    void                                                  ReplaceSegmentationLabel( unsigned int oldId, unsigned int newId );
    void                                                  ReplaceSegmentationLabelCurrentSlice( unsigned int oldId, unsigned int newId, MojoFloat3 pDataSpace );
    void                                                  ReplaceSegmentationLabelCurrentConnectedComponent( unsigned int oldId, unsigned int newId, MojoFloat3 pDataSpace );

    void                                                  DrawSplit( MojoFloat3 pointTileSpace, float radius );
    void                                                  DrawErase( MojoFloat3 pointTileSpace, float radius );
    void                                                  DrawRegionB( MojoFloat3 pointTileSpace, float radius );
    void                                                  DrawRegionA( MojoFloat3 pointTileSpace, float radius );

    void                                                  AddSplitSource( MojoFloat3 pointTileSpace );
    void                                                  RemoveSplitSource();
    void                                                  ResetSplitState( MojoFloat3 pointTileSpace );
    void                                                  PrepForSplit( unsigned int segId, MojoFloat3 pointTileSpace );
	void                                                  FindBoundaryJoinPoints2D( unsigned int segId, MojoFloat3 pointTileSpace );
	void                                                  FindBoundaryWithinRegion2D( unsigned int segId, MojoFloat3 pointTileSpace );
	void                                                  FindBoundaryBetweenRegions2D( unsigned int segId, MojoFloat3 pointTileSpace );
    int                                                   CompletePointSplit( unsigned int segId, MojoFloat3 pointTileSpace );
    int                                                   CompleteDrawSplit( unsigned int segId, MojoFloat3 pointTileSpace, bool join3D, int splitStartZ );
    void                                                  RecordSplitState( unsigned int segId, MojoFloat3 pointTileSpace );
    void                                                  PredictSplit( unsigned int segId, MojoFloat3 pointTileSpace, float radius );

    void                                                  ResetAdjustState( MojoFloat3 pointTileSpace );
    void                                                  PrepForAdjust( unsigned int segId, MojoFloat3 pointTileSpace );
    void                                                  CommitAdjustChange( unsigned int segId, MojoFloat3 pointTileSpace );

    void                                                  ResetDrawMergeState( MojoFloat3 pointTileSpace );
    void                                                  PrepForDrawMerge( MojoFloat3 pointTileSpace );
    unsigned int                                          CommitDrawMerge( MojoFloat3 pointTileSpace );
    unsigned int                                          CommitDrawMergeCurrentSlice( MojoFloat3 pointTileSpace );
    unsigned int                                          CommitDrawMergeCurrentConnectedComponent( MojoFloat3 pointTileSpace );

	void                                                  UndoChange();
	void                                                  RedoChange();
    void                                                  TempSaveAndClearFileSystemTileCache();
    void                                                  ClearFileSystemTileCache();

    MojoInt3                                              GetZoomLevel( const TiledDatasetView& tiledDatasetView );

private:
    template < typename TCudaType >
    void                                                  LoadTiledDatasetInternal( TiledDatasetDescription& tiledDatasetDescription );
    void                                                  UnloadTiledDatasetInternal();

    template < typename TCudaType >
    void                                                  LoadSegmentationInternal( TiledDatasetDescription& tiledDatasetDescription );
    void                                                  UnloadSegmentationInternal();

    std::list< MojoInt4 >                                 GetTileIndicesIntersectedByView( const TiledDatasetView& tiledDatasetView );

    void                                                  GetIndexTileSpace( MojoInt3 zoomLevel, MojoFloat3 pointDataSpace, MojoFloat4& pointTileSpace, MojoInt4& tileIndex );
    MojoInt3                                              GetIndexVoxelSpace( MojoFloat4 pointTileSpace, MojoInt3 numVoxelsPerTile );
    MojoInt3                                              GetOffsetVoxelSpace( MojoFloat4 pTileSpace, MojoInt4 pTileIndex, MojoInt3 numVoxelsPerTile );

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

    Core::PrimitiveMap                                    mConstParameters;                                                
    TiledDatasetDescription                               mTiledDatasetDescription;

    boost::array< TileCacheEntry,
        DEVICE_TILE_CACHE_SIZE >                          mTileCache;

    int                                                   mTileCacheSearchStart;

    marray::Marray< int >                                 mTileCachePageTable;
    marray::Marray< unsigned char >*                      mIdColorMap;
	marray::Marray< unsigned int >*                       mLabelIdMap;
    marray::Marray< unsigned char >*                      mIdConfidenceMap;

    bool                                                  mIsTiledDatasetLoaded;
    bool                                                  mIsSegmentationLoaded;
};

template < typename TCudaType >
inline void TileManager::LoadTiledDatasetInternal( TiledDatasetDescription& tiledDatasetDescription )
{
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
    Core::Printf( "\n    Before allocating GPU memory:\n",
        "        Free memory:  ", (unsigned int) adapterDesc.DedicatedVideoMemory  / ( 1024 * 1024 ), " MBytes.\n" );
        //"        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );

    //
    // inform the tile server
    //
    mTileServer->LoadTiledDataset( tiledDatasetDescription );

    RELEASE_ASSERT( mTileServer->IsTiledDatasetLoaded() );

    //
    // store a refernce to the tiled dataset description
    //
    mTiledDatasetDescription = tiledDatasetDescription;

    TiledVolumeDescription tiledVolumeDescriptionId = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );
    TiledVolumeDescription tiledVolumeDescriptionSource = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" );
    TiledVolumeDescription tiledVolumeDescriptionOverlay = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "OverlayMap" );

    //
    // initialize the tile cache
    //
    for ( int i = 0; i < DEVICE_TILE_CACHE_SIZE; i++ )
    {
        TileCacheEntry tileCacheEntry;


		//
		// Assume tile sizes are the same
		// 
        int numVoxelsPerTile = tiledVolumeDescriptionId.numVoxelsPerTileX * tiledVolumeDescriptionId.numVoxelsPerTileY * tiledVolumeDescriptionId.numVoxelsPerTileZ;

        //tileCacheEntry.deviceVectors.Set( "IdMap", thrust::device_vector< int >( numVoxelsPerTile, mConstParameters.Get< int >( "ID_MAP_INITIAL_VALUE" ) ) );
        //tileCacheEntry.deviceVectors.Set( "OverlayMap", thrust::device_vector< int >( numVoxelsPerTile, mConstParameters.Get< int >( "ID_MAP_INITIAL_VALUE" ) ) );

        D3D11_TEXTURE3D_DESC textureDesc3D;
        ZeroMemory( &textureDesc3D, sizeof( D3D11_TEXTURE3D_DESC ) );

        textureDesc3D.Width     = tiledVolumeDescriptionSource.numVoxelsPerTileX;
        textureDesc3D.Height    = tiledVolumeDescriptionSource.numVoxelsPerTileY;
        textureDesc3D.Depth     = tiledVolumeDescriptionSource.numVoxelsPerTileZ;
        textureDesc3D.MipLevels = 1;
        textureDesc3D.Usage     = D3D11_USAGE_DEFAULT;
        textureDesc3D.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        textureDesc3D.Format    = tiledVolumeDescriptionSource.dxgiFormat;
        
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

        textureDesc3D.Width     = tiledVolumeDescriptionId.numVoxelsPerTileX;
        textureDesc3D.Height    = tiledVolumeDescriptionId.numVoxelsPerTileY;
        textureDesc3D.Depth     = tiledVolumeDescriptionId.numVoxelsPerTileZ;
        textureDesc3D.MipLevels = 1;
        textureDesc3D.Usage     = D3D11_USAGE_DEFAULT;
        textureDesc3D.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        textureDesc3D.Format    = tiledVolumeDescriptionId.dxgiFormat;

        tileCacheEntry.d3d11CudaTextures.Set(
            "IdMap",
            new Core::D3D11CudaTexture< ID3D11Texture3D, int >(
                mD3D11Device,
                mD3D11DeviceContext,
                textureDesc3D ) );
        
        ZeroMemory( &textureDesc3D, sizeof( D3D11_TEXTURE3D_DESC ) );

        textureDesc3D.Width     = tiledVolumeDescriptionOverlay.numVoxelsPerTileX;
        textureDesc3D.Height    = tiledVolumeDescriptionOverlay.numVoxelsPerTileY;
        textureDesc3D.Depth     = tiledVolumeDescriptionOverlay.numVoxelsPerTileZ;
        textureDesc3D.MipLevels = 1;
        textureDesc3D.Usage     = D3D11_USAGE_DEFAULT;
        textureDesc3D.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        textureDesc3D.Format    = tiledVolumeDescriptionOverlay.dxgiFormat;

        tileCacheEntry.d3d11CudaTextures.Set(
            "OverlayMap",
            new Core::D3D11CudaTexture< ID3D11Texture3D, int >(
                mD3D11Device,
                mD3D11DeviceContext,
                textureDesc3D ) );

        mTileCache[ i ] = tileCacheEntry;
    }

    mTileCacheSearchStart = 0;

    //
    // initialize the page table
    //
    MojoInt4 numTiles = tiledVolumeDescriptionSource.numTiles();

    size_t shape[] = { numTiles.w, numTiles.z, numTiles.y, numTiles.x };
    mTileCachePageTable = marray::Marray< int >( shape, shape + 4 );

    for ( int w = 0; w < numTiles.w; w++ )
        for ( int z = 0; z < numTiles.z; z++ )
            for ( int y = 0; y < numTiles.y; y++ )
                for ( int x = 0; x < numTiles.x; x++ )
                    mTileCachePageTable( w, z, y, x ) = TILE_CACHE_BAD_INDEX; 

    //
    // initialize all state
    //
    mIsTiledDatasetLoaded = true;

	//
	// Clear file system tile cache
	//
	mTileServer->ClearFileSystemTileCache();

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
    //memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
    //RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );

    pDXGIAdapter->GetDesc(&adapterDesc);

    Core::Printf( "    After allocating GPU memory:\n",
        "        Free memory:  ", (unsigned int) adapterDesc.DedicatedVideoMemory  / ( 1024 * 1024 ), " MBytes.\n" );
        //"        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );
}

template < typename TCudaType >
inline void TileManager::LoadSegmentationInternal( TiledDatasetDescription& tiledDatasetDescription )
{

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
    Core::Printf( "\n    Before allocating GPU memory:\n",
        "        Free memory:  ", (unsigned int) adapterDesc.DedicatedVideoMemory  / ( 1024 * 1024 ), " MBytes.\n" );
        //"        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );

    //
    // inform the tile server
    //
    mTileServer->LoadSegmentation( tiledDatasetDescription );

    RELEASE_ASSERT( mTileServer->IsSegmentationLoaded() );

    //
    // store a copy of the tiled dataset description
    //
    mTiledDatasetDescription = tiledDatasetDescription;

    TiledVolumeDescription tiledVolumeDescriptionSource = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" );

    //
    // reinitialize the page table
    //
    MojoInt4 numTiles = tiledVolumeDescriptionSource.numTiles();

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
    mIdColorMap = mTileServer->GetIdColorMap();

    unsigned char* idColorMap = new unsigned char[ mIdColorMap->shape( 0 ) * 4 ];

    unsigned int i;
    for ( i = 0; i < mIdColorMap->shape( 0 ); i++ )
	{
        idColorMap[ i * 4 ] = (*mIdColorMap)( i, 0 );
		idColorMap[ i * 4 + 1 ] = (*mIdColorMap)( i, 1 );
		idColorMap[ i * 4 + 2 ] = (*mIdColorMap)( i, 2 );
		idColorMap[ i * 4 + 3 ] = 255;
	}

    D3D11_BUFFER_DESC bufferDesc;
    ZeroMemory( &bufferDesc, sizeof( D3D11_BUFFER_DESC ) );

    bufferDesc.ByteWidth           = (UINT) mIdColorMap->shape( 0 ) * sizeof( unsigned char ) * 4;
    bufferDesc.Usage               = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags           = (UINT) D3D11_BIND_SHADER_RESOURCE;
    bufferDesc.StructureByteStride = (UINT) sizeof( unsigned char ) * 4;
    
    D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
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

	Core::Printf( "Loading Label Id map.");

    mLabelIdMap = mTileServer->GetLabelIdMap();

	size_t nSegmentsToAllocate = mLabelIdMap->shape( 0 );

	unsigned int* labelIdMap = new unsigned int [ nSegmentsToAllocate ];

	for ( i = 0; i < mLabelIdMap->shape( 0 ); ++i )
	{
		labelIdMap[ i ] = ( (*mLabelIdMap)( i ) );
	}

    //D3D11_BUFFER_DESC bufferDesc;
    ZeroMemory( &bufferDesc, sizeof( D3D11_BUFFER_DESC ) );

    bufferDesc.ByteWidth           = (UINT) nSegmentsToAllocate * sizeof( unsigned int );
    bufferDesc.Usage               = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags           = (UINT) D3D11_BIND_SHADER_RESOURCE;
    bufferDesc.StructureByteStride = (UINT) sizeof( unsigned int );
    
    //D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
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

	Core::Printf( "Loading Lock Map.");

    mIdConfidenceMap = mTileServer->GetIdConfidenceMap();

    unsigned char* idConfidenceMap = new unsigned char[ mIdConfidenceMap->shape( 0 ) ];

	for ( i = 0; i < mIdConfidenceMap->shape( 0 ); ++i )
	{
		idConfidenceMap[ i ] = ( (*mIdConfidenceMap)( i ) );
		//if ( (*mIdConfidenceMap)( i ) > 0 )
		//{
		//	Core::Printf( "Segment ", i, " is locked(", (*mIdConfidenceMap)( i ), ").");
		//}
	}

    //D3D11_BUFFER_DESC bufferDesc;
    ZeroMemory( &bufferDesc, sizeof( D3D11_BUFFER_DESC ) );

    bufferDesc.ByteWidth           = (UINT) nSegmentsToAllocate * sizeof( unsigned char );
    bufferDesc.Usage               = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags           = (UINT) D3D11_BIND_SHADER_RESOURCE;
    bufferDesc.StructureByteStride = (UINT) sizeof( unsigned char );
    
    //D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
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

    //
    // initialize all state
    //
    mIsSegmentationLoaded = true;

	//
	// Clear file system tile cache
	//
	mTileServer->ClearFileSystemTileCache();

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
    //memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
    //RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );

    pDXGIAdapter->GetDesc(&adapterDesc);

    Core::Printf( "    After allocating GPU memory:\n",
        "        Free memory:  ", (unsigned int) adapterDesc.DedicatedVideoMemory  / ( 1024 * 1024 ), " MBytes.\n" );
        //"        Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );
}

}
}