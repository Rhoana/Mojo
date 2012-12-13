#pragma once

#include "Assert.hpp"
#include "D3D11.hpp"
#include "Cuda.hpp"
#include "Thrust.hpp"
#include "ID3D11CudaTexture.hpp"

namespace Mojo
{
namespace Core
{

template < typename TD3D11ResourceType, typename TCudaType >
class D3D11CudaTexture : public ID3D11CudaTexture
{
public:
    //
    // 1D
    //
    D3D11CudaTexture( ID3D11Device*                       d3d11Device,
                      ID3D11DeviceContext*                d3d11DeviceContext,
                      D3D11_TEXTURE1D_DESC                resourceDesc );
                                                          
    D3D11CudaTexture( ID3D11Device*                       d3d11Device,
                      ID3D11DeviceContext*                d3d11DeviceContext,
                      D3D11_TEXTURE1D_DESC                resourceDesc,
                      VolumeDescription                   volumeDescription );
                                                          
    D3D11CudaTexture( ID3D11Device*                       d3d11Device,
                      ID3D11DeviceContext*                d3d11DeviceContext,
                      D3D11_TEXTURE1D_DESC                resourceDesc,
                      int3                                numVoxels,
                      thrust::device_vector< TCudaType >& deviceVector );

    //
    // 2D
    //
    D3D11CudaTexture( ID3D11Device*                       d3d11Device,
                      ID3D11DeviceContext*                d3d11DeviceContext,
                      D3D11_TEXTURE2D_DESC                resourceDesc );
                                                          
    D3D11CudaTexture( ID3D11Device*                       d3d11Device,
                      ID3D11DeviceContext*                d3d11DeviceContext,
                      D3D11_TEXTURE2D_DESC                resourceDesc,
                      VolumeDescription                   volumeDescription );
                                                          
    D3D11CudaTexture( ID3D11Device*                       d3d11Device,
                      ID3D11DeviceContext*                d3d11DeviceContext,
                      D3D11_TEXTURE2D_DESC                resourceDesc,
                      int3                                numVoxels,
                      thrust::device_vector< TCudaType >& deviceVector );

    //
    // 3D
    //
    D3D11CudaTexture( ID3D11Device*                       d3d11Device,
                      ID3D11DeviceContext*                d3d11DeviceContext,
                      D3D11_TEXTURE3D_DESC                resourceDesc );
                                                          
    D3D11CudaTexture( ID3D11Device*                       d3d11Device,
                      ID3D11DeviceContext*                d3d11DeviceContext,
                      D3D11_TEXTURE3D_DESC                resourceDesc,
                      VolumeDescription                   volumeDescription );
                                                          
    D3D11CudaTexture( ID3D11Device*                       d3d11Device,
                      ID3D11DeviceContext*                d3d11DeviceContext,
                      D3D11_TEXTURE3D_DESC                resourceDesc,
                      int3                                numVoxels,
                      thrust::device_vector< TCudaType >& deviceVector );

    //
    // general-purpose methods
    //
    virtual ~D3D11CudaTexture();

    virtual void                      Update( VolumeDescription volumeDescription );

    virtual void                      MapCudaArray();
    virtual void                      UnmapCudaArray();
    virtual cudaArray*                GetMappedCudaArray( int mipLevel = 0 );

    virtual cudaGraphicsResource*     GetCudaGraphicsResource();
    virtual ID3D11ShaderResourceView* GetD3D11ShaderResourceView();

private:
    cudaGraphicsResource*     mCudaGraphicsResource;
    TD3D11ResourceType*       mD3D11Resource;
    ID3D11ShaderResourceView* mD3D11ShaderResourceView;

    ID3D11DeviceContext*      mD3D11DeviceContext;
};

//
// 1D
//
template < typename TD3D11ResourceType, typename TCudaType >
D3D11CudaTexture< TD3D11ResourceType, TCudaType >::D3D11CudaTexture( ID3D11Device*        d3d11Device,
                                                                     ID3D11DeviceContext* d3d11DeviceContext,
                                                                     D3D11_TEXTURE1D_DESC resourceDesc ) :
    mCudaGraphicsResource   ( NULL ),
    mD3D11Resource          ( NULL ),
    mD3D11ShaderResourceView( NULL ),
    mD3D11DeviceContext     ( d3d11DeviceContext )
{
    D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
    ZeroMemory( &shaderResourceViewDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );

    shaderResourceViewDesc.Format              = resourceDesc.Format;
    shaderResourceViewDesc.ViewDimension       = D3D11_SRV_DIMENSION_TEXTURE1D;
    shaderResourceViewDesc.Texture1D.MipLevels = -1;

    MOJO_D3D_SAFE( d3d11Device->CreateTexture1D( &resourceDesc, NULL, &mD3D11Resource ) );
    MOJO_D3D_SAFE( d3d11Device->CreateShaderResourceView( mD3D11Resource, &shaderResourceViewDesc, &mD3D11ShaderResourceView ) );
    MOJO_CUDA_SAFE( cudaGraphicsD3D11RegisterResource( &mCudaGraphicsResource, mD3D11Resource, cudaGraphicsRegisterFlagsNone ) );
}

template < typename TD3D11ResourceType, typename TCudaType >
D3D11CudaTexture< TD3D11ResourceType, TCudaType >::D3D11CudaTexture( ID3D11Device*        d3d11Device,
                                                                     ID3D11DeviceContext* d3d11DeviceContext,
                                                                     D3D11_TEXTURE1D_DESC resourceDesc,
                                                                     VolumeDescription    volumeDescription ) :
    mCudaGraphicsResource   ( NULL ),
    mD3D11Resource          ( NULL ),
    mD3D11ShaderResourceView( NULL ),
    mD3D11DeviceContext     ( d3d11DeviceContext )
{
    D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
    ZeroMemory( &shaderResourceViewDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );

    shaderResourceViewDesc.Format              = resourceDesc.Format;
    shaderResourceViewDesc.ViewDimension       = D3D11_SRV_DIMENSION_TEXTURE1D;
    shaderResourceViewDesc.Texture1D.MipLevels = -1;

    MOJO_D3D_SAFE( d3d11Device->CreateTexture1D( &resourceDesc, NULL, &mD3D11Resource ) );
    MOJO_D3D_SAFE( d3d11Device->CreateShaderResourceView( mD3D11Resource, &shaderResourceViewDesc, &mD3D11ShaderResourceView ) );
    MOJO_CUDA_SAFE( cudaGraphicsD3D11RegisterResource( &mCudaGraphicsResource, mD3D11Resource, cudaGraphicsRegisterFlagsNone ) );

    mD3D11DeviceContext->UpdateSubresource(
        mD3D11Resource,
        0,
        NULL,
        volumeDescription.data,
        volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel,
        volumeDescription.numVoxels.y * volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel );
}

template < typename TD3D11ResourceType, typename TCudaType >
D3D11CudaTexture< TD3D11ResourceType, TCudaType >::D3D11CudaTexture( ID3D11Device*                       d3d11Device,
                                                                     ID3D11DeviceContext*                d3d11DeviceContext,
                                                                     D3D11_TEXTURE1D_DESC                resourceDesc,
                                                                     int3                                numVoxels,
                                                                     thrust::device_vector< TCudaType >& deviceVector ) :
    mCudaGraphicsResource   ( NULL ),
    mD3D11Resource          ( NULL ),
    mD3D11ShaderResourceView( NULL ),
    mD3D11DeviceContext     ( d3d11DeviceContext )
{
    D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
    ZeroMemory( &shaderResourceViewDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );

    shaderResourceViewDesc.Format              = resourceDesc.Format;
    shaderResourceViewDesc.ViewDimension       = D3D11_SRV_DIMENSION_TEXTURE1D;
    shaderResourceViewDesc.Texture1D.MipLevels = -1;

    MOJO_D3D_SAFE( d3d11Device->CreateTexture1D( &resourceDesc, NULL, &mD3D11Resource ) );
    MOJO_D3D_SAFE( d3d11Device->CreateShaderResourceView( mD3D11Resource, &shaderResourceViewDesc, &mD3D11ShaderResourceView ) );
    MOJO_CUDA_SAFE( cudaGraphicsD3D11RegisterResource( &mCudaGraphicsResource, mD3D11Resource, cudaGraphicsRegisterFlagsNone ) );

    MapCudaArray();
    cudaArray* cudaArray = GetMappedCudaArray();
    Thrust::Memcpy2DToArray( cudaArray, deviceVector, numVoxels );
    UnmapCudaArray();
}

//
// 2D
//
template < typename TD3D11ResourceType, typename TCudaType >
D3D11CudaTexture< TD3D11ResourceType, TCudaType >::D3D11CudaTexture( ID3D11Device*        d3d11Device,
                                                                     ID3D11DeviceContext* d3d11DeviceContext,
                                                                     D3D11_TEXTURE2D_DESC resourceDesc ) :
    mCudaGraphicsResource   ( NULL ),
    mD3D11Resource          ( NULL ),
    mD3D11ShaderResourceView( NULL ),
    mD3D11DeviceContext     ( d3d11DeviceContext )
{
    D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
    ZeroMemory( &shaderResourceViewDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );

    shaderResourceViewDesc.Format              = resourceDesc.Format;
    shaderResourceViewDesc.ViewDimension       = D3D11_SRV_DIMENSION_TEXTURE2D;
    shaderResourceViewDesc.Texture2D.MipLevels = -1;

    MOJO_D3D_SAFE( d3d11Device->CreateTexture2D( &resourceDesc, NULL, &mD3D11Resource ) );
    MOJO_D3D_SAFE( d3d11Device->CreateShaderResourceView( mD3D11Resource, &shaderResourceViewDesc, &mD3D11ShaderResourceView ) );
    MOJO_CUDA_SAFE( cudaGraphicsD3D11RegisterResource( &mCudaGraphicsResource, mD3D11Resource, cudaGraphicsRegisterFlagsNone ) );
}

template < typename TD3D11ResourceType, typename TCudaType >
D3D11CudaTexture< TD3D11ResourceType, TCudaType >::D3D11CudaTexture( ID3D11Device*        d3d11Device,
                                                                     ID3D11DeviceContext* d3d11DeviceContext,
                                                                     D3D11_TEXTURE2D_DESC resourceDesc,
                                                                     VolumeDescription    volumeDescription ) :
    mCudaGraphicsResource   ( NULL ),
    mD3D11Resource          ( NULL ),
    mD3D11ShaderResourceView( NULL ),
    mD3D11DeviceContext     ( d3d11DeviceContext )
{
    D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
    ZeroMemory( &shaderResourceViewDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );

    shaderResourceViewDesc.Format              = resourceDesc.Format;
    shaderResourceViewDesc.ViewDimension       = D3D11_SRV_DIMENSION_TEXTURE2D;
    shaderResourceViewDesc.Texture2D.MipLevels = -1;

    MOJO_D3D_SAFE( d3d11Device->CreateTexture2D( &resourceDesc, NULL, &mD3D11Resource ) );
    MOJO_D3D_SAFE( d3d11Device->CreateShaderResourceView( mD3D11Resource, &shaderResourceViewDesc, &mD3D11ShaderResourceView ) );
    MOJO_CUDA_SAFE( cudaGraphicsD3D11RegisterResource( &mCudaGraphicsResource, mD3D11Resource, cudaGraphicsRegisterFlagsNone ) );

    mD3D11DeviceContext->UpdateSubresource(
        mD3D11Resource,
        0,
        NULL,
        volumeDescription.data,
        volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel,
        volumeDescription.numVoxels.y * volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel );
}

template < typename TD3D11ResourceType, typename TCudaType >
D3D11CudaTexture< TD3D11ResourceType, TCudaType >::D3D11CudaTexture( ID3D11Device*                       d3d11Device,
                                                                     ID3D11DeviceContext*                d3d11DeviceContext,
                                                                     D3D11_TEXTURE2D_DESC                resourceDesc,
                                                                     int3                                numVoxels,
                                                                     thrust::device_vector< TCudaType >& deviceVector ) :
    mCudaGraphicsResource   ( NULL ),
    mD3D11Resource          ( NULL ),
    mD3D11ShaderResourceView( NULL ),
    mD3D11DeviceContext     ( d3d11DeviceContext )
{
    D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
    ZeroMemory( &shaderResourceViewDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );

    shaderResourceViewDesc.Format              = resourceDesc.Format;
    shaderResourceViewDesc.ViewDimension       = D3D11_SRV_DIMENSION_TEXTURE2D;
    shaderResourceViewDesc.Texture2D.MipLevels = -1;

    MOJO_D3D_SAFE( d3d11Device->CreateTexture2D( &resourceDesc, NULL, &mD3D11Resource ) );
    MOJO_D3D_SAFE( d3d11Device->CreateShaderResourceView( mD3D11Resource, &shaderResourceViewDesc, &mD3D11ShaderResourceView ) );
    MOJO_CUDA_SAFE( cudaGraphicsD3D11RegisterResource( &mCudaGraphicsResource, mD3D11Resource, cudaGraphicsRegisterFlagsNone ) );

    MapCudaArray();
    cudaArray* cudaArray = GetMappedCudaArray();
    Thrust::Memcpy2DToArray( cudaArray, deviceVector, numVoxels );
    UnmapCudaArray();
}

//
// 3D
//
template < typename TD3D11ResourceType, typename TCudaType >
D3D11CudaTexture< TD3D11ResourceType, TCudaType >::D3D11CudaTexture( ID3D11Device*        d3d11Device,
                                                                     ID3D11DeviceContext* d3d11DeviceContext,
                                                                     D3D11_TEXTURE3D_DESC resourceDesc ) :
    mCudaGraphicsResource   ( NULL ),
    mD3D11Resource          ( NULL ),
    mD3D11ShaderResourceView( NULL ),
    mD3D11DeviceContext     ( d3d11DeviceContext )
{
    D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
    ZeroMemory( &shaderResourceViewDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );

    shaderResourceViewDesc.Format              = resourceDesc.Format;
    shaderResourceViewDesc.ViewDimension       = D3D11_SRV_DIMENSION_TEXTURE3D;
    shaderResourceViewDesc.Texture3D.MipLevels = -1;

    MOJO_D3D_SAFE( d3d11Device->CreateTexture3D( &resourceDesc, NULL, &mD3D11Resource ) );
    MOJO_D3D_SAFE( d3d11Device->CreateShaderResourceView( mD3D11Resource, &shaderResourceViewDesc, &mD3D11ShaderResourceView ) );
    MOJO_CUDA_SAFE( cudaGraphicsD3D11RegisterResource( &mCudaGraphicsResource, mD3D11Resource, cudaGraphicsRegisterFlagsNone ) );
}

template < typename TD3D11ResourceType, typename TCudaType >
D3D11CudaTexture< TD3D11ResourceType, TCudaType >::D3D11CudaTexture( ID3D11Device*        d3d11Device,
                                                                     ID3D11DeviceContext* d3d11DeviceContext,
                                                                     D3D11_TEXTURE3D_DESC resourceDesc,
                                                                     VolumeDescription    volumeDescription ) :
    mCudaGraphicsResource   ( NULL ),
    mD3D11Resource          ( NULL ),
    mD3D11ShaderResourceView( NULL ),
    mD3D11DeviceContext     ( d3d11DeviceContext )
{
    D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
    ZeroMemory( &shaderResourceViewDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );

    shaderResourceViewDesc.Format              = resourceDesc.Format;
    shaderResourceViewDesc.ViewDimension       = D3D11_SRV_DIMENSION_TEXTURE3D;
    shaderResourceViewDesc.Texture3D.MipLevels = -1;

    MOJO_D3D_SAFE( d3d11Device->CreateTexture3D( &resourceDesc, NULL, &mD3D11Resource ) );
    MOJO_D3D_SAFE( d3d11Device->CreateShaderResourceView( mD3D11Resource, &shaderResourceViewDesc, &mD3D11ShaderResourceView ) );
    MOJO_CUDA_SAFE( cudaGraphicsD3D11RegisterResource( &mCudaGraphicsResource, mD3D11Resource, cudaGraphicsRegisterFlagsNone ) );

    mD3D11DeviceContext->UpdateSubresource(
        mD3D11Resource,
        0,
        NULL,
        volumeDescription.data,
        volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel,
        volumeDescription.numVoxels.y * volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel );
}

template < typename TD3D11ResourceType, typename TCudaType >
D3D11CudaTexture< TD3D11ResourceType, TCudaType >::D3D11CudaTexture( ID3D11Device*                       d3d11Device,
                                                                     ID3D11DeviceContext*                d3d11DeviceContext,
                                                                     D3D11_TEXTURE3D_DESC                resourceDesc,
                                                                     int3                                numVoxels,
                                                                     thrust::device_vector< TCudaType >& deviceVector ) :
    mCudaGraphicsResource   ( NULL ),
    mD3D11Resource          ( NULL ),
    mD3D11ShaderResourceView( NULL ),
    mD3D11DeviceContext     ( d3d11DeviceContext )
{
    D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
    ZeroMemory( &shaderResourceViewDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );

    shaderResourceViewDesc.Format              = resourceDesc.Format;
    shaderResourceViewDesc.ViewDimension       = D3D11_SRV_DIMENSION_TEXTURE3D;
    shaderResourceViewDesc.Texture3D.MipLevels = -1;

    MOJO_D3D_SAFE( d3d11Device->CreateTexture3D( &resourceDesc, NULL, &mD3D11Resource ) );
    MOJO_D3D_SAFE( d3d11Device->CreateShaderResourceView( mD3D11Resource, &shaderResourceViewDesc, &mD3D11ShaderResourceView ) );
    MOJO_CUDA_SAFE( cudaGraphicsD3D11RegisterResource( &mCudaGraphicsResource, mD3D11Resource, cudaGraphicsRegisterFlagsNone ) );

    MapCudaArray();
    cudaArray* cudaArray = GetMappedCudaArray();
    Thrust::Memcpy3DToArray( cudaArray, deviceVector, numVoxels );
    UnmapCudaArray();
}

//
// general-purpose methods
//
template < typename TD3D11ResourceType, typename TCudaType >
D3D11CudaTexture< TD3D11ResourceType, TCudaType >::~D3D11CudaTexture()
{
    MOJO_CUDA_SAFE( cudaGraphicsUnregisterResource( mCudaGraphicsResource ) );

    mD3D11ShaderResourceView->Release();
    mD3D11ShaderResourceView = NULL;

    mD3D11Resource->Release();
    mD3D11Resource = NULL;
}

template < typename TD3D11ResourceType, typename TCudaType >
void D3D11CudaTexture< TD3D11ResourceType, TCudaType >::Update( VolumeDescription volumeDescription )
{
    mD3D11DeviceContext->UpdateSubresource(
        mD3D11Resource,
        0,
        NULL,
        volumeDescription.data,
        volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel,
        volumeDescription.numVoxels.y * volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel );
}

template < typename TD3D11ResourceType, typename TCudaType >
void D3D11CudaTexture< TD3D11ResourceType, TCudaType >::MapCudaArray()
{
    MOJO_CUDA_SAFE( cudaGraphicsMapResources( 1, &mCudaGraphicsResource, 0 ) );
}


template < typename TD3D11ResourceType, typename TCudaType >
void D3D11CudaTexture< TD3D11ResourceType, TCudaType >::UnmapCudaArray()
{
    MOJO_CUDA_SAFE( cudaGraphicsUnmapResources( 1, &mCudaGraphicsResource, 0 ) );
}


template < typename TD3D11ResourceType, typename TCudaType >
cudaArray* D3D11CudaTexture< TD3D11ResourceType, TCudaType >::GetMappedCudaArray( int mipLevel )
{
    cudaArray* mappedArray = NULL;
    MOJO_CUDA_SAFE( cudaGraphicsSubResourceGetMappedArray( &mappedArray, mCudaGraphicsResource, 0, mipLevel ) );
    return mappedArray;
}


template < typename TD3D11ResourceType, typename TCudaType >
cudaGraphicsResource* D3D11CudaTexture< TD3D11ResourceType, TCudaType >::GetCudaGraphicsResource()
{
    return mCudaGraphicsResource;
}

template < typename TD3D11ResourceType, typename TCudaType >
ID3D11ShaderResourceView* D3D11CudaTexture< TD3D11ResourceType, TCudaType >::GetD3D11ShaderResourceView()
{
    return mD3D11ShaderResourceView;
}

}
}