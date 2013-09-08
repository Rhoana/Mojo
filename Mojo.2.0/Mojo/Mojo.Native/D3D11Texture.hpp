#pragma once

#include "Assert.hpp"
#include "D3D11.hpp"
#include "ID3D11Texture.hpp"

namespace Mojo
{
namespace Native
{

template < typename TD3D11ResourceType >
class D3D11Texture : public ID3D11Texture
{
public:
    //
    // 1D
    //
    D3D11Texture( ID3D11Device*        d3d11Device,
                  ID3D11DeviceContext* d3d11DeviceContext,
                  D3D11_TEXTURE1D_DESC resourceDesc );
                                               
    D3D11Texture( ID3D11Device*        d3d11Device,
                  ID3D11DeviceContext* d3d11DeviceContext,
                  D3D11_TEXTURE1D_DESC resourceDesc,
                  VolumeDescription    volumeDescription );
                                                          
    //
    // 2D
    //
    D3D11Texture( ID3D11Device*        d3d11Device,
                  ID3D11DeviceContext* d3d11DeviceContext,
                  D3D11_TEXTURE2D_DESC resourceDesc );
                                           
    D3D11Texture( ID3D11Device*        d3d11Device,
                  ID3D11DeviceContext* d3d11DeviceContext,
                  D3D11_TEXTURE2D_DESC resourceDesc,
                  VolumeDescription    volumeDescription );

    //
    // 3D
    //
    D3D11Texture( ID3D11Device*        d3d11Device,
                  ID3D11DeviceContext* d3d11DeviceContext,
                  D3D11_TEXTURE3D_DESC resourceDesc );
                                           
    D3D11Texture( ID3D11Device*        d3d11Device,
                  ID3D11DeviceContext* d3d11DeviceContext,
                  D3D11_TEXTURE3D_DESC resourceDesc,
                  VolumeDescription    volumeDescription );
                                                          
    //
    // general-purpose methods
    //
    virtual ~D3D11Texture();

    virtual void                      Update( VolumeDescription volumeDescription );
    virtual ID3D11ShaderResourceView* GetD3D11ShaderResourceView();

private:
    TD3D11ResourceType*       mD3D11Resource;
    ID3D11ShaderResourceView* mD3D11ShaderResourceView;
    ID3D11DeviceContext*      mD3D11DeviceContext;
};

//
// 1D
//
template < typename TD3D11ResourceType >
D3D11Texture< TD3D11ResourceType >::D3D11Texture( ID3D11Device*        d3d11Device,
                                                  ID3D11DeviceContext* d3d11DeviceContext,
                                                  D3D11_TEXTURE1D_DESC resourceDesc ) :
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
}

template < typename TD3D11ResourceType >
D3D11Texture< TD3D11ResourceType >::D3D11Texture( ID3D11Device*        d3d11Device,
                                                  ID3D11DeviceContext* d3d11DeviceContext,
                                                  D3D11_TEXTURE1D_DESC resourceDesc,
                                                  VolumeDescription    volumeDescription ) :
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

    mD3D11DeviceContext->UpdateSubresource(
        mD3D11Resource,
        0,
        NULL,
        volumeDescription.data,
        volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel,
        volumeDescription.numVoxels.y * volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel );
}

//
// 2D
//
template < typename TD3D11ResourceType >
D3D11Texture< TD3D11ResourceType >::D3D11Texture( ID3D11Device*        d3d11Device,
                                                  ID3D11DeviceContext* d3d11DeviceContext,
                                                  D3D11_TEXTURE2D_DESC resourceDesc ) :
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
}

template < typename TD3D11ResourceType >
D3D11Texture< TD3D11ResourceType >::D3D11Texture( ID3D11Device*        d3d11Device,
                                                  ID3D11DeviceContext* d3d11DeviceContext,
                                                  D3D11_TEXTURE2D_DESC resourceDesc,
                                                  VolumeDescription    volumeDescription ) :
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

    mD3D11DeviceContext->UpdateSubresource(
        mD3D11Resource,
        0,
        NULL,
        volumeDescription.data,
        volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel,
        volumeDescription.numVoxels.y * volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel );
}

//
// 3D
//
template < typename TD3D11ResourceType >
D3D11Texture< TD3D11ResourceType >::D3D11Texture( ID3D11Device*        d3d11Device,
                                                  ID3D11DeviceContext* d3d11DeviceContext,
                                                  D3D11_TEXTURE3D_DESC resourceDesc ) :
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
}

template < typename TD3D11ResourceType >
D3D11Texture< TD3D11ResourceType >::D3D11Texture( ID3D11Device*        d3d11Device,
                                                  ID3D11DeviceContext* d3d11DeviceContext,
                                                  D3D11_TEXTURE3D_DESC resourceDesc,
                                                  VolumeDescription    volumeDescription ) :
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

    mD3D11DeviceContext->UpdateSubresource(
        mD3D11Resource,
        0,
        NULL,
        volumeDescription.data,
        volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel,
        volumeDescription.numVoxels.y * volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel );
}

//
// general-purpose methods
//
template < typename TD3D11ResourceType >
D3D11Texture< TD3D11ResourceType >::~D3D11Texture()
{
    mD3D11ShaderResourceView->Release();
    mD3D11ShaderResourceView = NULL;

    mD3D11Resource->Release();
    mD3D11Resource = NULL;
}

template < typename TD3D11ResourceType >
void D3D11Texture< TD3D11ResourceType >::Update( VolumeDescription volumeDescription )
{
    mD3D11DeviceContext->UpdateSubresource(
        mD3D11Resource,
        0,
        NULL,
        volumeDescription.data,
        volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel,
        volumeDescription.numVoxels.y * volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel );
}

template < typename TD3D11ResourceType >
ID3D11ShaderResourceView* D3D11Texture< TD3D11ResourceType >::GetD3D11ShaderResourceView()
{
    return mD3D11ShaderResourceView;
}

}
}