#include "D3D11CudaTextureMap.hpp"

#include "ForEach.hpp"
#include "ID3D11CudaTexture.hpp"

namespace Mojo
{
namespace Core
{

ID3D11CudaTexture* D3D11CudaTextureMap::Get( std::string key )
{
    RELEASE_ASSERT( mD3D11CudaTextures.find( key ) != mD3D11CudaTextures.end() );
    return mD3D11CudaTextures[ key ];
};

void D3D11CudaTextureMap::Set( std::string key, ID3D11CudaTexture* value )
{
    mD3D11CudaTextures[ key ] = value;
};

stdext::hash_map< std::string, ID3D11CudaTexture* >& D3D11CudaTextureMap::GetHashMap()
{
    return mD3D11CudaTextures;
};

void D3D11CudaTextureMap::MapCudaArrays()
{
    RELEASE_ASSERT( mCudaGraphicsResources.empty() );

    MOJO_FOR_EACH_KEY_VALUE( std::string key, ID3D11CudaTexture* d3d11CudaTexture, mD3D11CudaTextures )
    {
        mCudaGraphicsResources.push_back( d3d11CudaTexture->GetCudaGraphicsResource() );
    }

    MOJO_CUDA_SAFE( cudaGraphicsMapResources( mCudaGraphicsResources.size(), &mCudaGraphicsResources[ 0 ], 0 ) );
}

void D3D11CudaTextureMap::UnmapCudaArrays()
{
    MOJO_CUDA_SAFE( cudaGraphicsUnmapResources( mCudaGraphicsResources.size(), &mCudaGraphicsResources[ 0 ], 0 ) );
    mCudaGraphicsResources.clear();
}

}
}