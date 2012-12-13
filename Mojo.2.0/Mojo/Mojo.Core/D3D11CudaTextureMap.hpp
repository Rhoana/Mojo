#pragma once

#include "Stl.hpp"

struct cudaGraphicsResource;

namespace Mojo
{
namespace Core
{

class ID3D11CudaTexture;

class D3D11CudaTextureMap
{
public:
    ID3D11CudaTexture*                                   Get( std::string key );
    void                                                 Set( std::string key, ID3D11CudaTexture* value );

    stdext::hash_map< std::string, ID3D11CudaTexture* >& GetHashMap();

    void                                                 MapCudaArrays();
    void                                                 UnmapCudaArrays();

private:
    stdext::hash_map< std::string, ID3D11CudaTexture* > mD3D11CudaTextures;
    std::vector< cudaGraphicsResource* >                mCudaGraphicsResources;
};

}
}