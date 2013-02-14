#pragma once

#include "VolumeDescription.hpp"
#include "Assert.hpp"

//#include "Cuda.hpp"

struct ID3D11ShaderResourceView;

namespace Mojo
{
namespace Core
{

class ID3D11CudaTexture
{
public:
    virtual                           ~ID3D11CudaTexture() {};

    virtual void                      Update( VolumeDescription volumeDescription ) = 0;

    //virtual void                      MapCudaArray()                                = 0;
    //virtual void                      UnmapCudaArray()                              = 0;
    //virtual cudaArray*                GetMappedCudaArray( int mipLevel = 0 )        = 0;

    //virtual cudaGraphicsResource*     GetCudaGraphicsResource()                     = 0;
    virtual ID3D11ShaderResourceView* GetD3D11ShaderResourceView()                  = 0;
};

}
}