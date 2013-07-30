#pragma once

#include "VolumeDescription.hpp"
#include "Assert.hpp"

struct ID3D11ShaderResourceView;

namespace Mojo
{
namespace Native
{

class ID3D11Texture
{
public:
    virtual                           ~ID3D11Texture() {};

    virtual void                      Update( VolumeDescription volumeDescription ) = 0;
    virtual ID3D11ShaderResourceView* GetD3D11ShaderResourceView()                  = 0;
};

}
}