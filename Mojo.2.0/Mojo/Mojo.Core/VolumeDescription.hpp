#pragma once

#include "MojoVectors.hpp"
//#include <cuda_runtime.h>
#include <DXGIFormat.h>

namespace Mojo
{
namespace Core
{

class VolumeDescription
{
public:
    VolumeDescription();

    void*       data;

    MojoInt3    numVoxels;

    DXGI_FORMAT dxgiFormat;
    int         numBytesPerVoxel;
    bool        isSigned;
};

}
}