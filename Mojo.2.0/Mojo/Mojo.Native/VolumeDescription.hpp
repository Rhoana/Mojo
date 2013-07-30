#pragma once

#include "Types.hpp"
#include <DXGIFormat.h>

namespace Mojo
{
namespace Native
{

class VolumeDescription
{
public:
    VolumeDescription();

    void*       data;

    Int3        numVoxels;

    DXGI_FORMAT dxgiFormat;
    int         numBytesPerVoxel;
    bool        isSigned;
};

}
}