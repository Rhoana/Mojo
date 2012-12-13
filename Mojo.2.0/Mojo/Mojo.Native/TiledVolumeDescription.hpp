#pragma once

#include "Mojo.Core/D3D11.hpp"
#include "Mojo.Core/Cuda.hpp"

namespace Mojo
{
namespace Native
{

class TiledVolumeDescription
{
public:
    TiledVolumeDescription();

    std::string imageDataDirectory;
    std::string fileExtension;

    int4        numTiles;
    int3        numVoxelsPerTile;
    int3        numVoxels;

    DXGI_FORMAT dxgiFormat;
    int         numBytesPerVoxel;
    bool        isSigned;
};

}
}