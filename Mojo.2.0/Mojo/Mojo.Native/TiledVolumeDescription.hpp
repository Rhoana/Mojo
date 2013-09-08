#pragma once

#include "D3D11.hpp"
#include "Types.hpp"

namespace Mojo
{
namespace Native
{

class TiledVolumeDescription
{
public:
    TiledVolumeDescription();

    std::string fileExtension;

    Int4        numTiles;
    Int3        numVoxelsPerTile;
    Int3        numVoxels;

    DXGI_FORMAT dxgiFormat;
    int         numBytesPerVoxel;
    bool        isSigned;
};

}
}