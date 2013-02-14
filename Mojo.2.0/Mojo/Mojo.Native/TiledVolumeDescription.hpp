#pragma once

#include "Mojo.Core/D3D11.hpp"
//#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/MojoVectors.hpp"

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

    int        numTilesX;
    int        numTilesY;
    int        numTilesZ;
    int        numTilesW;

    int        numVoxelsPerTileX;
    int        numVoxelsPerTileY;
    int        numVoxelsPerTileZ;

    int        numVoxelsX;
    int        numVoxelsY;
    int        numVoxelsZ;

    DXGI_FORMAT dxgiFormat;
    int         numBytesPerVoxel;
    bool        isSigned;

    Mojo::Core::MojoInt4 TiledVolumeDescription::numTiles();
    Mojo::Core::MojoInt3 TiledVolumeDescription::numVoxelsPerTile();
    Mojo::Core::MojoInt3 TiledVolumeDescription::numVoxels();

};

}
}