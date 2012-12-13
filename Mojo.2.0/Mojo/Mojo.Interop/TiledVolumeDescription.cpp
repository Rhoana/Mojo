#include "TiledVolumeDescription.hpp"

#include <msclr/marshal_cppstd.h>

namespace Mojo
{
namespace Interop
{

TiledVolumeDescription::TiledVolumeDescription()
{
    NumTilesX         = -1;
    NumTilesY         = -1;
    NumTilesZ         = -1;
    NumTilesW         = -1;

    NumVoxelsPerTileX = -1;
    NumVoxelsPerTileY = -1;
    NumVoxelsPerTileZ = -1;

    NumVoxelsX        = -1;
    NumVoxelsY        = -1;
    NumVoxelsZ        = -1;

    DxgiFormat        = SlimDX::DXGI::Format::Unknown;
    NumBytesPerVoxel  = -1;
    IsSigned          = false;
}

TiledVolumeDescription::TiledVolumeDescription( Native::TiledVolumeDescription tiledVolumeDescription )
{
    ImageDataDirectory = msclr::interop::marshal_as< String^ >( tiledVolumeDescription.imageDataDirectory );
    FileExtension      = msclr::interop::marshal_as< String^ >( tiledVolumeDescription.fileExtension );

    NumTilesX         = tiledVolumeDescription.numTiles.x;
    NumTilesY         = tiledVolumeDescription.numTiles.y;
    NumTilesZ         = tiledVolumeDescription.numTiles.z;
    NumTilesW         = tiledVolumeDescription.numTiles.w;

    NumVoxelsPerTileX = tiledVolumeDescription.numVoxelsPerTile.x;
    NumVoxelsPerTileY = tiledVolumeDescription.numVoxelsPerTile.y;
    NumVoxelsPerTileZ = tiledVolumeDescription.numVoxelsPerTile.z;

    NumVoxelsX        = tiledVolumeDescription.numVoxels.x;
    NumVoxelsY        = tiledVolumeDescription.numVoxels.y;
    NumVoxelsZ        = tiledVolumeDescription.numVoxels.z;

    DxgiFormat        = (SlimDX::DXGI::Format)tiledVolumeDescription.dxgiFormat;
    NumBytesPerVoxel  = tiledVolumeDescription.numBytesPerVoxel;
    IsSigned          = tiledVolumeDescription.isSigned;
}

Native::TiledVolumeDescription TiledVolumeDescription::ToNative()
{
    Native::TiledVolumeDescription tiledVolumeDescription;

    tiledVolumeDescription.imageDataDirectory = msclr::interop::marshal_as< std::string >( ImageDataDirectory );
    tiledVolumeDescription.fileExtension      = msclr::interop::marshal_as< std::string >( FileExtension );

    tiledVolumeDescription.numTiles.x         = NumTilesX;
    tiledVolumeDescription.numTiles.y         = NumTilesY;
    tiledVolumeDescription.numTiles.z         = NumTilesZ;
    tiledVolumeDescription.numTiles.w         = NumTilesW;

    tiledVolumeDescription.numVoxelsPerTile.x = NumVoxelsPerTileX;
    tiledVolumeDescription.numVoxelsPerTile.y = NumVoxelsPerTileY;
    tiledVolumeDescription.numVoxelsPerTile.z = NumVoxelsPerTileZ;

    tiledVolumeDescription.numVoxels.x        = NumVoxelsX;
    tiledVolumeDescription.numVoxels.y        = NumVoxelsY;
    tiledVolumeDescription.numVoxels.z        = NumVoxelsZ;

    tiledVolumeDescription.dxgiFormat         = (DXGI_FORMAT)DxgiFormat;
    tiledVolumeDescription.numBytesPerVoxel   = NumBytesPerVoxel;
    tiledVolumeDescription.isSigned           = IsSigned;

    return tiledVolumeDescription;
}

}
}