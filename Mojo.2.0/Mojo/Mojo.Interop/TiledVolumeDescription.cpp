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

    NumTilesX         = tiledVolumeDescription.numTilesX;
    NumTilesY         = tiledVolumeDescription.numTilesY;
    NumTilesZ         = tiledVolumeDescription.numTilesZ;
    NumTilesW         = tiledVolumeDescription.numTilesW;

    NumVoxelsPerTileX = tiledVolumeDescription.numVoxelsPerTileX;
    NumVoxelsPerTileY = tiledVolumeDescription.numVoxelsPerTileY;
    NumVoxelsPerTileZ = tiledVolumeDescription.numVoxelsPerTileZ;

    NumVoxelsX        = tiledVolumeDescription.numVoxelsX;
    NumVoxelsY        = tiledVolumeDescription.numVoxelsY;
    NumVoxelsZ        = tiledVolumeDescription.numVoxelsZ;

    DxgiFormat        = (SlimDX::DXGI::Format)tiledVolumeDescription.dxgiFormat;
    NumBytesPerVoxel  = tiledVolumeDescription.numBytesPerVoxel;
    IsSigned          = tiledVolumeDescription.isSigned;
}

Native::TiledVolumeDescription TiledVolumeDescription::ToNative()
{
    Native::TiledVolumeDescription tiledVolumeDescription;

    tiledVolumeDescription.imageDataDirectory = msclr::interop::marshal_as< std::string >( ImageDataDirectory );
    tiledVolumeDescription.fileExtension      = msclr::interop::marshal_as< std::string >( FileExtension );

    tiledVolumeDescription.numTilesX         = NumTilesX;
    tiledVolumeDescription.numTilesY         = NumTilesY;
    tiledVolumeDescription.numTilesZ         = NumTilesZ;
    tiledVolumeDescription.numTilesW         = NumTilesW;

    tiledVolumeDescription.numVoxelsPerTileX = NumVoxelsPerTileX;
    tiledVolumeDescription.numVoxelsPerTileY = NumVoxelsPerTileY;
    tiledVolumeDescription.numVoxelsPerTileZ = NumVoxelsPerTileZ;

    tiledVolumeDescription.numVoxelsX        = NumVoxelsX;
    tiledVolumeDescription.numVoxelsY        = NumVoxelsY;
    tiledVolumeDescription.numVoxelsZ        = NumVoxelsZ;

    tiledVolumeDescription.dxgiFormat         = (DXGI_FORMAT)DxgiFormat;
    tiledVolumeDescription.numBytesPerVoxel   = NumBytesPerVoxel;
    tiledVolumeDescription.isSigned           = IsSigned;

    return tiledVolumeDescription;
}

}
}