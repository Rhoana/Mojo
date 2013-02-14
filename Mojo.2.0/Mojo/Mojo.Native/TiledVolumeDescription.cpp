#include "TiledVolumeDescription.hpp"

namespace Mojo
{
namespace Native
{

TiledVolumeDescription::TiledVolumeDescription() :
    numTilesX        ( -1 ),
    numTilesY        ( -1 ),
    numTilesZ        ( -1 ),
    numTilesW        ( -1 ),
    numVoxelsPerTileX( -1 ),
    numVoxelsPerTileY( -1 ),
    numVoxelsPerTileZ( -1 ),
    numVoxelsX       ( -1 ),
    numVoxelsY       ( -1 ),
    numVoxelsZ       ( -1 ),
    dxgiFormat      ( DXGI_FORMAT_UNKNOWN ),
    numBytesPerVoxel( -1 ),
    isSigned        ( false )
{
}

Mojo::Core::MojoInt4 TiledVolumeDescription::numTiles()
{
    return Mojo::Core::MojoInt4( numTilesX, numTilesY, numTilesZ, numTilesW );
}

Mojo::Core::MojoInt3 TiledVolumeDescription::numVoxelsPerTile()
{
    return Mojo::Core::MojoInt3( numVoxelsPerTileX, numVoxelsPerTileY, numVoxelsPerTileZ );
}

Mojo::Core::MojoInt3 TiledVolumeDescription::numVoxels()
{
    return Mojo::Core::MojoInt3( numVoxelsX, numVoxelsY, numVoxelsZ );
}

}
}