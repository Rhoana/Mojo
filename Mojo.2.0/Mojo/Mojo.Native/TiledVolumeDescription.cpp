#include "TiledVolumeDescription.hpp"

namespace Mojo
{
namespace Native
{

TiledVolumeDescription::TiledVolumeDescription() :
    numTiles        ( -1, -1, -1, -1 ),
    numVoxelsPerTile( -1, -1, -1 ),
    numVoxels       ( -1, -1, -1 ),
    dxgiFormat      ( DXGI_FORMAT_UNKNOWN ),
    numBytesPerVoxel( -1 ),
    isSigned        ( false )
{
}

}
}