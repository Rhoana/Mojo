#include "TiledVolumeDescription.hpp"

namespace Mojo
{
namespace Native
{

TiledVolumeDescription::TiledVolumeDescription() :
    numTiles        ( make_int4( -1, -1, -1, -1 ) ),
    numVoxelsPerTile( make_int3( -1, -1, -1 ) ),
    numVoxels       ( make_int3( -1, -1, -1 ) ),
    dxgiFormat      ( DXGI_FORMAT_UNKNOWN ),
    numBytesPerVoxel( -1 ),
    isSigned        ( false )
{
}

}
}