#include "VolumeDescription.hpp"

namespace Mojo
{
namespace Native
{

VolumeDescription::VolumeDescription() :
    data            ( 0 ),
    numVoxels       ( Int3( -1, -1, -1 ) ),
    dxgiFormat      ( DXGI_FORMAT_UNKNOWN ),
    numBytesPerVoxel( -1 ),
    isSigned        ( false )
{
}

}
}