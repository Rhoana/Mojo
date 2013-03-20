#include "VolumeDescription.hpp"

namespace Mojo
{
namespace Core
{

VolumeDescription::VolumeDescription() :
    data            ( 0 ),
    numVoxels       ( MojoInt3( -1, -1, -1 ) ),
    dxgiFormat      ( DXGI_FORMAT_UNKNOWN ),
    numBytesPerVoxel( -1 ),
    isSigned        ( false )
{
}

}
}