#include "VolumeDescription.hpp"

namespace Mojo
{
namespace Interop
{

VolumeDescription::VolumeDescription()
{
    Data             = IntPtr( nullptr );

    NumVoxelsX       = -1;
    NumVoxelsY       = -1;
    NumVoxelsZ       = -1;

    DxgiFormat       = SlimDX::DXGI::Format::Unknown;
    NumBytesPerVoxel = -1;
    IsSigned         = false;
}

VolumeDescription::VolumeDescription( const Core::VolumeDescription& volumeDescription )
{
    Data             = IntPtr( volumeDescription.data );

    NumVoxelsX       = volumeDescription.numVoxels.x;
    NumVoxelsY       = volumeDescription.numVoxels.y;
    NumVoxelsZ       = volumeDescription.numVoxels.z;

    DxgiFormat       = (SlimDX::DXGI::Format)volumeDescription.dxgiFormat;
    NumBytesPerVoxel = volumeDescription.numBytesPerVoxel;
    IsSigned         = volumeDescription.isSigned;
}

Core::VolumeDescription VolumeDescription::ToCore()
{
    Core::VolumeDescription volumeDescription;

    volumeDescription.data             = Data.ToPointer();

    volumeDescription.numVoxels.x      = NumVoxelsX;
    volumeDescription.numVoxels.y      = NumVoxelsY;
    volumeDescription.numVoxels.z      = NumVoxelsZ;
    
    volumeDescription.dxgiFormat       = (DXGI_FORMAT)DxgiFormat;
    volumeDescription.numBytesPerVoxel = NumBytesPerVoxel;
    volumeDescription.isSigned         = IsSigned;

    return volumeDescription;
}

}
}