#pragma once

#include "Mojo.Core/VolumeDescription.hpp"

#using <SlimDX.dll>

using namespace System;
using namespace SlimDX;
using namespace SlimDX::DXGI;

namespace Mojo
{
namespace Interop
{

#pragma managed
public ref class VolumeDescription
{
public:
    VolumeDescription();
    VolumeDescription( const Core::VolumeDescription& volumeDescription );

    Core::VolumeDescription ToCore();

    property DataStream^ DataStream;
    property IntPtr      Data;

    property int         NumVoxelsX;
    property int         NumVoxelsY;
    property int         NumVoxelsZ;

    property Format      DxgiFormat;
    property int         NumBytesPerVoxel;
    property bool        IsSigned;
};

}
}
