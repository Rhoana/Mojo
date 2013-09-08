#pragma once

#include "Mojo.Native/VolumeDescription.hpp"

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
    VolumeDescription( const Native::VolumeDescription& volumeDescription );

    Native::VolumeDescription ToCore();

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
