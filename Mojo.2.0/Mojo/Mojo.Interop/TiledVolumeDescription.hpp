#pragma once

#include "Mojo.Native/TiledVolumeDescription.hpp"

#using <SlimDX.dll>

using namespace System;
using namespace SlimDX;
using namespace SlimDX::DXGI;

namespace Mojo
{
namespace Interop
{

#pragma managed
public ref class TiledVolumeDescription
{
public:
    TiledVolumeDescription();
    TiledVolumeDescription( Native::TiledVolumeDescription tiledVolumeDescription );

    Native::TiledVolumeDescription ToNative();

    property String^ FileExtension;

    property int     NumTilesX;
    property int     NumTilesY;
    property int     NumTilesZ;
    property int     NumTilesW;

    property int     NumVoxelsPerTileX;
    property int     NumVoxelsPerTileY;
    property int     NumVoxelsPerTileZ;

    property int     NumVoxelsX;
    property int     NumVoxelsY;
    property int     NumVoxelsZ;

    property Format  DxgiFormat;
    property int     NumBytesPerVoxel;
    property bool    IsSigned;
};

}
}
