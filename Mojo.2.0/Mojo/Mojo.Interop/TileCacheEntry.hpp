#pragma once

#include "Mojo.Native/TileCacheEntry.hpp"

#include "ObservableDictionary.hpp"

#using <SlimDX.dll>

using namespace System;
using namespace SlimDX;

namespace Mojo
{
namespace Interop
{

#pragma managed
public ref class TileCacheEntry
{
public:
    TileCacheEntry();

    property ObservableDictionary< String^, SlimDX::Direct3D11::ShaderResourceView^ >^ D3D11Textures;
    property Vector4                                                                   IndexTileSpace;
    property Vector3                                                                   CenterDataSpace;
    property Vector3                                                                   ExtentDataSpace;
    property bool                                                                      Active;
};

}
}
