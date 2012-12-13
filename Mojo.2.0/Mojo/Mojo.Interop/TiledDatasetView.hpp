#pragma once

#include "Mojo.Native/TiledDatasetView.hpp"

#using <SlimDX.dll>

using namespace SlimDX;

namespace Mojo
{
namespace Interop
{

#pragma managed
public ref class TiledDatasetView
{
public:
    TiledDatasetView();

    Native::TiledDatasetView ToNative();

    property Vector3 CenterDataSpace;
    property Vector3 ExtentDataSpace;
    property int      WidthNumPixels;
};

}
}
