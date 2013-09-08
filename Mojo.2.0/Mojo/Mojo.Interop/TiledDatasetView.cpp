 #include "TiledDatasetView.hpp"

#include "Mojo.Native/Types.hpp"
#include "Mojo.Native/TiledDatasetView.hpp"

namespace Mojo
{
namespace Interop
{

TiledDatasetView::TiledDatasetView()
{
    CenterDataSpace   = Vector3( -1.0f, -1.0f, -1.0f );
    ExtentDataSpace   = Vector3( -1.0f, -1.0f, -1.0f );
    WidthNumPixels    = -1;
}

Native::TiledDatasetView TiledDatasetView::ToNative()
{
    Native::TiledDatasetView tiledDatasetView;

    tiledDatasetView.centerDataSpace = Native::Float3( CenterDataSpace.X, CenterDataSpace.Y, CenterDataSpace.Z );
    tiledDatasetView.extentDataSpace = Native::Float3( ExtentDataSpace.X, ExtentDataSpace.Y, ExtentDataSpace.Z );
    tiledDatasetView.widthNumPixels  = WidthNumPixels;

    return tiledDatasetView;
}

}
}