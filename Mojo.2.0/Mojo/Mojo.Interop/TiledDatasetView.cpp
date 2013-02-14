 #include "TiledDatasetView.hpp"

#include "Mojo.Core/MojoVectors.hpp"
#include "Mojo.Native/TiledDatasetView.hpp"

namespace Mojo
{
namespace Interop
{

TiledDatasetView::TiledDatasetView()
{
    CenterDataSpace   = Vector3( 0.0f, 0.0f, 0.0f );
    ExtentDataSpace   = Vector3( 1.0f, 1.0f, 0.0f );
    WidthNumPixels    = 0;
}

Native::TiledDatasetView TiledDatasetView::ToNative()
{
    Native::TiledDatasetView tiledDatasetView;

    tiledDatasetView.centerDataSpace = Core::MojoFloat3( CenterDataSpace.X, CenterDataSpace.Y, CenterDataSpace.Z );
    tiledDatasetView.extentDataSpace = Core::MojoFloat3( ExtentDataSpace.X, ExtentDataSpace.Y, ExtentDataSpace.Z );
    tiledDatasetView.widthNumPixels  = WidthNumPixels;

    return tiledDatasetView;
}

}
}