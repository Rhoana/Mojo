#include "TiledDatasetView.hpp"

namespace Mojo
{
namespace Native
{

TiledDatasetView::TiledDatasetView() :
centerDataSpace( make_float3( -1.0f, -1.0f, -1.0f ) ),
extentDataSpace( make_float3( -1.0f, -1.0f, -1.0f ) ),
widthNumPixels ( -1 )
{
}

}
}
