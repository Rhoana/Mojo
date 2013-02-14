#include "TiledDatasetView.hpp"

namespace Mojo
{
namespace Native
{

TiledDatasetView::TiledDatasetView() :
centerDataSpace( Mojo::Core::MojoFloat3( -1.0f, -1.0f, -1.0f ) ),
extentDataSpace( Mojo::Core::MojoFloat3( -1.0f, -1.0f, -1.0f ) ),
widthNumPixels ( -1 )
{
}

}
}
