#pragma once

//#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/MojoVectors.hpp"

namespace Mojo
{
namespace Native
{

struct TiledDatasetView
{
    TiledDatasetView();

    Mojo::Core::MojoFloat3 centerDataSpace;
    Mojo::Core::MojoFloat3 extentDataSpace;
    int    widthNumPixels;
};

}
}