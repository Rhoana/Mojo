#pragma once

#include "Mojo.Core/Cuda.hpp"

namespace Mojo
{
namespace Native
{

struct TiledDatasetView
{
    TiledDatasetView();

    float3 centerDataSpace;
    float3 extentDataSpace;
    int    widthNumPixels;
};

}
}