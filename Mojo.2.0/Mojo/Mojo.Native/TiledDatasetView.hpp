#pragma once

#include "Types.hpp"

namespace Mojo
{
namespace Native
{

struct TiledDatasetView
{
    TiledDatasetView();

    Float3 centerDataSpace;
    Float3 extentDataSpace;
    int    widthNumPixels;
};

}
}