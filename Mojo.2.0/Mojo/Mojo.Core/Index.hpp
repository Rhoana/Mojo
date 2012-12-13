#pragma once

#include "Cuda.hpp"

namespace Mojo
{
namespace Core
{
    int Index3DToIndex1D( int3 index3D, int3 numVoxels );
}
}