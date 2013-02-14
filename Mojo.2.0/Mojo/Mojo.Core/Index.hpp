#pragma once

//#include "Cuda.hpp"
#include "MojoVectors.hpp"

namespace Mojo
{
namespace Core
{
    //int Index3DToIndex1D( int3 index3D, int3 numVoxels );
    int Index3DToIndex1D( MojoInt3 index3D, MojoInt3 numVoxels );
}
}