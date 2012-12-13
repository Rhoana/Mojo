#include "Index.hpp"

namespace Mojo
{
namespace Core
{

int Index3DToIndex1D( int3 index3D, int3 numVoxels )
{
    return ( numVoxels.x * numVoxels.y * index3D.z ) + ( numVoxels.x * index3D.y ) + index3D.x;
}

}
}