#include "Index.hpp"

namespace Mojo
{
namespace Native
{

int Index3DToIndex1D( Int3 index3D, Int3 numVoxels )
{
    return ( numVoxels.x * numVoxels.y * index3D.z ) + ( numVoxels.x * index3D.y ) + index3D.x;
}

}
}