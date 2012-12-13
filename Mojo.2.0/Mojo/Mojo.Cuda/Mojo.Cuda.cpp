#include "Mojo.Core/Assert.hpp"

namespace Mojo
{
namespace Core
{
namespace Cuda
{

extern "C" void Dummy()
{
    RELEASE_ASSERT( 0 );
}

}
}
}