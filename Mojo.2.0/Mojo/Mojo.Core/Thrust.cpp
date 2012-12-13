#include "Thrust.hpp"

#include "Assert.hpp"

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{

bool   use_preallocated_scratchpad           = false;
size_t preallocated_scratchpad_size_in_bytes = 0;
void*  preallocated_scratchpad_pointer       = NULL;

}
}
}
}

namespace Mojo
{
namespace Core
{
namespace Thrust
{

int FAILURE = 0;

void Initialize()
{
    try
    {
        thrust::device_vector< int > dummyDeviceVector = thrust::device_vector< int >( 32 );
    }
    catch( thrust::system_error e )
    {
        Mojo::Core::Printf( e.what() );
        Mojo::Core::Printf( "Mojo requires an NVIDIA GTX 480 or newer graphics card to run." );
        RELEASE_ASSERT( Thrust::FAILURE );
    }
    catch( std::bad_alloc e )
    {
        Mojo::Core::Printf( e.what() );
        Mojo::Core::Printf( "Mojo requires an NVIDIA GTX 480 or newer graphics card to run." );
        RELEASE_ASSERT( Thrust::FAILURE );
    }
}

void Terminate()
{
}

}
}
}