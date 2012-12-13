#pragma once

#include "Stl.hpp"

#pragma warning( push )
#pragma warning( disable : 4995 )
#pragma warning( disable : 4996 )
#define generic __identifier(generic)

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#undef generic
#pragma warning( pop )

#include "VolumeDescription.hpp" 
#include "Printf.hpp"

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{
extern  bool   gMojoThrustUsePreallocatedScratchpad;
extern  size_t gMojoThrustPreallocatedPoolSizeInBytes;
extern  void*  gMojoThrustPreallocatedPoolPointer;
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
    void Initialize();
    void Terminate();

    template < typename TCudaType > void MemcpyHostToDevice( thrust::device_vector< TCudaType >& destination, void*                               source, int numElements );
    template < typename TCudaType > void MemcpyDeviceToHost( void*                               destination, thrust::device_vector< TCudaType >& source, int numElements );

    template < typename TCudaType > void MemcpyHostToDevice( thrust::device_reference< TCudaType >& destination, void*                                  source, int numElements );
    template < typename TCudaType > void MemcpyDeviceToHost( void*                                  destination, thrust::device_reference< TCudaType >& source, int numElements );

    template < typename TCudaType > void Memcpy2DToArray( cudaArray* cudaArray, thrust::device_vector< TCudaType >& deviceVector, int3 numVoxels );
    template < typename TCudaType > void Memcpy3DToArray( cudaArray* cudaArray, thrust::device_vector< TCudaType >& deviceVector, int3 numVoxels );

    template < typename TCudaType > void Memcpy2DToArray( cudaArray* cudaArray, thrust::device_reference< TCudaType >& deviceReference, int3 numVoxels );
    template < typename TCudaType > void Memcpy3DToArray( cudaArray* cudaArray, thrust::device_reference< TCudaType >& deviceReference, int3 numVoxels );

    template< typename InputIterator, typename TCudaType >                           TCudaType Reduce( InputIterator first, InputIterator last, TCudaType init, thrust::device_vector< TCudaType >& scratchpad );
    template< typename InputIterator, typename TCudaType, typename BinaryFunction >  TCudaType Reduce( InputIterator first, InputIterator last, TCudaType init, BinaryFunction binary_op, thrust::device_vector< TCudaType >& scratchpad );

    extern int FAILURE;
}
}
}


#ifdef _DEBUG
    #define MOJO_THRUST_SAFE_SYNCHRONIZE_DEBUG( functionCall ) \
        do                                                     \
        {                                                      \
            try                                                \
            {                                                  \
                functionCall ;                                 \
                MOJO_CUDA_SAFE( cudaThreadSynchronize() );     \
            }                                                  \
            catch( thrust::system_error e )                    \
            {                                                  \
                Mojo::Core::Printf( e.what() );                \
                ASSERT( Mojo::Core::Thrust::FAILURE );         \
            }                                                  \
            catch( std::bad_alloc e )                          \
            {                                                  \
                Mojo::Core::Printf( e.what() );                \
                ASSERT( Mojo::Core::Thrust::FAILURE );         \
            }                                                  \
        } while ( 0 )
#else
    #define MOJO_THRUST_SAFE_SYNCHRONIZE_DEBUG( functionCall ) \
        do                                                     \
        {                                                      \
            functionCall ;                                     \
            MOJO_CUDA_SAFE( cudaThreadSynchronize() );         \
        } while ( 0 )
#endif


#define MOJO_THRUST_SAFE_SYNCHRONIZE_RELEASE( functionCall ) \
    do                                                       \
    {                                                        \
        try                                                  \
        {                                                    \
            functionCall ;                                   \
            MOJO_CUDA_SAFE( cudaThreadSynchronize() );       \
        }                                                    \
        catch( thrust::system_error e )                      \
        {                                                    \
            Mojo::Core::Printf( e.what() );                  \
            RELEASE_ASSERT( Mojo::Core::Thrust::FAILURE );   \
        }                                                    \
        catch( std::bad_alloc e )                            \
        {                                                    \
            Mojo::Core::Printf( e.what() );                  \
            RELEASE_ASSERT( Mojo::Core::Thrust::FAILURE );   \
        }                                                    \
    } while ( 0 )


#ifdef _DEBUG
    #define MOJO_THRUST_SAFE_NO_SYNCHRONIZE_DEBUG( functionCall ) \
        do                                                        \
        {                                                         \
            try                                                   \
            {                                                     \
                functionCall ;                                    \
            }                                                     \
            catch ( thrust::system_error e )                      \
            {                                                     \
                Mojo::Core::Printf( e.what() );                   \
                ASSERT( Mojo::Core::Thrust::FAILURE );            \
            }                                                     \
            catch( std::bad_alloc e )                             \
            {                                                     \
                Mojo::Core::Printf( e.what() );                   \
                ASSERT( Mojo::Core::Thrust::FAILURE );            \
            }                                                     \
        }                                                         \
        while ( 0 )
#else
    #define MOJO_THRUST_SAFE_NO_SYNCHRONIZE_DEBUG( functionCall ) functionCall
#endif

#define MOJO_THRUST_SAFE_NO_SYNCHRONIZE_RELEASE( functionCall ) \
    do                                                          \
    {                                                           \
        try                                                     \
        {                                                       \
            functionCall ;                                      \
        }                                                       \
        catch( thrust::system_error e )                         \
        {                                                       \
            Mojo::Core::Printf( e.what() );                     \
            RELEASE_ASSERT( Mojo::Core::Thrust::FAILURE );      \
        }                                                       \
        catch( std::bad_alloc e )                               \
        {                                                       \
            Mojo::Core::Printf( e.what() );                     \
            ASSERT( Mojo::Core::Thrust::FAILURE );              \
        }                                                       \
    } while ( 0 )

#define MOJO_THRUST_SAFE( functionCall ) MOJO_THRUST_SAFE_SYNCHRONIZE_RELEASE( functionCall )
//#define MOJO_THRUST_SAFE( functionCall ) Mojo::Core::Printf( #functionCall ); MOJO_THRUST_SAFE_SYNCHRONIZE_RELEASE( functionCall )

namespace Mojo
{
namespace Core
{
namespace Thrust
{

template < typename TCudaType >
inline void MemcpyHostToDevice( thrust::device_vector< TCudaType >& destination, void* source, int numElements )
{
    MOJO_CUDA_SAFE( cudaMemcpy( thrust::raw_pointer_cast< TCudaType >( &destination[ 0 ] ),
                                source,
                                numElements * sizeof( TCudaType ),
                                cudaMemcpyHostToDevice ) );
}

template < typename TCudaType >
inline void MemcpyDeviceToHost( void* destination, thrust::device_vector< TCudaType >& source, int numElements )
{
    MOJO_CUDA_SAFE( cudaMemcpy( destination,
                                thrust::raw_pointer_cast< TCudaType >( &source[ 0 ] ),
                                numElements * sizeof( TCudaType ),
                                cudaMemcpyDeviceToHost ) );
}

template < typename TCudaType >
inline void MemcpyHostToDevice( thrust::device_reference< TCudaType >& destination, void* source, int numElements )
{
    MOJO_CUDA_SAFE( cudaMemcpy( thrust::raw_pointer_cast< TCudaType >( &destination ),
                                source,
                                numElements * sizeof( TCudaType ),
                                cudaMemcpyHostToDevice ) );
}

template < typename TCudaType >
inline void MemcpyDeviceToHost( void* destination, thrust::device_reference< TCudaType >& source, int numElements )
{
    MOJO_CUDA_SAFE( cudaMemcpy( destination,
                                thrust::raw_pointer_cast< TCudaType >( &source ),
                                numElements * sizeof( TCudaType ),
                                cudaMemcpyDeviceToHost ) );
}

template < typename TCudaType >
inline  void Memcpy2DToArray( cudaArray* cudaArray, thrust::device_vector< TCudaType >& deviceVector, int3 numVoxels )
{
    TCudaType* devicePointer = thrust::raw_pointer_cast( &deviceVector[ 0 ] );
    ASSERT( devicePointer != NULL );

    MOJO_CUDA_SAFE( cudaMemcpy2DToArray( cudaArray,
                                         0,
                                         0,
                                         devicePointer,
                                         numVoxels.x * sizeof( TCudaType ),
                                         numVoxels.x * sizeof( TCudaType ),
                                         numVoxels.y,
                                         cudaMemcpyDeviceToDevice ) );
}

template < typename TCudaType >
inline  void Memcpy3DToArray( cudaArray* cudaArray, thrust::device_vector< TCudaType >& deviceVector, int3 numVoxels )
{
    TCudaType* devicePointer = thrust::raw_pointer_cast( &deviceVector[ 0 ] );

    cudaMemcpy3DParms copyParams = {0};
    cudaPitchedPtr pitchedPointer;

    pitchedPointer.ptr   = devicePointer;
    pitchedPointer.pitch = numVoxels.x * sizeof( TCudaType );
    pitchedPointer.xsize = numVoxels.x;
    pitchedPointer.ysize = numVoxels.y;

    cudaExtent volumeExtent = make_cudaExtent( numVoxels.x, numVoxels.y, numVoxels.z );

    copyParams.srcPtr   = pitchedPointer;
    copyParams.dstArray = cudaArray;
    copyParams.extent   = volumeExtent;
    copyParams.kind     = cudaMemcpyDeviceToDevice;

    MOJO_CUDA_SAFE( cudaMemcpy3D( &copyParams ) );
}

template < typename TCudaType >
inline  void Memcpy2DToArray( cudaArray* cudaArray, thrust::device_reference< TCudaType >& deviceReference, int3 numVoxels )
{
    TCudaType* devicePointer = thrust::raw_pointer_cast( &deviceReference );
    ASSERT( devicePointer != NULL );

    MOJO_CUDA_SAFE( cudaMemcpy2DToArray( cudaArray,
                                         0,
                                         0,
                                         devicePointer,
                                         numVoxels.x * sizeof( TCudaType ),
                                         numVoxels.x * sizeof( TCudaType ),
                                         numVoxels.y,
                                         cudaMemcpyDeviceToDevice ) );
}

template < typename TCudaType >
inline  void Memcpy3DToArray( cudaArray* cudaArray, thrust::device_reference< TCudaType >& deviceReference, int3 numVoxels )
{
    TCudaType* devicePointer = thrust::raw_pointer_cast( &deviceReference );

    cudaMemcpy3DParms copyParams = {0};
    cudaPitchedPtr pitchedPointer;

    pitchedPointer.ptr   = devicePointer;
    pitchedPointer.pitch = numVoxels.x * sizeof( TCudaType );
    pitchedPointer.xsize = numVoxels.x;
    pitchedPointer.ysize = numVoxels.y;

    cudaExtent volumeExtent = make_cudaExtent( numVoxels.x, numVoxels.y, numVoxels.z );

    copyParams.srcPtr   = pitchedPointer;
    copyParams.dstArray = cudaArray;
    copyParams.extent   = volumeExtent;
    copyParams.kind     = cudaMemcpyDeviceToDevice;

    MOJO_CUDA_SAFE( cudaMemcpy3D( &copyParams ) );
}

template< typename InputIterator, typename TCudaType >
inline  TCudaType Reduce( InputIterator first, InputIterator last, TCudaType init, thrust::device_vector< TCudaType >& scratchpad )
{
    return Reduce( first, last, init, thrust::plus< TCudaType >(), scratchpad );
}

template< typename InputIterator, typename TCudaType, typename BinaryFunction >
inline  TCudaType Reduce( InputIterator first, InputIterator last, TCudaType init, BinaryFunction binary_op, thrust::device_vector< TCudaType >& scratchpad )
{
    thrust::detail::device::cuda::use_preallocated_scratchpad           = true;
    thrust::detail::device::cuda::preallocated_scratchpad_size_in_bytes = scratchpad.size() * sizeof( TCudaType );
    thrust::detail::device::cuda::preallocated_scratchpad_pointer       = (void*)thrust::raw_pointer_cast( &scratchpad[ 0 ] );

    TCudaType returnValue = thrust::reduce( first, last, init, binary_op );

    thrust::detail::device::cuda::use_preallocated_scratchpad           = false;
    thrust::detail::device::cuda::preallocated_scratchpad_size_in_bytes = 0;
    thrust::detail::device::cuda::preallocated_scratchpad_pointer       = NULL;

    return returnValue;
}

}
}
}