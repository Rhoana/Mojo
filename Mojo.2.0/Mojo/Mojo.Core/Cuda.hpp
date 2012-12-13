#pragma once

#define WIN32_LEAN_AND_MEAN

#include "Stl.hpp"

#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>

#ifndef __CUDACC__
    #include <cuda_d3d11_interop.h>
    #undef max
    #undef min
#endif

#include <cutil_math.h>

#include "Assert.hpp"
#include "Printf.hpp"
#include "VolumeDescription.hpp"

struct ID3D11Device;
struct cudaArray;

namespace Mojo
{
namespace Core
{
namespace Cuda
{
    void                                       Initialize( ID3D11Device* d3d11Device );
    void                                       Terminate();

    template < typename TCudaType > cudaArray* MallocArray2D( VolumeDescription volumeDescription );
    template < typename TCudaType > cudaArray* MallocArray3D( VolumeDescription volumeDescription );

    void                                       MemcpyHostToArray3D( cudaArray* cudaArray, VolumeDescription volumeDescription );

    void                                       BindTextureReferenceToArray( textureReference* textureReference, cudaArray* cudaArray );
    void                                       Synchronize();
}
}
}

#define MOJO_CUDA_CHECK_RESULT_DEBUG( cudaResult )                      \
    do                                                                  \
    {                                                                   \
        const char* resultString = cudaGetErrorString( cudaResult );    \
        if ( cudaResult != cudaSuccess )                                \
        {                                                               \
            Mojo::Core::Printf( "\nCuda Error: ", resultString, "\n" ); \
        }                                                               \
        ASSERT( cudaResult != cudaErrorMissingConfiguration       );    \
        ASSERT( cudaResult != cudaErrorMemoryAllocation           );    \
        ASSERT( cudaResult != cudaErrorInitializationError        );    \
        ASSERT( cudaResult != cudaErrorLaunchFailure              );    \
        ASSERT( cudaResult != cudaErrorPriorLaunchFailure         );    \
        ASSERT( cudaResult != cudaErrorLaunchTimeout              );    \
        ASSERT( cudaResult != cudaErrorLaunchOutOfResources       );    \
        ASSERT( cudaResult != cudaErrorInvalidDeviceFunction      );    \
        ASSERT( cudaResult != cudaErrorInvalidConfiguration       );    \
        ASSERT( cudaResult != cudaErrorInvalidDevice              );    \
        ASSERT( cudaResult != cudaErrorInvalidValue               );    \
        ASSERT( cudaResult != cudaErrorInvalidPitchValue          );    \
        ASSERT( cudaResult != cudaErrorInvalidSymbol              );    \
        ASSERT( cudaResult != cudaErrorMapBufferObjectFailed      );    \
        ASSERT( cudaResult != cudaErrorUnmapBufferObjectFailed    );    \
        ASSERT( cudaResult != cudaErrorInvalidHostPointer         );    \
        ASSERT( cudaResult != cudaErrorInvalidDevicePointer       );    \
        ASSERT( cudaResult != cudaErrorInvalidTexture             );    \
        ASSERT( cudaResult != cudaErrorInvalidTextureBinding      );    \
        ASSERT( cudaResult != cudaErrorInvalidChannelDescriptor   );    \
        ASSERT( cudaResult != cudaErrorInvalidMemcpyDirection     );    \
        ASSERT( cudaResult != cudaErrorAddressOfConstant          );    \
        ASSERT( cudaResult != cudaErrorTextureFetchFailed         );    \
        ASSERT( cudaResult != cudaErrorTextureNotBound            );    \
        ASSERT( cudaResult != cudaErrorSynchronizationError       );    \
        ASSERT( cudaResult != cudaErrorInvalidFilterSetting       );    \
        ASSERT( cudaResult != cudaErrorInvalidNormSetting         );    \
        ASSERT( cudaResult != cudaErrorMixedDeviceExecution       );    \
        ASSERT( cudaResult != cudaErrorCudartUnloading            );    \
        ASSERT( cudaResult != cudaErrorUnknown                    );    \
        ASSERT( cudaResult != cudaErrorNotYetImplemented          );    \
        ASSERT( cudaResult != cudaErrorMemoryValueTooLarge        );    \
        ASSERT( cudaResult != cudaErrorInvalidResourceHandle      );    \
        ASSERT( cudaResult != cudaErrorNotReady                   );    \
        ASSERT( cudaResult != cudaErrorInsufficientDriver         );    \
        ASSERT( cudaResult != cudaErrorSetOnActiveProcess         );    \
        ASSERT( cudaResult != cudaErrorInvalidSurface             );    \
        ASSERT( cudaResult != cudaErrorNoDevice                   );    \
        ASSERT( cudaResult != cudaErrorECCUncorrectable           );    \
        ASSERT( cudaResult != cudaErrorSharedObjectSymbolNotFound );    \
        ASSERT( cudaResult != cudaErrorSharedObjectInitFailed     );    \
        ASSERT( cudaResult != cudaErrorUnsupportedLimit           );    \
        ASSERT( cudaResult != cudaErrorDuplicateVariableName      );    \
        ASSERT( cudaResult != cudaErrorDuplicateTextureName       );    \
        ASSERT( cudaResult != cudaErrorDuplicateSurfaceName       );    \
        ASSERT( cudaResult != cudaErrorDevicesUnavailable         );    \
        ASSERT( cudaResult != cudaErrorInvalidKernelImage         );    \
        ASSERT( cudaResult != cudaErrorNoKernelImageForDevice     );    \
        ASSERT( cudaResult != cudaErrorIncompatibleDriverContext  );    \
        ASSERT( cudaResult != cudaErrorStartupFailure             );    \
        ASSERT( cudaResult == cudaSuccess                         );    \
    } while ( 0 )


#define MOJO_CUDA_CHECK_RESULT_RELEASE( cudaResult )                         \
    do                                                                       \
    {                                                                        \
        const char* resultString = cudaGetErrorString( cudaResult );         \
        if ( cudaResult != cudaSuccess )                                     \
        {                                                                    \
            Mojo::Core::Printf( "\nCuda Error: ", resultString, "\n" );      \
        }                                                                    \
        RELEASE_ASSERT( cudaResult != cudaErrorMissingConfiguration       ); \
        RELEASE_ASSERT( cudaResult != cudaErrorMemoryAllocation           ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInitializationError        ); \
        RELEASE_ASSERT( cudaResult != cudaErrorLaunchFailure              ); \
        RELEASE_ASSERT( cudaResult != cudaErrorPriorLaunchFailure         ); \
        RELEASE_ASSERT( cudaResult != cudaErrorLaunchTimeout              ); \
        RELEASE_ASSERT( cudaResult != cudaErrorLaunchOutOfResources       ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidDeviceFunction      ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidConfiguration       ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidDevice              ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidValue               ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidPitchValue          ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidSymbol              ); \
        RELEASE_ASSERT( cudaResult != cudaErrorMapBufferObjectFailed      ); \
        RELEASE_ASSERT( cudaResult != cudaErrorUnmapBufferObjectFailed    ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidHostPointer         ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidDevicePointer       ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidTexture             ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidTextureBinding      ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidChannelDescriptor   ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidMemcpyDirection     ); \
        RELEASE_ASSERT( cudaResult != cudaErrorAddressOfConstant          ); \
        RELEASE_ASSERT( cudaResult != cudaErrorTextureFetchFailed         ); \
        RELEASE_ASSERT( cudaResult != cudaErrorTextureNotBound            ); \
        RELEASE_ASSERT( cudaResult != cudaErrorSynchronizationError       ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidFilterSetting       ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidNormSetting         ); \
        RELEASE_ASSERT( cudaResult != cudaErrorMixedDeviceExecution       ); \
        RELEASE_ASSERT( cudaResult != cudaErrorCudartUnloading            ); \
        RELEASE_ASSERT( cudaResult != cudaErrorUnknown                    ); \
        RELEASE_ASSERT( cudaResult != cudaErrorNotYetImplemented          ); \
        RELEASE_ASSERT( cudaResult != cudaErrorMemoryValueTooLarge        ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidResourceHandle      ); \
        RELEASE_ASSERT( cudaResult != cudaErrorNotReady                   ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInsufficientDriver         ); \
        RELEASE_ASSERT( cudaResult != cudaErrorSetOnActiveProcess         ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidSurface             ); \
        RELEASE_ASSERT( cudaResult != cudaErrorNoDevice                   ); \
        RELEASE_ASSERT( cudaResult != cudaErrorECCUncorrectable           ); \
        RELEASE_ASSERT( cudaResult != cudaErrorSharedObjectSymbolNotFound ); \
        RELEASE_ASSERT( cudaResult != cudaErrorSharedObjectInitFailed     ); \
        RELEASE_ASSERT( cudaResult != cudaErrorUnsupportedLimit           ); \
        RELEASE_ASSERT( cudaResult != cudaErrorDuplicateVariableName      ); \
        RELEASE_ASSERT( cudaResult != cudaErrorDuplicateTextureName       ); \
        RELEASE_ASSERT( cudaResult != cudaErrorDuplicateSurfaceName       ); \
        RELEASE_ASSERT( cudaResult != cudaErrorDevicesUnavailable         ); \
        RELEASE_ASSERT( cudaResult != cudaErrorInvalidKernelImage         ); \
        RELEASE_ASSERT( cudaResult != cudaErrorNoKernelImageForDevice     ); \
        RELEASE_ASSERT( cudaResult != cudaErrorIncompatibleDriverContext  ); \
        RELEASE_ASSERT( cudaResult != cudaErrorStartupFailure             ); \
        RELEASE_ASSERT( cudaResult == cudaSuccess                         ); \
    } while ( 0 )


#define MOJO_CUDA_CHECK_RESULT( cudaResult ) MOJO_CUDA_CHECK_RESULT_DEBUG( cudaResult )


#ifdef _DEBUG
    #define MOJO_CUDA_SAFE_DEBUG( functionCall )          \
        do                                                \
        {                                                 \
            cudaError_t cudaResult = functionCall ;       \
            MOJO_CUDA_CHECK_RESULT_DEBUG( cudaResult );   \
        } while ( 0 )
#else
    #define MOJO_CUDA_SAFE_DEBUG( functionCall ) functionCall
#endif


#define MOJO_CUDA_SAFE_RELEASE( functionCall )        \
    do                                                \
    {                                                 \
        cudaError_t cudaResult = functionCall ;       \
        MOJO_CUDA_CHECK_RESULT_RELEASE( cudaResult ); \
    } while ( 0 )


#define MOJO_CUDA_SAFE( functionCall ) MOJO_CUDA_SAFE_RELEASE( functionCall )
//#define MOJO_CUDA_SAFE( functionCall ) Mojo::Core::Printf( #functionCall ); MOJO_CUDA_SAFE_RELEASE( functionCall )

#define MOJO_CUDA_BENCHMARK_KERNEL_BEGIN()                           \
    do                                                               \
    {                                                                \
        cudaEvent_t start, stop;                                     \
        cudaEventCreate( &start );                                   \
        cudaEventCreate( &stop );                                    \
        cudaEventRecord( start, 0 );


#define MOJO_CUDA_BENCHMARK_KERNEL_END( timerVariableRef )           \
        cudaEventRecord( stop, 0 );                                  \
        cudaEventSynchronize( stop );                                \
        cudaEventElapsedTime( timerVariableRef , start, stop );      \
        cudaEventDestroy( start );                                   \
        cudaEventDestroy( stop );                                    \
    } while ( 0 )


namespace Mojo
{
namespace Core
{
namespace Cuda
{
template <>
inline cudaArray* MallocArray2D< float >( VolumeDescription volumeDescription )
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat ); 

    cudaArray* cuArray; 
    MOJO_CUDA_SAFE( cudaMallocArray( &cuArray, &channelDesc, volumeDescription.numVoxels.x, volumeDescription.numVoxels.y ) ); 

    return cuArray;
}

template <>
inline cudaArray* MallocArray2D< float2 >( VolumeDescription volumeDescription )
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 32, 32, 0, 0, cudaChannelFormatKindFloat ); 

    cudaArray* cuArray; 
    MOJO_CUDA_SAFE( cudaMallocArray( &cuArray, &channelDesc, volumeDescription.numVoxels.x, volumeDescription.numVoxels.y ) ); 

    return cuArray;
}

template <>
inline cudaArray* MallocArray3D< uchar1 >( VolumeDescription volumeDescription )
{
    cudaChannelFormatDesc channelDesc  = cudaCreateChannelDesc( 8, 0, 0, 0, cudaChannelFormatKindUnsigned ); 
    cudaExtent            volumeExtent = make_cudaExtent( volumeDescription.numVoxels.x, volumeDescription.numVoxels.y, volumeDescription.numVoxels.z );

    cudaArray* cuArray; 
    MOJO_CUDA_SAFE( cudaMalloc3DArray( &cuArray, &channelDesc, volumeExtent ) ); 

    return cuArray;
}

template <>
inline cudaArray* MallocArray3D< uchar4 >( VolumeDescription volumeDescription )
{
    cudaChannelFormatDesc channelDesc  = cudaCreateChannelDesc( 8, 8, 8, 8, cudaChannelFormatKindUnsigned ); 
    cudaExtent            volumeExtent = make_cudaExtent( volumeDescription.numVoxels.x, volumeDescription.numVoxels.y, volumeDescription.numVoxels.z );

    cudaArray* cuArray; 
    MOJO_CUDA_SAFE( cudaMalloc3DArray( &cuArray, &channelDesc, volumeExtent ) ); 

    return cuArray;
}

}
}
}
