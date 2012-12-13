#include "Cuda.hpp"

namespace Mojo
{
namespace Core
{
namespace Cuda
{

void Initialize( ID3D11Device* d3d11Device )
{
    MOJO_CUDA_SAFE( cudaD3D11SetDirect3DDevice( d3d11Device ) );
    MOJO_CUDA_SAFE( cudaFree( NULL ) );
}

void Terminate()
{
    MOJO_CUDA_SAFE( cudaThreadExit() );
}

void MemcpyHostToArray3D( cudaArray* cudaArray, VolumeDescription volumeDescription )
{
    cudaMemcpy3DParms copyParams = {0};

    cudaPitchedPtr pitchedPointer;

    pitchedPointer.ptr   = volumeDescription.data;
    pitchedPointer.pitch = volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel;
    pitchedPointer.xsize = volumeDescription.numVoxels.x;
    pitchedPointer.ysize = volumeDescription.numVoxels.y;

    cudaExtent volumeExtent = make_cudaExtent( volumeDescription.numVoxels.x, volumeDescription.numVoxels.y, volumeDescription.numVoxels.z );

    copyParams.srcPtr   = pitchedPointer;
    copyParams.dstArray = cudaArray;
    copyParams.extent   = volumeExtent;
    copyParams.kind     = cudaMemcpyHostToDevice;

    MOJO_CUDA_SAFE( cudaMemcpy3D( &copyParams ) );

    Synchronize();
}

void BindTextureReferenceToArray( textureReference* textureReference, cudaArray* cudaArray )
{
    cudaChannelFormatDesc channelDesc;
    MOJO_CUDA_SAFE( cudaGetChannelDesc( &channelDesc, cudaArray ) );
    MOJO_CUDA_SAFE( cudaBindTextureToArray( textureReference, cudaArray, &channelDesc ) );
}

void Synchronize()
{
    cudaThreadSynchronize();
    MOJO_CUDA_SAFE( cudaGetLastError() );
}

}
}
}
