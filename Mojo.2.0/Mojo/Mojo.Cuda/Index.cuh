#pragma once

#ifdef __CUDACC__

    #include "Mojo.Core/Thrust.hpp"

    struct Index1DToIndex2DFunction : public thrust::unary_function< int, int2 >
    {
        int   mNumPixelsX;
        float mInvNumPixelsX;

        Index1DToIndex2DFunction( int  numPixelsX ) : mNumPixelsX( numPixelsX ),  mInvNumPixelsX( 1.0f / (float)numPixelsX ) {}
        Index1DToIndex2DFunction( int3 numPixels )  : mNumPixelsX( numPixels.x ), mInvNumPixelsX( 1.0f / (float)numPixels.x ) {}

        __device__
        int2 operator() ( int index1D )
        {
            int y = __float2uint_rd( index1D * mInvNumPixelsX );
            int x = index1D - ( y * mNumPixelsX );

            int2 index2D = make_int2( x, y );

            return index2D;
        }
    };

    struct Index1DToIndex3DFunction : public thrust::unary_function< int, int3 >
    {
        int mNumVoxelsX, mNumVoxelsXY;
        float mInvNumVoxelsX, mInvNumVoxelsXY;

        Index1DToIndex3DFunction( int3 numVoxels ) :
            mNumVoxelsX    ( numVoxels.x ),
            mNumVoxelsXY   ( numVoxels.x * numVoxels.y ),
            mInvNumVoxelsX ( 1.0f / (float)mNumVoxelsX ),
            mInvNumVoxelsXY( 1.0f / (float)mNumVoxelsXY )
        {}

        __device__
        int3 operator() ( int index1D )
        {
            int z        = __float2uint_rd( index1D * mInvNumVoxelsXY );
            index1D      = index1D - ( z * mNumVoxelsXY );
            int y        = __float2uint_rd( index1D * mInvNumVoxelsX );
            int x        = index1D - ( y * mNumVoxelsX );

            int3 index3D = make_int3( x, y, z );

            return index3D;
        }
    };

    inline __device__ int2 IndexBlock2DThread2DToIndex2D()
    {
        return make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
    }

    inline __device__ int IndexBlock1DThread1DToIndex1D()
    {
        return ( blockIdx.x * blockDim.x ) + threadIdx.x;
    }

#endif

inline __host__ __device__ int Index2DToIndex1D( int2 index2D, int2 numPixels )
{
    return ( numPixels.x * index2D.y ) + index2D.x;
}

inline __host__ __device__ int Index3DToIndex1D( int3 index3D, int3 numVoxels )
{
    return ( numVoxels.x * numVoxels.y * index3D.z ) + ( numVoxels.x * index3D.y ) + index3D.x;
}
