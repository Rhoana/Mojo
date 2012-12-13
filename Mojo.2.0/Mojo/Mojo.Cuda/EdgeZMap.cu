#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/Thrust.hpp"
#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/SegmenterState.hpp"
#include "Mojo.Core/ID3D11CudaTexture.hpp"

#include "Index.cuh"

template < typename TCudaType >
struct InitializeEdgeZMapFunction
{
    enum TupleLayout
    {
        INDEX_3D,
        EDGE_Z_MAP
    };

    TCudaType* mSourceMapBuffer;
    int3       mNumVoxels;
    float      mMinSourceValue;
    float      mMaxSourceValue;
    float      mEdgeMultiplier;
    float      mEdgePower;
    float      mEdgeMaxBeforeSaturate;

    InitializeEdgeZMapFunction( TCudaType* sourceMapBuffer,
                                int3       numVoxels,
                                float      minSourceValue,
                                float      maxSourceValue,
                                float      edgeMultiplier,
                                float      edgePower,
                                float      edgeMaxBeforeSaturate ) :
        mSourceMapBuffer      ( sourceMapBuffer ),
        mNumVoxels            ( numVoxels ),
        mMinSourceValue       ( minSourceValue ),
        mMaxSourceValue       ( maxSourceValue ),
        mEdgeMultiplier       ( edgeMultiplier ),
        mEdgePower            ( edgePower ),
        mEdgeMaxBeforeSaturate( edgeMaxBeforeSaturate )
    {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int3   index3D                  = thrust::get< INDEX_3D >( tuple );
        int3   index3DForwardZ          = make_int3( index3D.x, index3D.y, min( mNumVoxels.z - 1, index3D.z + 1 ) );
                                              
        float  source                   = ( __uint2float_rd( mSourceMapBuffer[ Index3DToIndex1D( index3D,         mNumVoxels ) ].x ) - mMinSourceValue ) / mMaxSourceValue;
        float  sourceForwardZ           = ( __uint2float_rd( mSourceMapBuffer[ Index3DToIndex1D( index3DForwardZ, mNumVoxels ) ].x ) - mMinSourceValue ) / mMaxSourceValue;

        float  sourceGradientZ          = sourceForwardZ - source;
        float  sourceGradientZMagnitude = abs( sourceGradientZ );
        float  edgeZ                    = pow( 2.71f, -1.0f * mEdgeMultiplier * pow( sourceGradientZMagnitude, mEdgePower ) );

        if ( edgeZ > mEdgeMaxBeforeSaturate )
        {
            edgeZ = 1.0f;
        }

        thrust::get< EDGE_Z_MAP >( tuple ) = edgeZ;
    }
};

template < typename TCudaType >
void InitializeEdgeZMapInternal( Mojo::Core::SegmenterState* segmenterState, Mojo::Core::HashMap< std::string, Mojo::Core::VolumeDescription >* volumeDescriptions, float minSourceValue, float maxSourceValue )
{
    int numElements = volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels.x * volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels.y * volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels.z;

    MOJO_CUDA_SAFE( cudaMemcpy( thrust::raw_pointer_cast< float >( &segmenterState->deviceVectors.Get< float >( "ScratchpadMap" )[ 0 ] ),
                                volumeDescriptions->Get( "FilteredSourceMap" ).data,
                                numElements * sizeof( TCudaType ),
                                cudaMemcpyHostToDevice ) );

    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( 0 ),
                        Index1DToIndex3DFunction( volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "EdgeZMap" ).begin() ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( numElements ),
                        Index1DToIndex3DFunction( volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "EdgeZMap" ).end() ) ),

            InitializeEdgeZMapFunction< TCudaType >(
                (TCudaType*)thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float >( "ScratchpadMap" )[ 0 ] ),
                volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels,
                minSourceValue,
                maxSourceValue,
                segmenterState->constParameters.Get< float >( "EDGE_MULTIPLIER" ),
                segmenterState->constParameters.Get< float >( "EDGE_POWER_Z" ),
                segmenterState->constParameters.Get< float >( "EDGE_MAX_BEFORE_SATURATE" ) ) ) );
}

extern "C" void InitializeEdgeZMap( Mojo::Core::SegmenterState* segmenterState, Mojo::Core::HashMap< std::string, Mojo::Core::VolumeDescription >* volumeDescriptions )
{
    switch ( segmenterState->volumeDescription.dxgiFormat )
    {
        case DXGI_FORMAT_R8_UNORM:
            InitializeEdgeZMapInternal< uchar1 >( segmenterState, volumeDescriptions, 0.0f, 255.0f );
            break;

        case DXGI_FORMAT_R8G8B8A8_UNORM:
            InitializeEdgeZMapInternal< uchar4 >( segmenterState, volumeDescriptions, 0.0f, 255.0f );
            break;

        default:
            RELEASE_ASSERT( 0 );
    }
}
