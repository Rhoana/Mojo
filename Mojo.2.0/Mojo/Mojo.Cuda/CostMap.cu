#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/Thrust.hpp"
#include "Mojo.Core/SegmenterState.hpp"
#include "Mojo.Core/ID3D11CudaTexture.hpp"
#include "Mojo.Core/ForEach.hpp"
#include "Mojo.Core/Printf.hpp"

#include "Index.cuh"
#include "Math.cuh"

texture< bool,   3, cudaReadModeNormalizedFloat > TEXTURE_SOURCE_MAP_BAD_FORMAT;
texture< uchar1, 3, cudaReadModeNormalizedFloat > TEXTURE_SOURCE_MAP_UCHAR1_3D_NORMALIZED_FLOAT;
texture< uchar4, 3, cudaReadModeNormalizedFloat > TEXTURE_SOURCE_MAP_UCHAR4_3D_NORMALIZED_FLOAT;
texture< float,  2, cudaReadModeElementType >     TEXTURE_TEMP_SCRATCHPAD_MAP_FLOAT_2D_ELEMENT_TYPE;

template < typename TCudaType > __host__ __device__ static texture< TCudaType, 3, cudaReadModeNormalizedFloat >& GetSourceMapTexture() { return TEXTURE_SOURCE_MAP_BAD_FORMAT; }
template <>                     __host__ __device__ static texture< uchar1,    3, cudaReadModeNormalizedFloat >& GetSourceMapTexture() { return TEXTURE_SOURCE_MAP_UCHAR1_3D_NORMALIZED_FLOAT; }
template <>                     __host__ __device__ static texture< uchar4,    3, cudaReadModeNormalizedFloat >& GetSourceMapTexture() { return TEXTURE_SOURCE_MAP_UCHAR4_3D_NORMALIZED_FLOAT; }
void DummyOptimalPath() { GetSourceMapTexture< uchar1 >(); GetSourceMapTexture< uchar4 >(); }


struct ErodeScratchpadMapFunction
{
    enum TupleLayout
    {
        INDEX_2D,
        SCRATCHPAD_MAP,
    };

    ErodeScratchpadMapFunction() {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int2  index2D                          = thrust::get< INDEX_2D >( tuple );
                                               
        float imageValue                       = tex2D( TEXTURE_TEMP_SCRATCHPAD_MAP_FLOAT_2D_ELEMENT_TYPE, index2D.x,        index2D.y        );
        float imageXForward                    = tex2D( TEXTURE_TEMP_SCRATCHPAD_MAP_FLOAT_2D_ELEMENT_TYPE, index2D.x + 1.0f, index2D.y        );
        float imageYForward                    = tex2D( TEXTURE_TEMP_SCRATCHPAD_MAP_FLOAT_2D_ELEMENT_TYPE, index2D.x       , index2D.y + 1.0f );
        float imageXBackward                   = tex2D( TEXTURE_TEMP_SCRATCHPAD_MAP_FLOAT_2D_ELEMENT_TYPE, index2D.x - 1.0f, index2D.y        );
        float imageYBackward                   = tex2D( TEXTURE_TEMP_SCRATCHPAD_MAP_FLOAT_2D_ELEMENT_TYPE, index2D.x       , index2D.y - 1.0f );

        thrust::get< SCRATCHPAD_MAP >( tuple ) = min( min( min( min( imageValue, imageXForward ), imageYForward ), imageXBackward ), imageYBackward );
    }
};

struct InitializeCostMapOnSliceWithForegroundConstraintsFunction
{
    enum TupleLayout
    {
        PRIMAL_MAP,
        COST_MAP
    };

    float mPrimalMapThreshold;
    float mMaxCost;

    InitializeCostMapOnSliceWithForegroundConstraintsFunction( float primalMapThreshold, float maxCost ) : mPrimalMapThreshold( primalMapThreshold ), mMaxCost( maxCost ) {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        float primal = thrust::get< PRIMAL_MAP >( tuple );
        float newCost;

        if ( primal > mPrimalMapThreshold )
        {
            newCost = 0.0f;
        }
        else
        {
            newCost = mMaxCost;
        }

        thrust::get< COST_MAP >( tuple ) = newCost;
    }
};

template< typename TCudaType >
void InitializeCostMapFromPrimalMapInternal( Mojo::Core::SegmenterState* segmenterState )
{
    if ( segmenterState->slicesWithForegroundConstraints.size() > 1 )
    {
        Mojo::Core::Printf( "\nComputing cost:" );

        cudaArray* sourceMapArray          = segmenterState->cudaArrays.Get( "SourceMap" );
        cudaArray* tempScratchpadMapArray  = segmenterState->cudaArrays.Get( "TempScratchpadMap" );

        texture< TCudaType, 3, cudaReadModeNormalizedFloat >& sourceMapTextureReference = GetSourceMapTexture< TCudaType >();

        sourceMapTextureReference.normalized                                  = false;
        sourceMapTextureReference.filterMode                                  = cudaFilterModePoint;
        sourceMapTextureReference.addressMode[ 0 ]                            = cudaAddressModeClamp;
        sourceMapTextureReference.addressMode[ 1 ]                            = cudaAddressModeClamp;
        sourceMapTextureReference.addressMode[ 2 ]                            = cudaAddressModeClamp;

        TEXTURE_TEMP_SCRATCHPAD_MAP_FLOAT_2D_ELEMENT_TYPE.normalized          = false;
        TEXTURE_TEMP_SCRATCHPAD_MAP_FLOAT_2D_ELEMENT_TYPE.filterMode          = cudaFilterModePoint;
        TEXTURE_TEMP_SCRATCHPAD_MAP_FLOAT_2D_ELEMENT_TYPE.addressMode[ 0 ]    = cudaAddressModeClamp;
        TEXTURE_TEMP_SCRATCHPAD_MAP_FLOAT_2D_ELEMENT_TYPE.addressMode[ 1 ]    = cudaAddressModeClamp;
        TEXTURE_TEMP_SCRATCHPAD_MAP_FLOAT_2D_ELEMENT_TYPE.addressMode[ 2 ]    = cudaAddressModeClamp;

        Mojo::Core::Cuda::BindTextureReferenceToArray( &sourceMapTextureReference,                         sourceMapArray );
        Mojo::Core::Cuda::BindTextureReferenceToArray( &TEXTURE_TEMP_SCRATCHPAD_MAP_FLOAT_2D_ELEMENT_TYPE, tempScratchpadMapArray );

        int numElementsPerSlice               = segmenterState->volumeDescription.numVoxels.x * segmenterState->volumeDescription.numVoxels.y;
        int minSliceWithForegroundConstraints = *std::min_element( segmenterState->slicesWithForegroundConstraints.begin(), segmenterState->slicesWithForegroundConstraints.end() );
        int maxSliceWithForegroundConstraints = *std::max_element( segmenterState->slicesWithForegroundConstraints.begin(), segmenterState->slicesWithForegroundConstraints.end() );
        
        std::vector< int > slicesWithConstraints;
        slicesWithConstraints.resize( segmenterState->slicesWithForegroundConstraints.size() + segmenterState->slicesWithBackgroundConstraints.size() );
        std::set_union( segmenterState->slicesWithForegroundConstraints.begin(),
                        segmenterState->slicesWithForegroundConstraints.end(),
                        segmenterState->slicesWithBackgroundConstraints.begin(),
                        segmenterState->slicesWithBackgroundConstraints.end(),
                        slicesWithConstraints.begin() );

        Mojo::Core::Printf( "    Copying PrimalMap to ScratchpadMap..." );
        MOJO_THRUST_SAFE(
            thrust::copy(
                segmenterState->deviceVectors.Get< float >( "PrimalMap"     ).begin(),
                segmenterState->deviceVectors.Get< float >( "PrimalMap"     ).end(),
                segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin() ) );

        Mojo::Core::Printf( "    Eroding ScratchpadMap..." );
        MOJO_FOR_EACH( int sliceWithConstraints, slicesWithConstraints )
        {
            if ( sliceWithConstraints >= minSliceWithForegroundConstraints &&
                 sliceWithConstraints <= maxSliceWithForegroundConstraints &&
                 !segmenterState->constParameters.Get< bool >( "DIRECT_SCRIBBLE_PROPAGATION" ) )
            {
                for ( int i = 0; i < segmenterState->constParameters.Get< float >( "PRIMAL_MAP_ERODE_NUM_PASSES" ); i++ )
                {
                    Mojo::Core::Thrust::Memcpy2DToArray( tempScratchpadMapArray, segmenterState->deviceVectors.Get< float >( "ScratchpadMap" )[ numElementsPerSlice * sliceWithConstraints ], segmenterState->volumeDescription.numVoxels );

                    MOJO_THRUST_SAFE(
                        thrust::for_each(
                            thrust::make_zip_iterator(
                                thrust::make_tuple(
                                    thrust::make_transform_iterator(
                                        thrust::make_counting_iterator( 0 ),
                                        Index1DToIndex2DFunction( segmenterState->volumeDescription.numVoxels.x ) ),
                                    segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin() + ( numElementsPerSlice * sliceWithConstraints ) ) ),

                            thrust::make_zip_iterator(
                                thrust::make_tuple( 
                                    thrust::make_transform_iterator(
                                        thrust::make_counting_iterator( numElementsPerSlice ),
                                        Index1DToIndex2DFunction( segmenterState->volumeDescription.numVoxels.x ) ),
                                    segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin() + ( numElementsPerSlice * ( sliceWithConstraints + 1 ) ) ) ),
                               
                            ErodeScratchpadMapFunction() ) );
                }
            }
        }

        Mojo::Core::Printf( "    Initializing CostForwardMap and CostBackwardMap..." );
        MOJO_THRUST_SAFE(
            thrust::fill(
                segmenterState->deviceVectors.Get< float >( "CostForwardMap" ).begin(),
                segmenterState->deviceVectors.Get< float >( "CostForwardMap" ).end(),
                0.0f ) );

        MOJO_THRUST_SAFE(
            thrust::fill(
                segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ).begin(),
                segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ).end(),
                0.0f ) );

        MOJO_FOR_EACH( int sliceWithForegroundConstraints, segmenterState->slicesWithForegroundConstraints )
        {
            MOJO_THRUST_SAFE(
                thrust::for_each(
                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            segmenterState->deviceVectors.Get< float >( "ScratchpadMap"  ).begin() + ( numElementsPerSlice * sliceWithForegroundConstraints ),
                            segmenterState->deviceVectors.Get< float >( "CostForwardMap" ).begin() + ( numElementsPerSlice * sliceWithForegroundConstraints ) ) ),

                    thrust::make_zip_iterator(
                        thrust::make_tuple( 
                            segmenterState->deviceVectors.Get< float >( "ScratchpadMap"  ).begin() + ( numElementsPerSlice * ( sliceWithForegroundConstraints + 1 ) ),
                            segmenterState->deviceVectors.Get< float >( "CostForwardMap" ).begin() + ( numElementsPerSlice * ( sliceWithForegroundConstraints + 1 ) ) ) ),
                               
                    InitializeCostMapOnSliceWithForegroundConstraintsFunction(
                        segmenterState->constParameters.Get< float >( "PRIMAL_MAP_THRESHOLD" ),
                        segmenterState->constParameters.Get< float >( "COST_MAP_MAX_VALUE" ) ) ) );

            MOJO_THRUST_SAFE(
                thrust::for_each(
                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            segmenterState->deviceVectors.Get< float >( "ScratchpadMap"   ).begin() + ( numElementsPerSlice * sliceWithForegroundConstraints ),
                            segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ).begin() + ( numElementsPerSlice * sliceWithForegroundConstraints ) ) ),

                    thrust::make_zip_iterator(
                        thrust::make_tuple( 
                            segmenterState->deviceVectors.Get< float >( "ScratchpadMap"   ).begin() + ( numElementsPerSlice * ( sliceWithForegroundConstraints + 1 ) ),
                            segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ).begin() + ( numElementsPerSlice * ( sliceWithForegroundConstraints + 1 ) ) ) ),
                               
                    InitializeCostMapOnSliceWithForegroundConstraintsFunction(
                        segmenterState->constParameters.Get< float >( "PRIMAL_MAP_THRESHOLD" ),
                        segmenterState->constParameters.Get< float >( "COST_MAP_MAX_VALUE" ) ) ) );
        }
    }
}

extern "C" void InitializeCostMapFromPrimalMap( Mojo::Core::SegmenterState* segmenterState )
{
    switch ( segmenterState->volumeDescription.dxgiFormat )
    {
        case DXGI_FORMAT_R8_UNORM:
            InitializeCostMapFromPrimalMapInternal< uchar1 >( segmenterState );
            break;

        case DXGI_FORMAT_R8G8B8A8_UNORM:
            InitializeCostMapFromPrimalMapInternal< uchar4 >( segmenterState );
            break;

        default:
            RELEASE_ASSERT( 0 );
    }
}

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define WINDOW_SIZE_IN_BLOCKS_X 1
#define WINDOW_SIZE_IN_BLOCKS_Y 1
#define WINDOW_SIZE_IN_PIXELS_X BLOCK_SIZE_X * WINDOW_SIZE_IN_BLOCKS_X
#define WINDOW_SIZE_IN_PIXELS_Y BLOCK_SIZE_Y * WINDOW_SIZE_IN_BLOCKS_Y

template< typename TCudaType > __global__ void IncrementCostKernel(
    float*      d_inConstraintMap,
    float2*     d_inOpticalFlow,
    float*      d_inoutCost,
    int3        numVoxels,
    DXGI_FORMAT dxgiFormat,
    int         currentSlice,
    int         adjacentSliceDirection,
    float       opticalFlowImportanceFactor,
    float       maximumCost ) 
{
    __shared__ float  s_sourceAdjacentSlice                       [ ( ( 2 * WINDOW_SIZE_IN_BLOCKS_Y ) + 1 ) * BLOCK_SIZE_Y ][ ( ( ( 2 * WINDOW_SIZE_IN_BLOCKS_X ) + 1 ) * BLOCK_SIZE_X ) + 1 ]; 
    __shared__ float  s_costAdjacentSlice                         [ ( ( 2 * WINDOW_SIZE_IN_BLOCKS_Y ) + 1 ) * BLOCK_SIZE_Y ][ ( ( ( 2 * WINDOW_SIZE_IN_BLOCKS_X ) + 1 ) * BLOCK_SIZE_X ) + 1 ]; 
    __shared__ float2 s_opticalFlowFromAdjacentSliceToCurrentSlice[ ( ( 2 * WINDOW_SIZE_IN_BLOCKS_Y ) + 1 ) * BLOCK_SIZE_Y ][ ( ( ( 2 * WINDOW_SIZE_IN_BLOCKS_X ) + 1 ) * BLOCK_SIZE_X ) + 1 ]; 

    int2  inImageIndex2D      = IndexBlock2DThread2DToIndex2D();
    int3  inImageIndex3D      = make_int3( inImageIndex2D.x, inImageIndex2D.y, currentSlice );
    int   inImageIndex1D      = Index3DToIndex1D( inImageIndex3D, numVoxels );
    
    float constraint          = d_inConstraintMap[ inImageIndex1D ];
    float newCostCurrentSlice = maximumCost;


    for ( int ySharedMemoryTileIndex = - WINDOW_SIZE_IN_BLOCKS_Y; ySharedMemoryTileIndex <= WINDOW_SIZE_IN_BLOCKS_Y; ySharedMemoryTileIndex++ )
    {
        for ( int xSharedMemoryTileIndex = - WINDOW_SIZE_IN_BLOCKS_X; xSharedMemoryTileIndex <= WINDOW_SIZE_IN_BLOCKS_X; xSharedMemoryTileIndex++ )
        {
            int2 inLoopImageIndex2D                   = make_int2( ( ( blockIdx.x + xSharedMemoryTileIndex ) * blockDim.x ) + threadIdx.x, ( ( blockIdx.y + ySharedMemoryTileIndex ) * blockDim.y ) + threadIdx.y );
            int2 inLoopSharedMemoryIndex2DZeroIndexed = make_int2( ( ( xSharedMemoryTileIndex + WINDOW_SIZE_IN_BLOCKS_X ) * blockDim.x ) + threadIdx.x, ( ( ySharedMemoryTileIndex + WINDOW_SIZE_IN_BLOCKS_Y ) * blockDim.y ) + threadIdx.y );
            
            if ( inLoopImageIndex2D.x >= 0 && inLoopImageIndex2D.x < numVoxels.x &&
                    inLoopImageIndex2D.y >= 0 && inLoopImageIndex2D.y < numVoxels.y )
            {
                int3 inLoopIndex3D                        = make_int3( inLoopImageIndex2D.x, inLoopImageIndex2D.y, currentSlice + adjacentSliceDirection );
                int  inLoopIndex1D                        = Index3DToIndex1D( inLoopIndex3D, numVoxels );
                    
                //
                // using TCudaType to get the texture reference crashes the v3.2 CUDA compiler, so we need this switch statement
                //
                switch ( dxgiFormat )
                {
                    case DXGI_FORMAT_R8_UNORM:
                        s_sourceAdjacentSlice[ inLoopSharedMemoryIndex2DZeroIndexed.y ][ inLoopSharedMemoryIndex2DZeroIndexed.x ] =
                            tex3D( TEXTURE_SOURCE_MAP_UCHAR1_3D_NORMALIZED_FLOAT, inLoopIndex3D.x + 0.5f, inLoopIndex3D.y + 0.5f, inLoopIndex3D.z + 0.5f ).x;
                        break;

                    case DXGI_FORMAT_R8G8B8A8_UNORM:
                        s_sourceAdjacentSlice[ inLoopSharedMemoryIndex2DZeroIndexed.y ][ inLoopSharedMemoryIndex2DZeroIndexed.x ] =
                            tex3D( TEXTURE_SOURCE_MAP_UCHAR4_3D_NORMALIZED_FLOAT, inLoopIndex3D.x + 0.5f, inLoopIndex3D.y + 0.5f, inLoopIndex3D.z + 0.5f ).x;
                        break;
                }

                s_opticalFlowFromAdjacentSliceToCurrentSlice[ inLoopSharedMemoryIndex2DZeroIndexed.y ][ inLoopSharedMemoryIndex2DZeroIndexed.x ] = d_inOpticalFlow[ inLoopIndex1D ];
                s_costAdjacentSlice                         [ inLoopSharedMemoryIndex2DZeroIndexed.y ][ inLoopSharedMemoryIndex2DZeroIndexed.x ] = d_inoutCost[ inLoopIndex1D ];
            }
            else
            {
                s_sourceAdjacentSlice                       [ inLoopSharedMemoryIndex2DZeroIndexed.y ][ inLoopSharedMemoryIndex2DZeroIndexed.x ] = 0.0f;
                s_opticalFlowFromAdjacentSliceToCurrentSlice[ inLoopSharedMemoryIndex2DZeroIndexed.y ][ inLoopSharedMemoryIndex2DZeroIndexed.x ] = make_float2( 0.0f, 0.0f );
                s_costAdjacentSlice                         [ inLoopSharedMemoryIndex2DZeroIndexed.y ][ inLoopSharedMemoryIndex2DZeroIndexed.x ] = maximumCost;
            }
        }
    }

    __syncthreads(); 

    if ( constraint > 0.0f )
    {
        newCostCurrentSlice = maximumCost;
    }
    else
    {
        //
        // using TCudaType to get the texture reference crashes the v3.2 CUDA compiler, so we need this switch statement
        //
        float sourceCurrentSlice;
        switch ( dxgiFormat )
        {
            case DXGI_FORMAT_R8_UNORM:
                sourceCurrentSlice = tex3D( TEXTURE_SOURCE_MAP_UCHAR1_3D_NORMALIZED_FLOAT, inImageIndex3D.x + 0.5f, inImageIndex3D.y + 0.5f, inImageIndex3D.z + 0.5f ).x;
                break;

            case DXGI_FORMAT_R8G8B8A8_UNORM:
                sourceCurrentSlice = tex3D( TEXTURE_SOURCE_MAP_UCHAR4_3D_NORMALIZED_FLOAT, inImageIndex3D.x + 0.5f, inImageIndex3D.y + 0.5f, inImageIndex3D.z + 0.5f ).x;
                break;
        }

        for ( int yAdjacentSlice = - WINDOW_SIZE_IN_PIXELS_Y; yAdjacentSlice <= WINDOW_SIZE_IN_PIXELS_Y; yAdjacentSlice++ )
        {
            for ( int xAdjacentSlice = - WINDOW_SIZE_IN_PIXELS_X; xAdjacentSlice <= WINDOW_SIZE_IN_PIXELS_X; xAdjacentSlice++ )
            {
                int2 inOffsetImageIndex2D = make_int2( inImageIndex2D.x + xAdjacentSlice, inImageIndex2D.y + yAdjacentSlice );
            
                if ( inOffsetImageIndex2D.x >= 0 && inOffsetImageIndex2D.x < numVoxels.x &&
                        inOffsetImageIndex2D.y >= 0 && inOffsetImageIndex2D.y < numVoxels.y )
                {
                    int2   inSharedMemoryIndex2DZeroIndexed           = make_int2( ( WINDOW_SIZE_IN_BLOCKS_X * blockDim.x ) + threadIdx.x + xAdjacentSlice, ( WINDOW_SIZE_IN_BLOCKS_Y * blockDim.y ) + threadIdx.y + yAdjacentSlice );

                    float  sourceAdjacentSlice                        = s_sourceAdjacentSlice                       [ inSharedMemoryIndex2DZeroIndexed.y ][ inSharedMemoryIndex2DZeroIndexed.x ];
                    float  costAdjacentSlice                          = s_costAdjacentSlice                         [ inSharedMemoryIndex2DZeroIndexed.y ][ inSharedMemoryIndex2DZeroIndexed.x ];
                    float2 opticalFlowFromAdjacentSliceToCurrentSlice = s_opticalFlowFromAdjacentSliceToCurrentSlice[ inSharedMemoryIndex2DZeroIndexed.y ][ inSharedMemoryIndex2DZeroIndexed.x ];

                    float   distanceToEstimateGivenByOpticalFlow       = length( make_float2( inImageIndex3D.x + xAdjacentSlice, inImageIndex3D.y + yAdjacentSlice ) + opticalFlowFromAdjacentSliceToCurrentSlice - make_float2( inImageIndex3D.x, inImageIndex3D.y ) );
                    float  sourceIntensityDifference                  = abs( sourceCurrentSlice - sourceAdjacentSlice );
                    float  costCurrentSlice                           = costAdjacentSlice + ( distanceToEstimateGivenByOpticalFlow * opticalFlowImportanceFactor ) + ( sourceIntensityDifference );

                    if ( costCurrentSlice < newCostCurrentSlice )
                    {
                        newCostCurrentSlice = costCurrentSlice; 
                    }
                }
            }
        }
    }

    d_inoutCost[ inImageIndex1D ] = newCostCurrentSlice;
}

template< typename TCudaType >
void IncrementCostMapFromPrimalMapForwardInternal( Mojo::Core::SegmenterState* segmenterState )
{
    if ( segmenterState->slicesWithForegroundConstraints.size() > 1 )
    {
        int minSliceWithForegroundConstraints = *std::min_element( segmenterState->slicesWithForegroundConstraints.begin(), segmenterState->slicesWithForegroundConstraints.end() );
        int maxSliceWithForegroundConstraints = *std::max_element( segmenterState->slicesWithForegroundConstraints.begin(), segmenterState->slicesWithForegroundConstraints.end() );
        
        std::vector< int > slicesWithConstraints;
        slicesWithConstraints.resize( segmenterState->slicesWithForegroundConstraints.size() + segmenterState->slicesWithBackgroundConstraints.size() );
        std::set_union( segmenterState->slicesWithForegroundConstraints.begin(),
                        segmenterState->slicesWithForegroundConstraints.end(),
                        segmenterState->slicesWithBackgroundConstraints.begin(),
                        segmenterState->slicesWithBackgroundConstraints.end(),
                        slicesWithConstraints.begin() );

        Mojo::Core::Printf( "    Computing CostForwardMap..." );
        for ( int z = minSliceWithForegroundConstraints + 1; z < maxSliceWithForegroundConstraints; z++ )
        {
            if ( std::find( slicesWithConstraints.begin(), slicesWithConstraints.end(), z ) == slicesWithConstraints.end() )
            {
                Mojo::Core::Printf( "        Computing forward cost on slice ", z, "..." );

                dim3 blockDimensions( BLOCK_SIZE_X, BLOCK_SIZE_Y );
                dim3 gridDimensions( segmenterState->volumeDescription.numVoxels.x / BLOCK_SIZE_X, segmenterState->volumeDescription.numVoxels.y / BLOCK_SIZE_Y ); 

                IncrementCostKernel< TCudaType ><<< gridDimensions, blockDimensions >>>(
                    thrust::raw_pointer_cast< float  >( &segmenterState->deviceVectors.Get< float  >( "ConstraintMap"         )[ 0 ] ),
                    thrust::raw_pointer_cast< float2 >( &segmenterState->deviceVectors.Get< float2 >( "OpticalFlowForwardMap" )[ 0 ] ),
                    thrust::raw_pointer_cast< float  >( &segmenterState->deviceVectors.Get< float  >( "CostForwardMap"        )[ 0 ] ),
                    segmenterState->volumeDescription.numVoxels,
                    segmenterState->volumeDescription.dxgiFormat,
                    z,
                    -1,
                    segmenterState->constParameters.Get< float >( "COST_MAP_OPTICAL_FLOW_IMPORTANCE_FACTOR" ),
                    segmenterState->constParameters.Get< float >( "COST_MAP_MAX_VALUE" ) );

                Mojo::Core::Cuda::Synchronize();
            }
        }
    }
}

template< typename TCudaType >
void IncrementCostMapFromPrimalMapBackwardInternal( Mojo::Core::SegmenterState* segmenterState )
{
    if ( segmenterState->slicesWithForegroundConstraints.size() > 1 )
    {
        int minSliceWithForegroundConstraints = *std::min_element( segmenterState->slicesWithForegroundConstraints.begin(), segmenterState->slicesWithForegroundConstraints.end() );
        int maxSliceWithForegroundConstraints = *std::max_element( segmenterState->slicesWithForegroundConstraints.begin(), segmenterState->slicesWithForegroundConstraints.end() );
        
        std::vector< int > slicesWithConstraints;
        slicesWithConstraints.resize( segmenterState->slicesWithForegroundConstraints.size() + segmenterState->slicesWithBackgroundConstraints.size() );
        std::set_union( segmenterState->slicesWithForegroundConstraints.begin(),
                        segmenterState->slicesWithForegroundConstraints.end(),
                        segmenterState->slicesWithBackgroundConstraints.begin(),
                        segmenterState->slicesWithBackgroundConstraints.end(),
                        slicesWithConstraints.begin() );

        Mojo::Core::Printf( "    Computing CostBackwardMap..." );
        for ( int z = maxSliceWithForegroundConstraints - 1; z > minSliceWithForegroundConstraints; z-- )
        {
            if ( std::find( slicesWithConstraints.begin(), slicesWithConstraints.end(), z ) == slicesWithConstraints.end() )
            {
                Mojo::Core::Printf( "        Computing backward cost on slice ", z, "..." );

                dim3 blockDimensions( BLOCK_SIZE_X, BLOCK_SIZE_Y );
                dim3 gridDimensions( segmenterState->volumeDescription.numVoxels.x / BLOCK_SIZE_X, segmenterState->volumeDescription.numVoxels.y / BLOCK_SIZE_Y ); 

                IncrementCostKernel< TCudaType ><<< gridDimensions, blockDimensions >>>(
                    thrust::raw_pointer_cast< float  >( &segmenterState->deviceVectors.Get< float  >( "ConstraintMap"          )[ 0 ] ),
                    thrust::raw_pointer_cast< float2 >( &segmenterState->deviceVectors.Get< float2 >( "OpticalFlowBackwardMap" )[ 0 ] ),
                    thrust::raw_pointer_cast< float  >( &segmenterState->deviceVectors.Get< float  >( "CostBackwardMap"        )[ 0 ] ),
                    segmenterState->volumeDescription.numVoxels,
                    segmenterState->volumeDescription.dxgiFormat,
                    z,
                    1,
                    segmenterState->constParameters.Get< float >( "COST_MAP_OPTICAL_FLOW_IMPORTANCE_FACTOR" ),
                    segmenterState->constParameters.Get< float >( "COST_MAP_MAX_VALUE" ) );

                Mojo::Core::Cuda::Synchronize();
            }
        }
    }
}

extern "C" void IncrementCostMapFromPrimalMapForward( Mojo::Core::SegmenterState* segmenterState )
{
    switch ( segmenterState->volumeDescription.dxgiFormat )
    {
        case DXGI_FORMAT_R8_UNORM:
            IncrementCostMapFromPrimalMapForwardInternal< uchar1 >( segmenterState );
            break;

        case DXGI_FORMAT_R8G8B8A8_UNORM:
            IncrementCostMapFromPrimalMapForwardInternal< uchar4 >( segmenterState );
            break;

        default:
            RELEASE_ASSERT( 0 );
    }
}

extern "C" void IncrementCostMapFromPrimalMapBackward( Mojo::Core::SegmenterState* segmenterState )
{
    switch ( segmenterState->volumeDescription.dxgiFormat )
    {
        case DXGI_FORMAT_R8_UNORM:
            IncrementCostMapFromPrimalMapBackwardInternal< uchar1 >( segmenterState );
            break;

        case DXGI_FORMAT_R8G8B8A8_UNORM:
            IncrementCostMapFromPrimalMapBackwardInternal< uchar4 >( segmenterState );
            break;

        default:
            RELEASE_ASSERT( 0 );
    }
}

extern "C" void FinalizeCostMapFromPrimalMap( Mojo::Core::SegmenterState* segmenterState )
{
    if ( segmenterState->slicesWithForegroundConstraints.size() > 1 )
    {
        int numElementsPerSlice               = segmenterState->volumeDescription.numVoxels.x * segmenterState->volumeDescription.numVoxels.y;
        int minSliceWithForegroundConstraints = *std::min_element( segmenterState->slicesWithForegroundConstraints.begin(), segmenterState->slicesWithForegroundConstraints.end() );
        int maxSliceWithForegroundConstraints = *std::max_element( segmenterState->slicesWithForegroundConstraints.begin(), segmenterState->slicesWithForegroundConstraints.end() );
        
        std::vector< int > slicesWithConstraints;
        slicesWithConstraints.resize( segmenterState->slicesWithForegroundConstraints.size() + segmenterState->slicesWithBackgroundConstraints.size() );
        std::set_union( segmenterState->slicesWithForegroundConstraints.begin(),
                        segmenterState->slicesWithForegroundConstraints.end(),
                        segmenterState->slicesWithBackgroundConstraints.begin(),
                        segmenterState->slicesWithBackgroundConstraints.end(),
                        slicesWithConstraints.begin() );

        Mojo::Core::Printf( "    Computing total cost..." );
        MOJO_THRUST_SAFE(
            thrust::transform(
                segmenterState->deviceVectors.Get< float >( "CostForwardMap"  ).begin() + ( numElementsPerSlice * minSliceWithForegroundConstraints ),
                segmenterState->deviceVectors.Get< float >( "CostForwardMap"  ).begin() + ( numElementsPerSlice * ( maxSliceWithForegroundConstraints + 1 ) ),
                segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ).begin() + ( numElementsPerSlice * minSliceWithForegroundConstraints ),
                segmenterState->deviceVectors.Get< float >( "ScratchpadMap"   ).begin() + ( numElementsPerSlice * minSliceWithForegroundConstraints ),
                thrust::plus< float >() ) );

        Mojo::Core::Printf( "    Copying total cost into CostForwardMap..." );
        MOJO_THRUST_SAFE(
            thrust::copy(
                segmenterState->deviceVectors.Get< float >( "ScratchpadMap"  ).begin(),
                segmenterState->deviceVectors.Get< float >( "ScratchpadMap"  ).end(),
                segmenterState->deviceVectors.Get< float >( "CostForwardMap" ).begin() ) );

        Mojo::Core::Printf( "    Finding minimum costs per slice..." );

        segmenterState->minCostsPerSlice.GetHashMap().clear();

        for ( int z = minSliceWithForegroundConstraints; z <= maxSliceWithForegroundConstraints; z++ )
        {
            float minCostOnSlice = Mojo::Core::Thrust::Reduce(
                segmenterState->deviceVectors.Get< float >( "CostForwardMap" ).begin() + ( numElementsPerSlice * ( z + 0 ) ),
                segmenterState->deviceVectors.Get< float >( "CostForwardMap" ).begin() + ( numElementsPerSlice * ( z + 1 ) ),
                segmenterState->constParameters.Get< float >( "COST_MAP_MAX_VALUE" ),
                thrust::minimum< float >(),
                segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ) );

            segmenterState->minCostsPerSlice.Set( z, minCostOnSlice );
        }

        Mojo::Core::Printf( "    Finished computing cost." );
    }
}

struct UpdateConstraintMapAndPrimalMapFromCostMapFunction
{
    enum TupleLayout
    {
        COST_MAP,
        CONSTRAINT_MAP,
        PRIMAL_MAP
    };

    float mMinPossibleForegroundCost;
    float mMaxPossibleForegroundCost;
    float mMinPossibleForegroundConstraintValue;
    float mMaxPossibleBackgroundConstraintValue;
    float mHardForegroundConstraintValue;
    float mHardBackgroundConstraintValue;
    float mPrimalMapForeground;

    UpdateConstraintMapAndPrimalMapFromCostMapFunction( float minPossibleForegroundCost,
                                                        float maxPossibleForegroundCost,
                                                        float minPossibleForegroundConstraintValue,
                                                        float maxPossibleBackgroundConstraintValue,
                                                        float hardForegroundConstraintValue,
                                                        float hardBackgroundConstraintValue,
                                                        float primalMapForeground ) :
        mMinPossibleForegroundCost           ( minPossibleForegroundCost ),
        mMaxPossibleForegroundCost           ( maxPossibleForegroundCost ),
        mMinPossibleForegroundConstraintValue( minPossibleForegroundConstraintValue ),
        mMaxPossibleBackgroundConstraintValue( maxPossibleBackgroundConstraintValue ),
        mHardForegroundConstraintValue       ( hardForegroundConstraintValue ),
        mHardBackgroundConstraintValue       ( hardBackgroundConstraintValue ),
        mPrimalMapForeground                 ( primalMapForeground ) {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        float cost = thrust::get< COST_MAP >( tuple );

        float constraintNoClamp =
            ( ( cost - mMinPossibleForegroundCost ) * ( mMaxPossibleBackgroundConstraintValue - ( mMinPossibleForegroundConstraintValue ) ) /
            ( mMaxPossibleForegroundCost - mMinPossibleForegroundCost ) ) + ( mMinPossibleForegroundConstraintValue );

        float constraint = clamp( constraintNoClamp, mMinPossibleForegroundConstraintValue, mMaxPossibleBackgroundConstraintValue );

        if ( constraint < 0.0f )
        {
            thrust::get< CONSTRAINT_MAP >( tuple ) = clamp( thrust::get< CONSTRAINT_MAP >( tuple ) + constraint, mHardForegroundConstraintValue, mHardBackgroundConstraintValue );
            thrust::get< PRIMAL_MAP >( tuple )     = mPrimalMapForeground;
        }
    }
};

extern "C" void UpdateConstraintMapAndPrimalMapFromCostMap( Mojo::Core::SegmenterState* segmenterState )
{
    if ( segmenterState->slicesWithForegroundConstraints.size() > 1 )
    {
        int numElementsPerSlice               = segmenterState->volumeDescription.numVoxels.x * segmenterState->volumeDescription.numVoxels.y;
        int minSliceWithForegroundConstraints = *std::min_element( segmenterState->slicesWithForegroundConstraints.begin(), segmenterState->slicesWithForegroundConstraints.end() );
        int maxSliceWithForegroundConstraints = *std::max_element( segmenterState->slicesWithForegroundConstraints.begin(), segmenterState->slicesWithForegroundConstraints.end() );

        for ( int z = minSliceWithForegroundConstraints + 1; z < maxSliceWithForegroundConstraints; z++ )
        {
            MOJO_THRUST_SAFE(
                thrust::for_each(
                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            segmenterState->deviceVectors.Get< float >( "CostForwardMap" ).begin() + ( numElementsPerSlice * ( z ) ),
                            segmenterState->deviceVectors.Get< float >( "ConstraintMap"  ).begin() + ( numElementsPerSlice * ( z ) ),
                            segmenterState->deviceVectors.Get< float >( "PrimalMap"      ).begin() + ( numElementsPerSlice * ( z ) ) ) ),

                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            segmenterState->deviceVectors.Get< float >( "CostForwardMap" ).begin() + ( numElementsPerSlice * ( z + 1 ) ),
                            segmenterState->deviceVectors.Get< float >( "ConstraintMap"  ).begin() + ( numElementsPerSlice * ( z + 1 ) ),
                            segmenterState->deviceVectors.Get< float >( "PrimalMap"      ).begin() + ( numElementsPerSlice * ( z + 1 ) ) ) ),
                               
                    UpdateConstraintMapAndPrimalMapFromCostMapFunction(
                        segmenterState->minCostsPerSlice.Get( z ),
                        segmenterState->minCostsPerSlice.Get( z ) + segmenterState->dynamicParameters.Get< float >( "MaxForegroundCostDelta" ),
                        segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_INITIALIZE_FROM_COST_MAP_MIN_FOREGROUND" ),
                        segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_INITIALIZE_FROM_COST_MAP_MAX_BACKGROUND" ),
                        segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_HARD_FOREGROUND_AUTO" ),
                        segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_HARD_BACKGROUND_AUTO" ),
                        segmenterState->constParameters.Get< float >( "PRIMAL_MAP_FOREGROUND" ) ) ) );
        }
    }
}
