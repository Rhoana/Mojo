#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/Thrust.hpp"
#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/SegmenterState.hpp"
#include "Mojo.Core/ID3D11CudaTexture.hpp"

#include "Index.cuh"

template < typename TCudaType >
struct InitializeEdgeXYMapFunction
{
    enum TupleLayout
    {
        INDEX_3D,
        EDGE_XY_MAP
    };

    TCudaType* mSourceMapBuffer;
    int3       mNumVoxels;
    float      mMinSourceValue;
    float      mMaxSourceValue;
    float      mEdgeMultiplier;
    float      mEdgePower;
    float      mEdgeMaxBeforeSaturate;

    InitializeEdgeXYMapFunction( TCudaType* sourceMapBuffer,
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
        int3   index3D                   = thrust::get< INDEX_3D >( tuple );
        int3   index3DForwardX           = make_int3( min( mNumVoxels.x - 1, index3D.x + 1 ), index3D.y,                              index3D.z );
        int3   index3DForwardY           = make_int3( index3D.x,                              min( mNumVoxels.y - 1, index3D.y + 1 ), index3D.z );
                                              
        float  source                    = ( __uint2float_rd( mSourceMapBuffer[ Index3DToIndex1D( index3D,         mNumVoxels ) ].x ) - mMinSourceValue ) / mMaxSourceValue;
        float  sourceForwardX            = ( __uint2float_rd( mSourceMapBuffer[ Index3DToIndex1D( index3DForwardX, mNumVoxels ) ].x ) - mMinSourceValue ) / mMaxSourceValue;
        float  sourceForwardY            = ( __uint2float_rd( mSourceMapBuffer[ Index3DToIndex1D( index3DForwardY, mNumVoxels ) ].x ) - mMinSourceValue ) / mMaxSourceValue;

        float2 sourceGradientXY          = make_float2( sourceForwardX - source, sourceForwardY - source );
        float  sourceGradientXYMagnitude = sqrt( ( sourceGradientXY.x * sourceGradientXY.x ) + ( sourceGradientXY.y * sourceGradientXY.y ) );
        float  edgeXY                    = pow( 2.71f, -1.0f * mEdgeMultiplier * pow( sourceGradientXYMagnitude, mEdgePower ) );

        if ( edgeXY > mEdgeMaxBeforeSaturate )
        {
            edgeXY = 1.0f;
        }

        thrust::get< EDGE_XY_MAP >( tuple ) = edgeXY;
    }
};

template < typename TCudaType >
void InitializeEdgeXYMapInternal( Mojo::Core::SegmenterState* segmenterState, Mojo::Core::HashMap< std::string, Mojo::Core::VolumeDescription >* volumeDescriptions, float minSourceValue, float maxSourceValue )
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
                    segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).begin() ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator( 
                        thrust::make_counting_iterator( numElements ),
                        Index1DToIndex3DFunction( volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).end() ) ),
              
            InitializeEdgeXYMapFunction< TCudaType >(
                (TCudaType*)thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float >( "ScratchpadMap" )[ 0 ] ),
                volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels,
                minSourceValue,
                maxSourceValue,
                segmenterState->constParameters.Get< float >( "EDGE_MULTIPLIER" ),
                segmenterState->constParameters.Get< float >( "EDGE_POWER_XY" ),
                segmenterState->constParameters.Get< float >( "EDGE_MAX_BEFORE_SATURATE" ) ) ) );
}

extern "C" void InitializeEdgeXYMap( Mojo::Core::SegmenterState* segmenterState, Mojo::Core::HashMap< std::string, Mojo::Core::VolumeDescription >* volumeDescriptions )
{
    switch ( segmenterState->volumeDescription.dxgiFormat )
    {
        case DXGI_FORMAT_R8_UNORM:
            InitializeEdgeXYMapInternal< uchar1 >( segmenterState, volumeDescriptions, 0.0f, 255.0f );
            break;

        case DXGI_FORMAT_R8G8B8A8_UNORM:
            InitializeEdgeXYMapInternal< uchar4 >( segmenterState, volumeDescriptions, 0.0f, 255.0f );
            break;

        default:
            RELEASE_ASSERT( 0 );
    }
}


struct InitializeStencilMapFromIdMapFunction
{
    enum TupleLayout
    {
        ID_MAP,
        STENCIL_MAP
    };

    int   mId;
    float mStencilMapBackgroundValue;

    InitializeStencilMapFromIdMapFunction( int id, float stencilMapBackgroundValue ) : mId( id ), mStencilMapBackgroundValue( stencilMapBackgroundValue ) {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int id = thrust::get< ID_MAP >( tuple );

        if ( id != mId )
        {
            thrust::get< STENCIL_MAP >( tuple ) = mStencilMapBackgroundValue;
        }
    }
};

struct UpdateStencilMapFunction
{
    float* mStencilMapBuffer;
    int3   mNumVoxels;
    float  mOldValue;
    float  mNeighborValue;
    float  mNewValue;

    UpdateStencilMapFunction( float* stencilMapBuffer,
                              int3   numVoxels,
                              float  oldValue,
                              float  neighborValue,
                              float  newValue ) :
        mStencilMapBuffer( stencilMapBuffer ),
        mNumVoxels       ( numVoxels        ),
        mOldValue        ( oldValue         ),
        mNeighborValue   ( neighborValue    ),
        mNewValue        ( newValue         )
    {}

    __device__ void operator() ( int3 index3D )
    {
        int   index1D         = Index3DToIndex1D( index3D, mNumVoxels );
        float stencilMapValue = mStencilMapBuffer[ index1D ];

        if ( stencilMapValue == mOldValue )
        {
            int3  index3DForwardX          = make_int3( min( mNumVoxels.x - 1, index3D.x + 1 ), index3D.y,                              index3D.z );
            int3  index3DForwardY          = make_int3( index3D.x,                              min( mNumVoxels.y - 1, index3D.y + 1 ), index3D.z );
            int3  index3DBackwardX         = make_int3( max( 0, index3D.x - 1 ),                index3D.y,                              index3D.z );
            int3  index3DBackwardY         = make_int3( index3D.x,                              max( 0, index3D.y - 1 ),                index3D.z );
                                           
            int   index1DForwardX          = Index3DToIndex1D( index3DForwardX,  mNumVoxels );
            int   index1DForwardY          = Index3DToIndex1D( index3DForwardY,  mNumVoxels );
            int   index1DBackwardX         = Index3DToIndex1D( index3DBackwardX, mNumVoxels );
            int   index1DBackwardY         = Index3DToIndex1D( index3DBackwardY, mNumVoxels );

            float stencilMapValueForwardX  = mStencilMapBuffer[ index1DForwardX  ];
            float stencilMapValueForwardY  = mStencilMapBuffer[ index1DForwardY  ];
            float stencilMapValueBackwardX = mStencilMapBuffer[ index1DBackwardX ];
            float stencilMapValueBackwardY = mStencilMapBuffer[ index1DBackwardY ];

            if ( stencilMapValueForwardX  == mNeighborValue ||
                 stencilMapValueForwardY  == mNeighborValue ||
                 stencilMapValueBackwardX == mNeighborValue ||
                 stencilMapValueBackwardY == mNeighborValue )
            {
                mStencilMapBuffer[ index1D ] = mNewValue;
            }
        }
    }
};

template < typename TCudaType >
struct InitializeEdgeXYMapForSplittingFunction
{
    enum TupleLayout
    {
        INDEX_3D,
        STENCIL_MAP,
        EDGE_XY_MAP
    };

    TCudaType* mSourceMapBuffer;
    int3       mNumVoxels;
    float      mMinSourceValue;
    float      mMaxSourceValue;
    float      mEdgeMultiplier;
    float      mEdgePower;
    float      mEdgeBoost;
    float      mStrongestEdgeStencilValue;
    float      mWeakestEdgeStencilValue;

    InitializeEdgeXYMapForSplittingFunction( TCudaType* sourceMapBuffer,
                                             int3       numVoxels,
                                             float      minSourceValue,
                                             float      maxSourceValue,
                                             float      edgeMultiplier,
                                             float      edgePower,
                                             float      edgeBoost,
                                             float      strongestEdgeStencilValue,
                                             float      weakestEdgeStencilValue ) :
        mSourceMapBuffer          ( sourceMapBuffer           ),
        mNumVoxels                ( numVoxels                 ),
        mMinSourceValue           ( minSourceValue            ),
        mMaxSourceValue           ( maxSourceValue            ),
        mEdgeMultiplier           ( edgeMultiplier            ),
        mEdgePower                ( edgePower                 ),
        mEdgeBoost                ( edgeBoost                 ),
        mStrongestEdgeStencilValue( strongestEdgeStencilValue ),
        mWeakestEdgeStencilValue  ( weakestEdgeStencilValue   )
    {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        float stencil = thrust::get< STENCIL_MAP >( tuple );
        float edgeXY;

        if ( stencil == mStrongestEdgeStencilValue )
        {
            edgeXY = 0.0f;
        }
        else
        if ( stencil == mWeakestEdgeStencilValue )
        {
            edgeXY = 1.0f;
        }
        else
        {
            int3   index3D                   = thrust::get< INDEX_3D >( tuple );
            int3   index3DForwardX           = make_int3( min( mNumVoxels.x - 1, index3D.x + 1 ), index3D.y,                              index3D.z );
            int3   index3DForwardY           = make_int3( index3D.x,                              min( mNumVoxels.y - 1, index3D.y + 1 ), index3D.z );

            float  source                    = ( __uint2float_rd( mSourceMapBuffer[ Index3DToIndex1D( index3D,         mNumVoxels ) ].x ) - mMinSourceValue ) / mMaxSourceValue;
            float  sourceForwardX            = ( __uint2float_rd( mSourceMapBuffer[ Index3DToIndex1D( index3DForwardX, mNumVoxels ) ].x ) - mMinSourceValue ) / mMaxSourceValue;
            float  sourceForwardY            = ( __uint2float_rd( mSourceMapBuffer[ Index3DToIndex1D( index3DForwardY, mNumVoxels ) ].x ) - mMinSourceValue ) / mMaxSourceValue;

            float2 sourceGradientXY          = make_float2( sourceForwardX - source, sourceForwardY - source );
            float  sourceGradientXYMagnitude = sqrt( ( sourceGradientXY.x * sourceGradientXY.x ) + ( sourceGradientXY.y * sourceGradientXY.y ) );
            edgeXY                           = mEdgeBoost + pow( 2.71f, -1.0f * mEdgeMultiplier * pow( sourceGradientXYMagnitude, mEdgePower ) );

            if ( edgeXY > 1.0f )
            {
                edgeXY = 1.0f;
            }
        }

        thrust::get< EDGE_XY_MAP >( tuple ) = edgeXY;
    }
};

template < typename TCudaType >
void InitializeEdgeXYMapForSplittingInternal( Mojo::Core::SegmenterState* segmenterState, Mojo::Core::HashMap< std::string, Mojo::Core::VolumeDescription >* volumeDescriptions, int id, float minSourceValue, float maxSourceValue )
{
    int numElements = volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels.x * volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels.y * volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels.z;


    //
    // put the image data in ScratchpadMap
    //
    MOJO_CUDA_SAFE( cudaMemcpy( thrust::raw_pointer_cast< float >( &segmenterState->deviceVectors.Get< float >( "ScratchpadMap" )[ 0 ] ),
                                volumeDescriptions->Get( "FilteredSourceMap" ).data,
                                numElements * sizeof( TCudaType ),
                                cudaMemcpyHostToDevice ) );


    //
    // ScratchpadMap is already being used to store the image data, so we need to use CostBackwardMap to temporarily store our stencil map
    //
    MOJO_THRUST_SAFE(
        thrust::fill(
            segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ).begin(),
            segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ).end(),
            segmenterState->constParameters.Get< float >( "STENCIL_MAP_INITIAL_VALUE" ) ) );


    //
    // set the stencil map background values
    //
    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int   >( "IdMap" ).begin(),
                    segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ).begin() ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int   >( "IdMap" ).end(),
                    segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ).end() ) ),

            InitializeStencilMapFromIdMapFunction(
                id,
                segmenterState->constParameters.Get< float >( "STENCIL_MAP_BACKGROUND_VALUE" ) ) ) );


    //
    // shrink the background regions slightly
    //
    for ( int i = 0; i < segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_SPLIT_BACKGROUND_CONTRACT_NUM_PASSES" ); i++ )
    {
        MOJO_THRUST_SAFE(
            thrust::for_each(
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator( 0 ),
                    Index1DToIndex3DFunction( volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels ) ),

                thrust::make_transform_iterator(
                        thrust::make_counting_iterator( numElements ),
                        Index1DToIndex3DFunction( volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels ) ),

                UpdateStencilMapFunction(
                    thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float >( "CostBackwardMap" )[ 0 ] ),
                    segmenterState->volumeDescription.numVoxels,
                    segmenterState->constParameters.Get< float >( "STENCIL_MAP_BACKGROUND_VALUE" ),
                    segmenterState->constParameters.Get< float >( "STENCIL_MAP_INITIAL_VALUE" ),
                    segmenterState->constParameters.Get< float >( "STENCIL_MAP_INITIAL_VALUE" ) ) ) );
    }

    //
    // set the stencil map strongest edge values
    //
    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_transform_iterator(
                thrust::make_counting_iterator( 0 ),
                Index1DToIndex3DFunction( volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels ) ),

            thrust::make_transform_iterator(
                    thrust::make_counting_iterator( numElements ),
                    Index1DToIndex3DFunction( volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels ) ),

            UpdateStencilMapFunction(
                thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float >( "CostBackwardMap" )[ 0 ] ),
                segmenterState->volumeDescription.numVoxels,
                segmenterState->constParameters.Get< float >( "STENCIL_MAP_INITIAL_VALUE" ),
                segmenterState->constParameters.Get< float >( "STENCIL_MAP_BACKGROUND_VALUE" ),
                segmenterState->constParameters.Get< float >( "STENCIL_MAP_STRONGEST_EDGE_VALUE" ) ) ) );


    //
    // initialize the edge map based on the stencil map
    //
    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( 0 ),
                        Index1DToIndex3DFunction( volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ).begin(),
                    segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).begin() ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator( 
                        thrust::make_counting_iterator( numElements ),
                        Index1DToIndex3DFunction( volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ).end(),
                    segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).end() ) ),
              
            InitializeEdgeXYMapForSplittingFunction< TCudaType >(
                (TCudaType*)thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float >( "ScratchpadMap" )[ 0 ] ),
                volumeDescriptions->Get( "FilteredSourceMap" ).numVoxels,
                minSourceValue,
                maxSourceValue,
                segmenterState->constParameters.Get< float >( "EDGE_MULTIPLIER" ),
                segmenterState->constParameters.Get< float >( "EDGE_POWER_XY" ),
                segmenterState->constParameters.Get< float >( "EDGE_SPLIT_BOOST" ),
                segmenterState->constParameters.Get< float >( "STENCIL_MAP_STRONGEST_EDGE_VALUE" ),
                segmenterState->constParameters.Get< float >( "STENCIL_MAP_WEAKEST_EDGE_VALUE" ) ) ) );
}

extern "C" void InitializeEdgeXYMapForSplitting( Mojo::Core::SegmenterState* segmenterState, Mojo::Core::HashMap< std::string, Mojo::Core::VolumeDescription >* volumeDescriptions, int id )
{
    switch ( segmenterState->volumeDescription.dxgiFormat )
    {
        case DXGI_FORMAT_R8_UNORM:
            InitializeEdgeXYMapForSplittingInternal< uchar1 >( segmenterState, volumeDescriptions, id, 0.0f, 255.0f );
            break;

        case DXGI_FORMAT_R8G8B8A8_UNORM:
            InitializeEdgeXYMapForSplittingInternal< uchar4 >( segmenterState, volumeDescriptions, id, 0.0f, 255.0f );
            break;

        default:
            RELEASE_ASSERT( 0 );
    }
}