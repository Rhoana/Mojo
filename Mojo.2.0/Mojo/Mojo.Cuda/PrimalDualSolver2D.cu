#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/Thrust.hpp"
#include "Mojo.Core/SegmenterState.hpp"
#include "Mojo.Core/ID3D11CudaTexture.hpp"

#include "Index.cuh"
#include "Math.cuh"

struct UpdateDualMap2DFunction
{
    enum TupleLayout
    {
        INDEX_3D,
        EDGE_XY_MAP,
        DUAL_MAP
    };

    float* mOldPrimalMapBuffer;
    int3   mNumVoxels;
    float  mSigma;

    UpdateDualMap2DFunction( float* oldPrimalMapBuffer, int3 numVoxels, float sigma ) : mOldPrimalMapBuffer( oldPrimalMapBuffer ), mNumVoxels( numVoxels ), mSigma( sigma ) {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int3   index3D                   = thrust::get< INDEX_3D >( tuple );
        float  edgeXY                    = thrust::get< EDGE_XY_MAP >( tuple );
        float4 dual                      = thrust::get< DUAL_MAP >( tuple );

        int3   index3DForwardX           = make_int3( min( mNumVoxels.x - 1, index3D.x + 1 ), index3D.y,                              index3D.z );
        int3   index3DForwardY           = make_int3( index3D.x,                              min( mNumVoxels.y - 1, index3D.y + 1 ), index3D.z );

        float  oldPrimal                 = mOldPrimalMapBuffer[ Index3DToIndex1D( index3D,         mNumVoxels ) ];
        float  oldPrimalForwardX         = mOldPrimalMapBuffer[ Index3DToIndex1D( index3DForwardX, mNumVoxels ) ];
        float  oldPrimalForwardY         = mOldPrimalMapBuffer[ Index3DToIndex1D( index3DForwardY, mNumVoxels ) ];

        float2 oldPrimalGradientXY       = make_float2( oldPrimalForwardX - oldPrimal, oldPrimalForwardY - oldPrimal );

        float2 dualXY                    = make_float2( dual.x, dual.y ) + ( mSigma * oldPrimalGradientXY );
        float  dualXYLength              = sqrt( ( dualXY.x * dualXY.x ) + ( dualXY.y * dualXY.y ) );
        float2 dualXYProjected           = dualXY / max( 1.0f, dualXYLength / edgeXY );

        thrust::get< DUAL_MAP >( tuple ) = make_float4( dualXYProjected.x, dualXYProjected.y, 0.0f, 0.0f );
    }
};

extern "C" void UpdateDualMap2D( Mojo::Core::SegmenterState* segmenterState, float sigma, int zSlice )
{
    int numElementsPerSlice = segmenterState->volumeDescription.numVoxels.x * segmenterState->volumeDescription.numVoxels.y;

    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( numElementsPerSlice * zSlice ),
                        Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).begin() + ( numElementsPerSlice * ( zSlice + 0 ) ),
                    segmenterState->deviceVectors.Get< float4 >( "DualMap" ).begin()  + ( numElementsPerSlice * ( zSlice + 0 ) ) ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( numElementsPerSlice * ( zSlice + 1 ) ),
                        Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).begin() + ( numElementsPerSlice * ( zSlice + 1 ) ),
                    segmenterState->deviceVectors.Get< float4 >( "DualMap" ).begin()  + ( numElementsPerSlice * ( zSlice + 1 ) ) ) ),
              
            UpdateDualMap2DFunction(
                thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float >( "OldPrimalMap" )[ 0 ] ),
                segmenterState->volumeDescription.numVoxels,
                sigma ) ) );
};


struct UpdatePrimalMap2DFunction
{
    enum TupleLayout
    {
        INDEX_3D,
        CONSTRAINT_MAP,
        PRIMAL_MAP,
        OLD_PRIMAL_MAP
    };

    float4* mDualMapBuffer;
    int3    mNumVoxels;
    float   mLambda, mTau;

    UpdatePrimalMap2DFunction( float4* dualMapBuffer, int3 numVoxels, float lambda, float tau ) : mDualMapBuffer( dualMapBuffer ), mNumVoxels( numVoxels ), mLambda( lambda ), mTau( tau ) {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int3   index3D                         = thrust::get< INDEX_3D >( tuple );
        float  constraint                      = thrust::get< CONSTRAINT_MAP >( tuple );
        float  primal                          = thrust::get< PRIMAL_MAP >( tuple );
        float  oldPrimal                       = primal;

        int3   index3DBackwardX                = make_int3( max( 0, index3D.x - 1 ), index3D.y,               index3D.z );
        int3   index3DBackwardY                = make_int3( index3D.x,               max( 0, index3D.y - 1 ), index3D.z );

        float4 dual                            = mDualMapBuffer[ Index3DToIndex1D( index3D,          mNumVoxels ) ];
        float4 dualBackwardX                   = mDualMapBuffer[ Index3DToIndex1D( index3DBackwardX, mNumVoxels ) ];
        float4 dualBackwardY                   = mDualMapBuffer[ Index3DToIndex1D( index3DBackwardY, mNumVoxels ) ];

        if ( index3D.x == 0 )
            dualBackwardX.x = 0.0f;
        else if ( index3D.x >= mNumVoxels.x )
            dual.x = 0.0f;

        if ( index3D.y == 0 )
            dualBackwardY.y = 0.0f;
        else if ( index3D.y >= mNumVoxels.y )
            dual.y = 0.0f;

        float  dualDivergence                  = dual.x - dualBackwardX.x + dual.y - dualBackwardY.y;
        primal                                 = clamp( primal + mTau * ( dualDivergence - constraint * mLambda ), 0.0f, 1.0f );

        thrust::get< PRIMAL_MAP >( tuple )     = primal;
        thrust::get< OLD_PRIMAL_MAP >( tuple ) = 2.0f * primal - oldPrimal;
    }
};

extern "C" void UpdatePrimalMap2D( Mojo::Core::SegmenterState* segmenterState, float lambda, float tau, int zSlice )
{
    int numElementsPerSlice = segmenterState->volumeDescription.numVoxels.x * segmenterState->volumeDescription.numVoxels.y;

    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( numElementsPerSlice * zSlice ),
                        Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin() + ( numElementsPerSlice * ( zSlice + 0 ) ),
                    segmenterState->deviceVectors.Get< float >( "PrimalMap" ).begin()     + ( numElementsPerSlice * ( zSlice + 0 ) ),
                    segmenterState->deviceVectors.Get< float >( "OldPrimalMap" ).begin()  + ( numElementsPerSlice * ( zSlice + 0 ) ) ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( numElementsPerSlice * ( zSlice + 1 ) ),
                        Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin() + ( numElementsPerSlice * ( zSlice + 1 ) ),
                    segmenterState->deviceVectors.Get< float >( "PrimalMap" ).begin()     + ( numElementsPerSlice * ( zSlice + 1 ) ),
                    segmenterState->deviceVectors.Get< float >( "OldPrimalMap" ).begin()  + ( numElementsPerSlice * ( zSlice + 1 ) ) ) ),
              
            UpdatePrimalMap2DFunction(
                thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float4 >( "DualMap" )[ 0 ] ),
                segmenterState->volumeDescription.numVoxels,
                lambda,
                tau ) ) );
};


struct DualEnergy2DFunction
{
    enum TupleLayout
    {
        INDEX_3D,
        CONSTRAINT_MAP,
        SCRATCHPAD_MAP
    };

    float4* mDualMapBuffer;
    int3    mNumVoxels;
    float   mLambda;

    DualEnergy2DFunction( float4* dualMapBuffer, int3 numVoxels, float lambda ) : mDualMapBuffer( dualMapBuffer ), mNumVoxels( numVoxels ), mLambda( lambda ) {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int3   index3D                          = thrust::get< INDEX_3D >( tuple );
        float  constraint                       = thrust::get< CONSTRAINT_MAP >( tuple );
                                                
        int3   index3DBackwardX                 = make_int3( max( 0, index3D.x - 1 ), index3D.y,               index3D.z );
        int3   index3DBackwardY                 = make_int3( index3D.x,               max( 0, index3D.y - 1 ), index3D.z );
                                                
        float4 dual                             = mDualMapBuffer[ Index3DToIndex1D( index3D,          mNumVoxels ) ];
        float4 dualBackwardX                    = mDualMapBuffer[ Index3DToIndex1D( index3DBackwardX, mNumVoxels ) ];
        float4 dualBackwardY                    = mDualMapBuffer[ Index3DToIndex1D( index3DBackwardY, mNumVoxels ) ];

        if ( index3D.x == 0 )
            dualBackwardX.x = 0.0f;
        else if ( index3D.x >= mNumVoxels.x )
            dual.x = 0.0f;

        if ( index3D.y == 0 )
            dualBackwardY.y = 0.0f;
        else if ( index3D.y >= mNumVoxels.y )
            dual.y = 0.0f;

        float  dualDivergence                   = dual.x - dualBackwardX.x + dual.y - dualBackwardY.y;
        float  energy                           = ( -dualDivergence ) + ( mLambda * constraint );

        if ( energy > 0.0f )
            energy = 0.0f;

        thrust::get< SCRATCHPAD_MAP >( tuple ) = energy;
    }
};

extern "C" void CalculateDualEnergy2D( Mojo::Core::SegmenterState* segmenterState, float lambda, int zSlice, float& dualEnergy )
{
    int numElementsPerSlice = segmenterState->volumeDescription.numVoxels.x * segmenterState->volumeDescription.numVoxels.y;

    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( numElementsPerSlice * zSlice ),
                        Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin() + ( numElementsPerSlice * ( zSlice + 0 ) ),
                    segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin() + ( numElementsPerSlice * ( zSlice + 0 ) ) ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( numElementsPerSlice * ( zSlice + 1 ) ),
                        Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin() + ( numElementsPerSlice * ( zSlice + 1 ) ),
                    segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin() + ( numElementsPerSlice * ( zSlice + 1 ) ) ) ),
                               
            DualEnergy2DFunction(
                thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float4 >( "DualMap" )[ 0 ] ),
                segmenterState->volumeDescription.numVoxels,
                lambda ) ) );

    // use CostBackwardMap as a scratchpad for the reduction, since we are already using the ScratchpadMap as the data source
    dualEnergy = Mojo::Core::Thrust::Reduce(
        segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin() + ( numElementsPerSlice * ( zSlice + 0 ) ),
        segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin() + ( numElementsPerSlice * ( zSlice + 1 ) ),
        0.0f,
        segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ) );
};


struct PrimalEnergy2DFunction
{
    enum TupleLayout
    {
        INDEX_3D,
        CONSTRAINT_MAP,
        EDGE_XY_MAP,
        SCRATCHPAD_MAP
    };

    float* mPrimalMapBuffer;
    int3   mNumVoxels;
    float  mLambda;

    PrimalEnergy2DFunction( float* primalMapBuffer, int3 numVoxels, float lambda ) : mPrimalMapBuffer( primalMapBuffer ), mNumVoxels( numVoxels ), mLambda( lambda ) {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int3   index3D                          = thrust::get< INDEX_3D >( tuple );
        float  constraint                       = thrust::get< CONSTRAINT_MAP >( tuple );
        float  edgeXY                           = thrust::get< EDGE_XY_MAP >( tuple );

        int3   index3DForwardX                  = make_int3( min( mNumVoxels.x - 1, index3D.x + 1 ), index3D.y,                              index3D.z );
        int3   index3DForwardY                  = make_int3( index3D.x,                              min( mNumVoxels.y - 1, index3D.y + 1 ), index3D.z );
                                              
        float  primal                           = mPrimalMapBuffer[ Index3DToIndex1D( index3D,         mNumVoxels ) ];
        float  primalForwardX                   = mPrimalMapBuffer[ Index3DToIndex1D( index3DForwardX, mNumVoxels ) ];
        float  primalForwardY                   = mPrimalMapBuffer[ Index3DToIndex1D( index3DForwardY, mNumVoxels ) ];
                    
        float2 primalGradientXY                 = make_float2( primalForwardX - primal, primalForwardY - primal );
        float  primalGradientXYLength           = sqrt( primalGradientXY.x * primalGradientXY.x + primalGradientXY.y * primalGradientXY.y );

        float  energy                           = ( edgeXY * primalGradientXYLength ) + ( mLambda * constraint * primal );

        thrust::get< SCRATCHPAD_MAP >( tuple )  = energy;
    }
};

extern "C" void CalculatePrimalEnergy2D( Mojo::Core::SegmenterState* segmenterState, float lambda, int zSlice, float& primalEnergy )
{
    int numElementsPerSlice = segmenterState->volumeDescription.numVoxels.x * segmenterState->volumeDescription.numVoxels.y;

    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( numElementsPerSlice * zSlice ),
                        Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin() + ( numElementsPerSlice * ( zSlice + 0 ) ),
                    segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).begin()     + ( numElementsPerSlice * ( zSlice + 0 ) ),
                    segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin() + ( numElementsPerSlice * ( zSlice + 0 ) ) ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( numElementsPerSlice * ( zSlice + 1 ) ),
                        Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin() + ( numElementsPerSlice * ( zSlice + 1 ) ),
                    segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).begin()     + ( numElementsPerSlice * ( zSlice + 1 ) ),
                    segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin() + ( numElementsPerSlice * ( zSlice + 1 ) ) ) ),
                               
            PrimalEnergy2DFunction(
                thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float >( "PrimalMap" )[ 0 ] ),
                segmenterState->volumeDescription.numVoxels,
                lambda ) ) );

    // use CostBackwardMap as a scratchpad for the reduction, since we are already using the ScratchpadMap as the data source
    primalEnergy = Mojo::Core::Thrust::Reduce(
        segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin() + ( numElementsPerSlice * ( zSlice + 0 ) ),
        segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin() + ( numElementsPerSlice * ( zSlice + 1 ) ),
        0.0f,
        segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ) );
}