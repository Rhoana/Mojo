#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/Thrust.hpp"
#include "Mojo.Core/SegmenterState.hpp"
#include "Mojo.Core/ID3D11CudaTexture.hpp"

#include "Index.cuh"
#include "Math.cuh"

struct UpdateDualMap3DFunction
{
    enum TupleLayout
    {
        INDEX_3D,
        EDGE_XY_MAP,
        EDGE_Z_MAP,
        DUAL_MAP
    };

    float* mOldPrimalMapBuffer;
    int3   mNumVoxels;
    float  mSigma;
    float  mEdgeStrengthZ;

    UpdateDualMap3DFunction( float* oldPrimalMapBuffer, int3 numVoxels, float sigma, float edgeStrengthZ ) : mOldPrimalMapBuffer( oldPrimalMapBuffer ), mNumVoxels( numVoxels ), mSigma( sigma ), mEdgeStrengthZ( edgeStrengthZ ) {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int3   index3D                   = thrust::get< INDEX_3D >( tuple );
        float  edgeXY                    = thrust::get< EDGE_XY_MAP >( tuple );
        float  edgeZ                     = thrust::get< EDGE_Z_MAP >( tuple ) * mEdgeStrengthZ;
        float4 dual                      = thrust::get< DUAL_MAP >( tuple );

        int3   index3DForwardX           = make_int3( min( mNumVoxels.x - 1, index3D.x + 1 ), index3D.y,                              index3D.z );
        int3   index3DForwardY           = make_int3( index3D.x,                              min( mNumVoxels.y - 1, index3D.y + 1 ), index3D.z );
        int3   index3DForwardZ           = make_int3( index3D.x,                              index3D.y,                              min( mNumVoxels.z - 1, index3D.z + 1 ) );

        float  oldPrimal                 = mOldPrimalMapBuffer[ Index3DToIndex1D( index3D,         mNumVoxels ) ];
        float  oldPrimalForwardX         = mOldPrimalMapBuffer[ Index3DToIndex1D( index3DForwardX, mNumVoxels ) ];
        float  oldPrimalForwardY         = mOldPrimalMapBuffer[ Index3DToIndex1D( index3DForwardY, mNumVoxels ) ];
        float  oldPrimalForwardZ         = mOldPrimalMapBuffer[ Index3DToIndex1D( index3DForwardZ, mNumVoxels ) ];

        float2 oldPrimalGradientXY       = make_float2( oldPrimalForwardX - oldPrimal, oldPrimalForwardY - oldPrimal );
        float  oldPrimalGradientZ        = oldPrimalForwardZ - oldPrimal;

        float2 dualXY                    = make_float2( dual.x, dual.y ) + ( mSigma * oldPrimalGradientXY );
        float  dualXYLength              = sqrt( ( dualXY.x * dualXY.x ) + ( dualXY.y * dualXY.y ) );
        float2 dualXYProjected           = dualXY / max( 1.0f, dualXYLength / edgeXY );

        float  dualZ                     = dual.z + ( mSigma * oldPrimalGradientZ );
        float  dualZProjected            = clamp( dualZ, - edgeZ, edgeZ );

        thrust::get< DUAL_MAP >( tuple ) = make_float4( dualXYProjected.x, dualXYProjected.y, dualZProjected, 0.0f );
    }
};

extern "C" void UpdateDualMap3D( Mojo::Core::SegmenterState* segmenterState, float sigma )
{
    int numElements = segmenterState->volumeDescription.numVoxels.x * segmenterState->volumeDescription.numVoxels.y * segmenterState->volumeDescription.numVoxels.z;

    if ( !segmenterState->constParameters.Get< bool >( "DIRECT_ANISOTROPIC_TV" ) )
    {
        MOJO_THRUST_SAFE(
            thrust::for_each(
                thrust::make_zip_iterator(
                    thrust::make_tuple( 
                        thrust::make_transform_iterator(
                            thrust::make_counting_iterator( 0 ),
                            Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                        segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).begin(),
                        thrust::make_constant_iterator( 1.0f ),
                        segmenterState->deviceVectors.Get< float4 >( "DualMap" ).begin() ) ),

                thrust::make_zip_iterator(
                    thrust::make_tuple( 
                        thrust::make_transform_iterator(
                            thrust::make_counting_iterator( numElements ),
                            Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                        segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).end(),
                        thrust::make_constant_iterator( 1.0f ),
                        segmenterState->deviceVectors.Get< float4 >( "DualMap" ).end() ) ),
              
                UpdateDualMap3DFunction(
                    thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float >( "OldPrimalMap" )[ 0 ] ),
                    segmenterState->volumeDescription.numVoxels,
                    sigma,
                    segmenterState->constParameters.Get< float >( "EDGE_STRENGTH_Z" ) ) ) );
    }
    else
    {
        MOJO_THRUST_SAFE(
            thrust::for_each(
                thrust::make_zip_iterator(
                    thrust::make_tuple( 
                        thrust::make_transform_iterator(
                            thrust::make_counting_iterator( 0 ),
                            Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                        segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).begin(),
                        segmenterState->deviceVectors.Get< float >( "EdgeZMap" ).begin(),
                        segmenterState->deviceVectors.Get< float4 >( "DualMap" ).begin() ) ),

                thrust::make_zip_iterator(
                    thrust::make_tuple( 
                        thrust::make_transform_iterator(
                            thrust::make_counting_iterator( numElements ),
                            Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                        segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).end(),
                        segmenterState->deviceVectors.Get< float >( "EdgeZMap" ).begin(),
                        segmenterState->deviceVectors.Get< float4 >( "DualMap" ).end() ) ),
              
                UpdateDualMap3DFunction(
                    thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float >( "OldPrimalMap" )[ 0 ] ),
                    segmenterState->volumeDescription.numVoxels,
                    sigma,
                    segmenterState->constParameters.Get< float >( "EDGE_STRENGTH_Z" ) ) ) );
    }
};



struct UpdatePrimalMap3DFunction
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

    UpdatePrimalMap3DFunction( float4* dualMapBuffer, int3 numVoxels, float lambda, float tau ) : mDualMapBuffer( dualMapBuffer ), mNumVoxels( numVoxels ), mLambda( lambda ), mTau( tau ) {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int3   index3D                         = thrust::get< INDEX_3D >( tuple );
        float  constraint                      = thrust::get< CONSTRAINT_MAP >( tuple );
        float  primal                          = thrust::get< PRIMAL_MAP >( tuple );
        float  oldPrimal                       = primal;

        int3   index3DBackwardX                = make_int3( max( 0, index3D.x - 1 ), index3D.y,               index3D.z );
        int3   index3DBackwardY                = make_int3( index3D.x,               max( 0, index3D.y - 1 ), index3D.z );
        int3   index3DBackwardZ                = make_int3( index3D.x,               index3D.y,               max( 0, index3D.z - 1 ) );

        float4 dual                            = mDualMapBuffer[ Index3DToIndex1D( index3D,          mNumVoxels ) ];
        float4 dualBackwardX                   = mDualMapBuffer[ Index3DToIndex1D( index3DBackwardX, mNumVoxels ) ];
        float4 dualBackwardY                   = mDualMapBuffer[ Index3DToIndex1D( index3DBackwardY, mNumVoxels ) ];
        float4 dualBackwardZ                   = mDualMapBuffer[ Index3DToIndex1D( index3DBackwardZ, mNumVoxels ) ];

        if ( index3D.x == 0 )
            dualBackwardX.x = 0.0f;
        else if ( index3D.x >= mNumVoxels.x )
            dual.x = 0.0f;

        if ( index3D.y == 0 )
            dualBackwardY.y = 0.0f;
        else if ( index3D.y >= mNumVoxels.y )
            dual.y = 0.0f;

        if ( index3D.z == 0 )
            dualBackwardZ.z = 0.0f;
        else if ( index3D.z >= mNumVoxels.z )
            dual.z = 0.0f;

        float  dualDivergence                  = dual.x - dualBackwardX.x + dual.y - dualBackwardY.y + dual.z - dualBackwardZ.z;
        primal                                 = clamp( primal + mTau * ( dualDivergence - constraint * mLambda ), 0.0f, 1.0f );

        thrust::get< PRIMAL_MAP >( tuple )     = primal;
        thrust::get< OLD_PRIMAL_MAP >( tuple ) = 2.0f * primal - oldPrimal;
    }
};

extern "C" void UpdatePrimalMap3D( Mojo::Core::SegmenterState* segmenterState, float lambda, float tau )
{
    int numElements = segmenterState->volumeDescription.numVoxels.x * segmenterState->volumeDescription.numVoxels.y * segmenterState->volumeDescription.numVoxels.z;

    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( 0 ),
                        Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin(),
                    segmenterState->deviceVectors.Get< float >( "PrimalMap" ).begin(),
                    segmenterState->deviceVectors.Get< float >( "OldPrimalMap" ).begin() ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( numElements ),
                        Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).end(),
                    segmenterState->deviceVectors.Get< float >( "PrimalMap" ).end(),
                    segmenterState->deviceVectors.Get< float >( "OldPrimalMap" ).end() ) ),
              
            UpdatePrimalMap3DFunction(
                thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float4 >( "DualMap" )[ 0 ] ),
                segmenterState->volumeDescription.numVoxels,
                lambda,
                tau ) ) );
};

struct DualEnergy3DFunction
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

    DualEnergy3DFunction( float4* dualMapBuffer, int3 numVoxels, float lambda ) : mDualMapBuffer( dualMapBuffer ), mNumVoxels( numVoxels ), mLambda( lambda ) {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int3   index3D                          = thrust::get< INDEX_3D >( tuple );
        float  constraint                       = thrust::get< CONSTRAINT_MAP >( tuple );
                                                
        int3   index3DBackwardX                 = make_int3( max( 0, index3D.x - 1 ), index3D.y,               index3D.z );
        int3   index3DBackwardY                 = make_int3( index3D.x,               max( 0, index3D.y - 1 ), index3D.z );
        int3   index3DBackwardZ                 = make_int3( index3D.x,               index3D.y,               max( 0, index3D.z - 1 ) );
                                                
        float4 dual                             = mDualMapBuffer[ Index3DToIndex1D( index3D,          mNumVoxels ) ];
        float4 dualBackwardX                    = mDualMapBuffer[ Index3DToIndex1D( index3DBackwardX, mNumVoxels ) ];
        float4 dualBackwardY                    = mDualMapBuffer[ Index3DToIndex1D( index3DBackwardY, mNumVoxels ) ];
        float4 dualBackwardZ                    = mDualMapBuffer[ Index3DToIndex1D( index3DBackwardZ, mNumVoxels ) ];

        if ( index3D.x == 0 )
            dualBackwardX.x = 0.0f;
        else if ( index3D.x >= mNumVoxels.x )
            dual.x = 0.0f;

        if ( index3D.y == 0 )
            dualBackwardY.y = 0.0f;
        else if ( index3D.y >= mNumVoxels.y )
            dual.y = 0.0f;

        if ( index3D.z == 0 )
            dualBackwardZ.z = 0.0f;
        else if ( index3D.z >= mNumVoxels.z )
            dual.z = 0.0f;

        float  dualDivergence                   = dual.x - dualBackwardX.x + dual.y - dualBackwardY.y + dual.z - dualBackwardZ.z;
        float  energy                           = ( -dualDivergence ) + ( mLambda * constraint );

        if ( energy > 0.0f )
            energy = 0.0f;

        thrust::get< SCRATCHPAD_MAP >( tuple ) = energy;
    }
};

extern "C" void CalculateDualEnergy3D( Mojo::Core::SegmenterState* segmenterState, float lambda, float& dualEnergy )
{
    int numElements = segmenterState->volumeDescription.numVoxels.x * segmenterState->volumeDescription.numVoxels.y * segmenterState->volumeDescription.numVoxels.z;

    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( 0 ),
                        Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin(),
                    segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin() ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( numElements ),
                        Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).end(),
                    segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).end() ) ),
                               
            DualEnergy3DFunction(
                thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float4 >( "DualMap" )[ 0 ] ),
                segmenterState->volumeDescription.numVoxels,
                lambda ) ) );

    // use CostBackwardMap as a scratchpad for the reduction, since we are already using the ScratchpadMap as the data source
    dualEnergy = Mojo::Core::Thrust::Reduce(
        segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin(),
        segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).end(),
        0.0f,
        segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ) );
};


struct PrimalEnergy3DFunction
{
    enum TupleLayout
    {
        INDEX_3D,
        CONSTRAINT_MAP,
        EDGE_XY_MAP,
        EDGE_Z_MAP,
        SCRATCHPAD_MAP
    };

    float* mPrimalMapBuffer;
    int3   mNumVoxels;
    float  mLambda;
    float  mEdgeStrengthZ;

    PrimalEnergy3DFunction( float* primalMapBuffer, int3 numVoxels, float lambda, float edgeStrengthZ ) : mPrimalMapBuffer( primalMapBuffer ), mNumVoxels( numVoxels ), mLambda( lambda ), mEdgeStrengthZ( edgeStrengthZ ) {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int3   index3D                          = thrust::get< INDEX_3D >( tuple );
        float  constraint                       = thrust::get< CONSTRAINT_MAP >( tuple );
        float  edgeXY                           = thrust::get< EDGE_XY_MAP >( tuple );
        float  edgeZ                            = thrust::get< EDGE_Z_MAP >( tuple ) * mEdgeStrengthZ;

        int3   index3DForwardX                  = make_int3( min( mNumVoxels.x - 1, index3D.x + 1 ), index3D.y,                              index3D.z );
        int3   index3DForwardY                  = make_int3( index3D.x,                              min( mNumVoxels.y - 1, index3D.y + 1 ), index3D.z );
        int3   index3DForwardZ                  = make_int3( index3D.x,                              index3D.y,                              min( mNumVoxels.z - 1, index3D.z + 1 ) );
                                              
        float  primal                           = mPrimalMapBuffer[ Index3DToIndex1D( index3D,         mNumVoxels ) ];
        float  primalForwardX                   = mPrimalMapBuffer[ Index3DToIndex1D( index3DForwardX, mNumVoxels ) ];
        float  primalForwardY                   = mPrimalMapBuffer[ Index3DToIndex1D( index3DForwardY, mNumVoxels ) ];
        float  primalForwardZ                   = mPrimalMapBuffer[ Index3DToIndex1D( index3DForwardZ, mNumVoxels ) ];
                    
        float2 primalGradientXY                 = make_float2( primalForwardX - primal, primalForwardY - primal );
        float  primalGradientXYLength           = sqrt( primalGradientXY.x * primalGradientXY.x + primalGradientXY.y * primalGradientXY.y );

        float  primalGradientZ                  = primalForwardZ - primal;
        float  primalGradientZLength            = abs( primalGradientZ );

        float  energy                           = ( edgeXY * primalGradientXYLength ) + ( edgeZ * primalGradientZLength ) + ( mLambda * constraint * primal );

        thrust::get< SCRATCHPAD_MAP >( tuple )  = energy;
    }
};

extern "C" void CalculatePrimalEnergy3D( Mojo::Core::SegmenterState* segmenterState, float lambda, float& primalEnergy )
{
    int numElements = segmenterState->volumeDescription.numVoxels.x * segmenterState->volumeDescription.numVoxels.y * segmenterState->volumeDescription.numVoxels.z;

    if ( !segmenterState->constParameters.Get< bool >( "DIRECT_ANISOTROPIC_TV" ) )
    {
        MOJO_THRUST_SAFE(
            thrust::for_each(
                thrust::make_zip_iterator(
                    thrust::make_tuple(
                        thrust::make_transform_iterator(
                            thrust::make_counting_iterator( 0 ),
                            Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                        segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin(),
                        segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).begin(),
                        thrust::make_constant_iterator( 1.0f ),
                        segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin() ) ),

                thrust::make_zip_iterator(
                    thrust::make_tuple( 
                        thrust::make_transform_iterator(
                            thrust::make_counting_iterator( numElements ),
                            Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                        segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).end(),
                        segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).end(),
                        thrust::make_constant_iterator( 1.0f ),
                        segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).end() ) ),
                               
                PrimalEnergy3DFunction(
                    thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float >( "PrimalMap" )[ 0 ] ),
                    segmenterState->volumeDescription.numVoxels,
                    lambda,
                    segmenterState->constParameters.Get< float >( "EDGE_STRENGTH_Z" ) ) ) );
    }
    else
    {
        MOJO_THRUST_SAFE(
            thrust::for_each(
                thrust::make_zip_iterator(
                    thrust::make_tuple(
                        thrust::make_transform_iterator(
                            thrust::make_counting_iterator( 0 ),
                            Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                        segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin(),
                        segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).begin(),
                        segmenterState->deviceVectors.Get< float >( "EdgeZMap" ).begin(),
                        segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin() ) ),

                thrust::make_zip_iterator(
                    thrust::make_tuple( 
                        thrust::make_transform_iterator(
                            thrust::make_counting_iterator( numElements ),
                            Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                        segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).end(),
                        segmenterState->deviceVectors.Get< float >( "EdgeXYMap" ).end(),
                        segmenterState->deviceVectors.Get< float >( "EdgeZMap" ).end(),
                        segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).end() ) ),
                               
                PrimalEnergy3DFunction(
                    thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float >( "PrimalMap" )[ 0 ] ),
                    segmenterState->volumeDescription.numVoxels,
                    lambda,
                    segmenterState->constParameters.Get< float >( "EDGE_STRENGTH_Z" ) ) ) );
    }

    // use CostBackwardMap as a scratchpad for the reduction, since we are already using the ScratchpadMap as the data source
    primalEnergy = Mojo::Core::Thrust::Reduce(
        segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin(),
        segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).end(),
        0.0f,
        segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ) );
}
