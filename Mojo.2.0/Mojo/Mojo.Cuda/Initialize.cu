#include "Mojo.Core/D3D11.hpp"
#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/Thrust.hpp"
#include "Mojo.Core/SegmenterState.hpp"
#include "Mojo.Core/ID3D11CudaTexture.hpp"
#include "Mojo.Core/ForEach.hpp"

#include "Index.cuh"

extern "C" void InitializeCommittedSegmentation( Mojo::Core::SegmenterState* segmenterState )
{
    MOJO_THRUST_SAFE(
        thrust::fill(
            segmenterState->deviceVectors.Get< uchar4 >( "ColorMap" ).begin(),
            segmenterState->deviceVectors.Get< uchar4 >( "ColorMap" ).end(),
            segmenterState->constParameters.Get< uchar4 >( "COLOR_MAP_INITIAL_VALUE" ) ) );

    MOJO_THRUST_SAFE(
        thrust::fill(
            segmenterState->deviceVectors.Get< int >( "IdMap" ).begin(),
            segmenterState->deviceVectors.Get< int >( "IdMap" ).end(),
            segmenterState->constParameters.Get< int >( "ID_MAP_INITIAL_VALUE" ) ) );
}

extern "C" void InitializeSegmentation( Mojo::Core::SegmenterState* segmenterState )
{   
    MOJO_THRUST_SAFE(
        thrust::fill(
            segmenterState->deviceVectors.Get< float >( "PrimalMap" ).begin(),
            segmenterState->deviceVectors.Get< float >( "PrimalMap" ).end(),
            segmenterState->constParameters.Get< float >( "PRIMAL_MAP_INITIAL_VALUE" ) ) );

    MOJO_THRUST_SAFE(
        thrust::fill(
            segmenterState->deviceVectors.Get< float >( "OldPrimalMap" ).begin(),
            segmenterState->deviceVectors.Get< float >( "OldPrimalMap" ).end(),
            segmenterState->constParameters.Get< float >( "PRIMAL_MAP_INITIAL_VALUE" ) ) );

    MOJO_THRUST_SAFE(
        thrust::fill(
            segmenterState->deviceVectors.Get< float4 >( "DualMap" ).begin(),
            segmenterState->deviceVectors.Get< float4 >( "DualMap" ).end(),
            segmenterState->constParameters.Get< float4 >( "DUAL_MAP_INITIAL_VALUE" ) ) );
}

extern "C" void InitializeConstraintMap( Mojo::Core::SegmenterState* segmenterState )
{
    MOJO_THRUST_SAFE(
        thrust::fill(
            segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin(),
            segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).end(),
            segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_INITIAL_VALUE" ) ) );
}

extern "C" void InitializeScratchpad( Mojo::Core::SegmenterState* segmenterState )
{
    MOJO_THRUST_SAFE(
        thrust::fill(
            segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).begin(),
            segmenterState->deviceVectors.Get< float >( "ScratchpadMap" ).end(),
            segmenterState->constParameters.Get< float >( "SCRATCHPAD_MAP_INITIAL_VALUE" ) ) );
}

extern "C" void InitializeCostMap( Mojo::Core::SegmenterState* segmenterState )
{
    MOJO_THRUST_SAFE(
        thrust::fill(
            segmenterState->deviceVectors.Get< float >( "CostForwardMap" ).begin(),
            segmenterState->deviceVectors.Get< float >( "CostForwardMap" ).end(),
            segmenterState->constParameters.Get< float >( "COST_MAP_INITIAL_VALUE" ) ) );

    MOJO_THRUST_SAFE(
        thrust::fill(
            segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ).begin(),
            segmenterState->deviceVectors.Get< float >( "CostBackwardMap" ).end(),
            segmenterState->constParameters.Get< float >( "COST_MAP_INITIAL_VALUE" ) ) );
}