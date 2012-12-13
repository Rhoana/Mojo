#include "Segmenter.hpp"

#include "Mojo.Core/Stl.hpp"

#include "Mojo.Core/Assert.hpp"
#include "Mojo.Core/D3D11.hpp"
#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/Thrust.hpp"
#include "Mojo.Core/SegmenterState.hpp"
#include "Mojo.Core/D3D11CudaTexture.hpp"
#include "Mojo.Core/ForEach.hpp"
#include "Mojo.Core/Printf.hpp"
#include "Mojo.Core/Index.hpp"

#include "Mojo.Cuda/Mojo.Cuda.hpp"

namespace Mojo
{
namespace Native
{

Segmenter::Segmenter( ID3D11Device* d3d11Device, ID3D11DeviceContext* d3d11DeviceContext, Core::PrimitiveMap constParameters )
{
    mSegmenterState.constParameters = constParameters;

    mSegmenterState.dynamicParameters.Set( "ConvergenceGap",         0.0f );
    mSegmenterState.dynamicParameters.Set( "ConvergenceGapDelta",    0.0f );
    mSegmenterState.dynamicParameters.Set( "MaxForegroundCostDelta", mSegmenterState.constParameters.Get< float >( "COST_MAP_INITIAL_MAX_FOREGROUND_COST_DELTA" ) );
    mSegmenterState.dynamicParameters.Set( "IsVolumeLoaded",         false );

    mSegmenterState.d3d11Device = d3d11Device;
    mSegmenterState.d3d11Device->AddRef();

    mSegmenterState.d3d11DeviceContext = d3d11DeviceContext;
    mSegmenterState.d3d11DeviceContext->AddRef();
}

Segmenter::~Segmenter()
{
    mSegmenterState.d3d11DeviceContext->Release();
    mSegmenterState.d3d11DeviceContext = NULL;

    mSegmenterState.d3d11Device->Release();
    mSegmenterState.d3d11Device = NULL;
}

void Segmenter::LoadVolume( Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions )
{
    switch ( volumeDescriptions.Get( "SourceMap" ).dxgiFormat )
    {
        case DXGI_FORMAT_R8_UNORM:
            LoadVolumeInternal< uchar1 >( volumeDescriptions );
            break;

        case DXGI_FORMAT_R8G8B8A8_UNORM:
            LoadVolumeInternal< uchar4 >( volumeDescriptions );
            break;

        default:
            RELEASE_ASSERT( 0 );
            break;
    }
}

void Segmenter::UnloadVolume()
{
    UnloadVolumeInternal();
}

void Segmenter::LoadSegmentation( Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions )
{
    InitializeSegmentation();
    InitializeConstraintMap();
    RedoLastChangeToCommittedSegmentation();

    int numVoxels = mSegmenterState.volumeDescription.numVoxels.x * mSegmenterState.volumeDescription.numVoxels.y * mSegmenterState.volumeDescription.numVoxels.z;

    mSegmenterState.hostVectors.Get< int    >( "UndoIdMap"    ) = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "UndoColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );

    Core::Thrust::MemcpyHostToDevice( mSegmenterState.deviceVectors.Get< int    >( "IdMap"    ), volumeDescriptions.Get( "IdMap"    ).data, numVoxels );
    Core::Thrust::MemcpyHostToDevice( mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" ), volumeDescriptions.Get( "ColorMap" ).data, numVoxels );

    mSegmenterState.hostVectors.Get< int    >( "RedoIdMap"    ) = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "RedoColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );
}

void Segmenter::SaveSegmentationAs( Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions )
{
    int numVoxels = mSegmenterState.volumeDescription.numVoxels.x * mSegmenterState.volumeDescription.numVoxels.y * mSegmenterState.volumeDescription.numVoxels.z;

    Core::Thrust::MemcpyDeviceToHost( volumeDescriptions.Get( "IdMap"    ).data, mSegmenterState.deviceVectors.Get< int    >( "IdMap"    ), numVoxels );
    Core::Thrust::MemcpyDeviceToHost( volumeDescriptions.Get( "ColorMap" ).data, mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" ), numVoxels );
}

void Segmenter::InitializeEdgeXYMap( Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions )
{
    ::InitializeEdgeXYMap( &mSegmenterState, &volumeDescriptions );
}

void Segmenter::InitializeEdgeXYMapForSplitting( Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions, int segmentationLabelId )
{
    ::InitializeEdgeXYMapForSplitting( &mSegmenterState, &volumeDescriptions, segmentationLabelId );
}

void Segmenter::InitializeConstraintMap()
{
    ::InitializeConstraintMap( &mSegmenterState );

    mSegmenterState.slicesWithForegroundConstraints.clear();
    mSegmenterState.slicesWithBackgroundConstraints.clear();
}

void Segmenter::InitializeConstraintMapFromIdMap( int segmentationLabelId )
{
    ::InitializeConstraintMapFromIdMap( &mSegmenterState, segmentationLabelId );
}

void Segmenter::InitializeConstraintMapFromIdMapForSplitting( int segmentationLabelId )
{
    ::InitializeConstraintMapFromIdMapForSplitting( &mSegmenterState, segmentationLabelId );
}

void Segmenter::InitializeConstraintMapFromPrimalMap()
{
    ::InitializeConstraintMapFromPrimalMap( &mSegmenterState );
}

void Segmenter::DilateConstraintMap()
{
    ::DilateConstraintMap( &mSegmenterState );
}

void Segmenter::InitializeSegmentation()
{
    ::InitializeSegmentation( &mSegmenterState );
}

void Segmenter::InitializeSegmentationAndRemoveFromCommittedSegmentation( int segmentationLabelId )
{
    ::InitializeSegmentationAndRemoveFromCommittedSegmentation( &mSegmenterState, segmentationLabelId );
}

void Segmenter::AddForegroundHardConstraint( int3 p, float radius )
{
    AddHardConstraint( p,
                       p,
                       radius,
                       mSegmenterState.constParameters.Get< float >( "CONSTRAINT_MAP_HARD_FOREGROUND_USER" ),
                       mSegmenterState.constParameters.Get< float >( "PRIMAL_MAP_FOREGROUND" ) );

    mSegmenterState.slicesWithForegroundConstraints.insert( p.z );
}

void Segmenter::AddBackgroundHardConstraint( int3 p, float radius )
{
    AddHardConstraint( p,
                       p,
                       radius,
                       mSegmenterState.constParameters.Get< float >( "CONSTRAINT_MAP_HARD_BACKGROUND_USER" ),
                       mSegmenterState.constParameters.Get< float >( "PRIMAL_MAP_BACKGROUND" ) );

    mSegmenterState.slicesWithBackgroundConstraints.insert( p.z );
}

void Segmenter::AddForegroundHardConstraint( int3 p1, int3 p2, float radius )
{
    AddHardConstraint( p1,
                       p2,
                       radius,
                       mSegmenterState.constParameters.Get< float >( "CONSTRAINT_MAP_HARD_FOREGROUND_USER" ),
                       mSegmenterState.constParameters.Get< float >( "PRIMAL_MAP_FOREGROUND" ) );

    mSegmenterState.slicesWithForegroundConstraints.insert( p1.z );
}

void Segmenter::AddBackgroundHardConstraint( int3 p1, int3 p2, float radius )
{
    AddHardConstraint( p1,
                       p2,
                       radius,
                       mSegmenterState.constParameters.Get< float >( "CONSTRAINT_MAP_HARD_BACKGROUND_USER" ),
                       mSegmenterState.constParameters.Get< float >( "PRIMAL_MAP_BACKGROUND" ) );

    mSegmenterState.slicesWithBackgroundConstraints.insert( p1.z );
}

bool Segmenter::AddHardConstraint( int3 p1, int3 p2, float radius, float constraintValue, float primalValue )
{
    RELEASE_ASSERT( p1.z == p2.z );

    if ( p1.x < 0 || p1.y < 0 || p2.x < 0 || p2.y < 0 )
    {
        return false;
    }

    ::AddHardConstraint( &mSegmenterState, p1, p2, radius, constraintValue, primalValue );

    return true;
}

void Segmenter::Update2D( int numIterations, int zSlice )
{
#ifdef _DEBUG
    numIterations = 1;
#endif

    int   checkEnergyFrequency = numIterations - 1;
    float lambda               = 0.001f;
    float convergenceGap       = -1.0f;
    float L                    = 9.0f;
    float tau                  = 1.0f / sqrt( L );
    float sigma                = 1.0f / sqrt( L );

    // Start calculation
    for ( int i = 0; i < numIterations; i++ )
    {
        ::UpdatePrimalMap2D( &mSegmenterState, lambda, tau, zSlice );
        ::UpdateDualMap2D( &mSegmenterState, sigma, zSlice );

        if( ( i == 0 && checkEnergyFrequency == 0 ) || ( i % checkEnergyFrequency == 0 ) )
        {
            float primalEnergy = -1.0f;
            float dualEnergy   = -1.0f;

            ::CalculateDualEnergy2D( &mSegmenterState, lambda, zSlice, dualEnergy );
            ::CalculatePrimalEnergy2D( &mSegmenterState, lambda, zSlice, primalEnergy );

            float oldconvergenceGap             = mSegmenterState.dynamicParameters.Get< float >( "ConvergenceGap" );
            float newconvergenceGap             = primalEnergy - dualEnergy;
            float convergenceGapDelta           = abs( newconvergenceGap - oldconvergenceGap ) / (float)numIterations;

            mSegmenterState.dynamicParameters.Set( "ConvergenceGap",      newconvergenceGap );
            mSegmenterState.dynamicParameters.Set( "ConvergenceGapDelta", convergenceGapDelta );
        }
    }
}

void Segmenter::Update3D( int numIterations )
{
#ifdef _DEBUG
    numIterations = 1;
#endif

    int   checkEnergyFrequency = numIterations - 1;
    float lambda               = 0.001f;
    float convergenceGap       = -1.0f;
    float L                    = 9.0f;
    float tau                  = 1.0f / sqrt( L );
    float sigma                = 1.0f / sqrt( L );

    // Start calculation
    for ( int i = 0; i < numIterations; i++ )
    {
        ::UpdatePrimalMap3D( &mSegmenterState, lambda, tau );
        ::UpdateDualMap3D( &mSegmenterState, sigma );

        if( ( i == 0 && checkEnergyFrequency == 0 ) || ( i % checkEnergyFrequency == 0 ) )
        {
            float primalEnergy = -1.0f;
            float dualEnergy   = -1.0f;

            ::CalculateDualEnergy3D( &mSegmenterState, lambda, dualEnergy );
            ::CalculatePrimalEnergy3D( &mSegmenterState, lambda, primalEnergy );
                                    
            float oldconvergenceGap             = mSegmenterState.dynamicParameters.Get< float >( "ConvergenceGap" );
            float newconvergenceGap             = primalEnergy - dualEnergy;
            float convergenceGapDelta           = abs( newconvergenceGap - oldconvergenceGap ) / (float)numIterations;

            mSegmenterState.dynamicParameters.Set( "ConvergenceGap",      newconvergenceGap );
            mSegmenterState.dynamicParameters.Set( "ConvergenceGapDelta", convergenceGapDelta );
        }
    }
}

void Segmenter::VisualUpdate()
{
    cudaArray* cudaArray = NULL;
    mSegmenterState.d3d11CudaTextures.MapCudaArrays();

    cudaArray = mSegmenterState.d3d11CudaTextures.Get( "PrimalMap" )->GetMappedCudaArray();
    Core::Thrust::Memcpy3DToArray( cudaArray, mSegmenterState.deviceVectors.Get< float >( "PrimalMap" ), mSegmenterState.volumeDescription.numVoxels );

    cudaArray = mSegmenterState.d3d11CudaTextures.Get( "ConstraintMap" )->GetMappedCudaArray();
    Core::Thrust::Memcpy3DToArray( cudaArray, mSegmenterState.deviceVectors.Get< float >( "ConstraintMap" ), mSegmenterState.volumeDescription.numVoxels );

    mSegmenterState.d3d11CudaTextures.UnmapCudaArrays();
}

void Segmenter::UpdateCommittedSegmentation( int segmentationLabelId, int4 segmentationLabelColor )
{
    RedoLastChangeToCommittedSegmentation();

    uchar4 segmentationLabelColorUChar4 = make_uchar4( segmentationLabelColor.x, segmentationLabelColor.y, segmentationLabelColor.z, segmentationLabelColor.w );

    mSegmenterState.hostVectors.Get< int    >( "UndoIdMap"    ) = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "UndoColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );

    ::UpdateCommittedSegmentation( &mSegmenterState, segmentationLabelId, segmentationLabelColorUChar4 );

    mSegmenterState.hostVectors.Get< int    >( "RedoIdMap"    ) = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "RedoColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );
}

void Segmenter::UpdateCommittedSegmentationDoNotRemove( int segmentationLabelId, int4 segmentationLabelColor )
{
    RedoLastChangeToCommittedSegmentation();

    uchar4 segmentationLabelColorUChar4 = make_uchar4( segmentationLabelColor.x, segmentationLabelColor.y, segmentationLabelColor.z, segmentationLabelColor.w );

    mSegmenterState.hostVectors.Get< int    >( "UndoIdMap"    ) = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "UndoColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );

    ::UpdateCommittedSegmentationDoNotRemove( &mSegmenterState, segmentationLabelId, segmentationLabelColorUChar4 );

    mSegmenterState.hostVectors.Get< int    >( "RedoIdMap"    ) = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "RedoColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );
}

void Segmenter::ReplaceSegmentationLabelInCommittedSegmentation2D( int oldId, int newId, int4 newColor, int slice )
{
    uchar4 newColorUChar4 = make_uchar4( newColor.x, newColor.y, newColor.z, newColor.w );

    mSegmenterState.hostVectors.Get< int    >( "UndoIdMap"    ) = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "UndoColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );

    ::ReplaceSegmentationLabelInCommittedSegmentation2D( &mSegmenterState, oldId, newId, newColorUChar4, slice );

    mSegmenterState.hostVectors.Get< int    >( "RedoIdMap"    ) = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "RedoColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );
}

void Segmenter::ReplaceSegmentationLabelInCommittedSegmentation3D( int oldId, int newId, int4 newColor )
{
    uchar4 newColorUChar4 = make_uchar4( newColor.x, newColor.y, newColor.z, newColor.w );

    mSegmenterState.hostVectors.Get< int    >( "UndoIdMap"    ) = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "UndoColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );

    ::ReplaceSegmentationLabelInCommittedSegmentation3D( &mSegmenterState, oldId, newId, newColorUChar4 );

    mSegmenterState.hostVectors.Get< int    >( "RedoIdMap"    ) = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "RedoColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );
}

void Segmenter::ReplaceSegmentationLabelInCommittedSegmentation2DConnectedComponentOnly( int oldId, int newId, int4 newColor, int slice, int2 seed )
{
    uchar4 newColorUChar4 = make_uchar4( newColor.x, newColor.y, newColor.z, newColor.w );
    int3   seedInt3       = make_int3( seed, slice );
    int    seedIndex1D    = Core::Index3DToIndex1D( seedInt3, mSegmenterState.volumeDescription.numVoxels );

    mSegmenterState.hostVectors.Get< int    >( "UndoIdMap"    ) = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "UndoColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );

    

    //
    // perform a 2D flood fill on the CPU
    //
    mSegmenterState.hostVectors.Get< int    >( "FloodfillIdMap" )    = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "FloodfillColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );

    MOJO_THRUST_SAFE(
        thrust::fill( 
            mSegmenterState.hostVectors.Get< char >( "FloodfillVisitedMap" ).begin(),
            mSegmenterState.hostVectors.Get< char >( "FloodfillVisitedMap" ).end(),
            0 ) );

    RELEASE_ASSERT( mSegmenterState.hostVectors.Get< char >( "FloodfillIdMap" )[ seedIndex1D ] == oldId );

    std::queue< int3 > unprocessedElements;

    mSegmenterState.hostVectors.Get< int    >( "FloodfillIdMap"      )[ seedIndex1D ] = newId;
    mSegmenterState.hostVectors.Get< uchar4 >( "FloodfillColorMap"   )[ seedIndex1D ] = newColorUChar4;
    mSegmenterState.hostVectors.Get< char   >( "FloodfillVisitedMap" )[ seedIndex1D ] = 1;

    unprocessedElements.push( seedInt3 );

    while( !unprocessedElements.empty() )
    {
        int3 currentElement = unprocessedElements.front();
        unprocessedElements.pop();

        int3 adjacentElements[ 4 ];
        adjacentElements[ 0 ].x = currentElement.x - 1; adjacentElements[ 0 ].y = currentElement.y;     adjacentElements[ 0 ].z = currentElement.z;
        adjacentElements[ 1 ].x = currentElement.x + 1; adjacentElements[ 1 ].y = currentElement.y;     adjacentElements[ 1 ].z = currentElement.z;
        adjacentElements[ 2 ].x = currentElement.x;     adjacentElements[ 2 ].y = currentElement.y - 1; adjacentElements[ 2 ].z = currentElement.z;
        adjacentElements[ 3 ].x = currentElement.x;     adjacentElements[ 3 ].y = currentElement.y + 1; adjacentElements[ 3 ].z = currentElement.z;

        for( int i = 0; i < 4; i++ )
        {
            if ( adjacentElements[ i ].x < 0 || adjacentElements[ i ].x >= mSegmenterState.volumeDescription.numVoxels.x ||
                 adjacentElements[ i ].y < 0 || adjacentElements[ i ].y >= mSegmenterState.volumeDescription.numVoxels.y )
            {
                continue;
            }

            int adjacentIndex1D = Core::Index3DToIndex1D( adjacentElements[ i ], mSegmenterState.volumeDescription.numVoxels );

            if ( mSegmenterState.hostVectors.Get< char >( "FloodfillVisitedMap" )[ adjacentIndex1D ] == 1 )
            {
                continue;
            }
                
            mSegmenterState.hostVectors.Get< char >( "FloodfillVisitedMap" )[ adjacentIndex1D ] = 1;

            if ( mSegmenterState.hostVectors.Get< char >( "UndoIdMap" )[ adjacentIndex1D ] == oldId )
            {
                mSegmenterState.hostVectors.Get< int    >( "FloodfillIdMap"      )[ adjacentIndex1D ] = newId;
                mSegmenterState.hostVectors.Get< uchar4 >( "FloodfillColorMap"   )[ adjacentIndex1D ] = newColorUChar4;
                mSegmenterState.hostVectors.Get< char   >( "FloodfillVisitedMap" )[ adjacentIndex1D ] = 1;

                unprocessedElements.push( adjacentElements[ i ] );
            }
        }
    }

    mSegmenterState.deviceVectors.Get< int    >( "IdMap"    ) = mSegmenterState.hostVectors.Get< int    >( "FloodfillIdMap"    );
    mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" ) = mSegmenterState.hostVectors.Get< uchar4 >( "FloodfillColorMap" );



    mSegmenterState.hostVectors.Get< int    >( "RedoIdMap"    ) = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "RedoColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );
}

void Segmenter::ReplaceSegmentationLabelInCommittedSegmentation3DConnectedComponentOnly( int oldId, int newId, int4 newColor, int3 seed )
{
    uchar4 newColorUChar4 = make_uchar4( newColor.x, newColor.y, newColor.z, newColor.w );
    int    seedIndex1D    = Core::Index3DToIndex1D( seed, mSegmenterState.volumeDescription.numVoxels );

    mSegmenterState.hostVectors.Get< int    >( "UndoIdMap"    ) = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "UndoColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );

    

    //
    // perform a 3D flood fill on the CPU
    //
    mSegmenterState.hostVectors.Get< int    >( "FloodfillIdMap" )    = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "FloodfillColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );

    MOJO_THRUST_SAFE(
        thrust::fill( 
            mSegmenterState.hostVectors.Get< char >( "FloodfillVisitedMap" ).begin(),
            mSegmenterState.hostVectors.Get< char >( "FloodfillVisitedMap" ).end(),
            0 ) );

    RELEASE_ASSERT( mSegmenterState.hostVectors.Get< char >( "FloodfillIdMap" )[ seedIndex1D ] == oldId );

    std::queue< int3 > unprocessedElements;

    mSegmenterState.hostVectors.Get< int    >( "FloodfillIdMap"      )[ seedIndex1D ] = newId;
    mSegmenterState.hostVectors.Get< uchar4 >( "FloodfillColorMap"   )[ seedIndex1D ] = newColorUChar4;
    mSegmenterState.hostVectors.Get< char   >( "FloodfillVisitedMap" )[ seedIndex1D ] = 1;

    unprocessedElements.push( seed );

    while( !unprocessedElements.empty() )
    {
        int3 currentElement = unprocessedElements.front();
        unprocessedElements.pop();

        int3 adjacentElements[ 6 ];
        adjacentElements[ 0 ].x = currentElement.x - 1; adjacentElements[ 0 ].y = currentElement.y;     adjacentElements[ 0 ].z = currentElement.z;
        adjacentElements[ 1 ].x = currentElement.x + 1; adjacentElements[ 1 ].y = currentElement.y;     adjacentElements[ 1 ].z = currentElement.z;
        adjacentElements[ 2 ].x = currentElement.x;     adjacentElements[ 2 ].y = currentElement.y - 1; adjacentElements[ 2 ].z = currentElement.z;
        adjacentElements[ 3 ].x = currentElement.x;     adjacentElements[ 3 ].y = currentElement.y + 1; adjacentElements[ 3 ].z = currentElement.z;
        adjacentElements[ 4 ].x = currentElement.x;     adjacentElements[ 4 ].y = currentElement.y - 1; adjacentElements[ 4 ].z = currentElement.z - 1;
        adjacentElements[ 5 ].x = currentElement.x;     adjacentElements[ 5 ].y = currentElement.y + 1; adjacentElements[ 5 ].z = currentElement.z + 1;

        for( int i = 0; i < 6; i++ )
        {
            if ( adjacentElements[ i ].x < 0 || adjacentElements[ i ].x >= mSegmenterState.volumeDescription.numVoxels.x ||
                 adjacentElements[ i ].y < 0 || adjacentElements[ i ].y >= mSegmenterState.volumeDescription.numVoxels.y || 
                 adjacentElements[ i ].z < 0 || adjacentElements[ i ].z >= mSegmenterState.volumeDescription.numVoxels.z )
            {
                continue;
            }

            int adjacentIndex1D = Core::Index3DToIndex1D( adjacentElements[ i ], mSegmenterState.volumeDescription.numVoxels );

            if ( mSegmenterState.hostVectors.Get< char >( "FloodfillVisitedMap" )[ adjacentIndex1D ] == 1 )
            {
                continue;
            }
                
            mSegmenterState.hostVectors.Get< char >( "FloodfillVisitedMap" )[ adjacentIndex1D ] = 1;

            if ( mSegmenterState.hostVectors.Get< char >( "UndoIdMap" )[ adjacentIndex1D ] == oldId )
            {
                mSegmenterState.hostVectors.Get< int    >( "FloodfillIdMap"      )[ adjacentIndex1D ] = newId;
                mSegmenterState.hostVectors.Get< uchar4 >( "FloodfillColorMap"   )[ adjacentIndex1D ] = newColorUChar4;
                mSegmenterState.hostVectors.Get< char   >( "FloodfillVisitedMap" )[ adjacentIndex1D ] = 1;

                unprocessedElements.push( adjacentElements[ i ] );
            }
        }
    }

    mSegmenterState.deviceVectors.Get< int    >( "IdMap"    ) = mSegmenterState.hostVectors.Get< int    >( "FloodfillIdMap"    );
    mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" ) = mSegmenterState.hostVectors.Get< uchar4 >( "FloodfillColorMap" );



    mSegmenterState.hostVectors.Get< int    >( "RedoIdMap"    ) = mSegmenterState.deviceVectors.Get< int    >( "IdMap"    );
    mSegmenterState.hostVectors.Get< uchar4 >( "RedoColorMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );
}

void Segmenter::UndoLastChangeToCommittedSegmentation()
{
    mSegmenterState.deviceVectors.Get< int    >( "IdMap"    ) = mSegmenterState.hostVectors.Get< int    >( "UndoIdMap"    );
    mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" ) = mSegmenterState.hostVectors.Get< uchar4 >( "UndoColorMap" );
}

void Segmenter::RedoLastChangeToCommittedSegmentation()
{
    mSegmenterState.deviceVectors.Get< int    >( "IdMap"    ) = mSegmenterState.hostVectors.Get< int    >( "RedoIdMap"    );
    mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" ) = mSegmenterState.hostVectors.Get< uchar4 >( "RedoColorMap" );
}

void Segmenter::VisualUpdateColorMap()
{
    cudaArray* cudaArray = NULL;
    mSegmenterState.d3d11CudaTextures.MapCudaArrays();

    cudaArray = mSegmenterState.d3d11CudaTextures.Get( "ColorMap" )->GetMappedCudaArray();
    Core::Thrust::Memcpy3DToArray( cudaArray, mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" ), mSegmenterState.volumeDescription.numVoxels );

    mSegmenterState.d3d11CudaTextures.UnmapCudaArrays();
}

void Segmenter::InitializeCostMap()
{
    ::InitializeCostMap( &mSegmenterState );
}

void Segmenter::InitializeCostMapFromPrimalMap()
{
    ::InitializeCostMapFromPrimalMap( &mSegmenterState );
}

void Segmenter::IncrementCostMapFromPrimalMapForward()
{
    ::IncrementCostMapFromPrimalMapForward( &mSegmenterState );
}

void Segmenter::IncrementCostMapFromPrimalMapBackward()
{
    ::IncrementCostMapFromPrimalMapBackward( &mSegmenterState );
}

void Segmenter::FinalizeCostMapFromPrimalMap()
{
    ::FinalizeCostMapFromPrimalMap( &mSegmenterState );
}


void Segmenter::UpdateConstraintMapAndPrimalMapFromCostMap()
{
    ::UpdateConstraintMapAndPrimalMapFromCostMap( &mSegmenterState );
}

int Segmenter::GetSegmentationLabelId( int3 p )
{
    int    numVoxelsX      = mSegmenterState.volumeDescription.numVoxels.x;
    int    numVoxelsY      = mSegmenterState.volumeDescription.numVoxels.y;
    int    numVoxelsZ      = mSegmenterState.volumeDescription.numVoxels.z;
    int    segmentationLabelId = mSegmenterState.deviceVectors.Get< int >( "IdMap" )[ ( numVoxelsX * numVoxelsY * p.z ) + ( numVoxelsX * p.y ) + p.x ];

    return segmentationLabelId;
}

int4 Segmenter::GetSegmentationLabelColor( int3 p )
{
    int    numVoxelsX                 = mSegmenterState.volumeDescription.numVoxels.x;
    int    numVoxelsY                 = mSegmenterState.volumeDescription.numVoxels.y;
    int    numVoxelsZ                 = mSegmenterState.volumeDescription.numVoxels.z;
    uchar4 segmentationLabelColor     = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" )[ ( numVoxelsX * numVoxelsY * p.z ) + ( numVoxelsX * p.y ) + p.x ];
    int4   segmentationLabelColorInt4 = make_int4( segmentationLabelColor.x, segmentationLabelColor.y, segmentationLabelColor.z, segmentationLabelColor.w );

    return segmentationLabelColorInt4;
}

float Segmenter::GetPrimalValue( int3 p )
{
    int   numVoxelsX  = mSegmenterState.volumeDescription.numVoxels.x;
    int   numVoxelsY  = mSegmenterState.volumeDescription.numVoxels.y;
    int   numVoxelsZ  = mSegmenterState.volumeDescription.numVoxels.z;
    float primalValue = mSegmenterState.deviceVectors.Get< float >( "PrimalMap" )[ ( numVoxelsX * numVoxelsY * p.z ) + ( numVoxelsX * p.y ) + p.x ];

    return primalValue;
}

void Segmenter::DumpIntermediateData()
{
    int numVoxels = mSegmenterState.volumeDescription.numVoxels.x * mSegmenterState.volumeDescription.numVoxels.y * mSegmenterState.volumeDescription.numVoxels.z;

    if ( mSegmenterState.constParameters.Get< bool >( "DUMP_PRIMAL_MAP" ) )
    {
        mSegmenterState.hostVectors.Get< float >( "ScratchpadMap" ) = mSegmenterState.deviceVectors.Get< float >( "PrimalMap" );
        
        std::ofstream file( "PrimalMap.raw", std::ios::binary );
        file.write( (char*)( &mSegmenterState.hostVectors.Get< float >( "ScratchpadMap" )[ 0 ] ), numVoxels * sizeof( float ) );
        file.close();
    }

    if ( mSegmenterState.constParameters.Get< bool >( "DUMP_EDGE_XY_MAP" ) )
    {
        mSegmenterState.hostVectors.Get< float >( "ScratchpadMap" ) = mSegmenterState.deviceVectors.Get< float >( "EdgeXYMap" );

        std::ofstream file( "EdgeXYMap.raw", std::ios::binary );
        file.write( (char*)( &mSegmenterState.hostVectors.Get< float >( "ScratchpadMap" )[ 0 ] ), numVoxels * sizeof( float ) );
        file.close();
    }

    if ( mSegmenterState.constParameters.Get< bool >( "DUMP_EDGE_Z_MAP" ) )
    {
        RELEASE_ASSERT( mSegmenterState.constParameters.Get< bool >( "DIRECT_ANISOTROPIC_TV" ) );

        mSegmenterState.hostVectors.Get< float >( "ScratchpadMap" ) = mSegmenterState.deviceVectors.Get< float >( "EdgeZMap" );

        std::ofstream file( "EdgeZMap.raw", std::ios::binary );
        file.write( (char*)( &mSegmenterState.hostVectors.Get< float >( "ScratchpadMap" )[ 0 ] ), numVoxels * sizeof( float ) );
        file.close();
    }

    if ( mSegmenterState.constParameters.Get< bool >( "DUMP_CONSTRAINT_MAP" ) )
    {
        mSegmenterState.hostVectors.Get< float >( "ScratchpadMap" ) = mSegmenterState.deviceVectors.Get< float >( "ConstraintMap" );

        std::ofstream file( "ConstraintMap.raw", std::ios::binary );
        file.write( (char*)( &mSegmenterState.hostVectors.Get< float >( "ScratchpadMap" )[ 0 ] ), numVoxels * sizeof( float ) );
        file.close();
    }

    if ( mSegmenterState.constParameters.Get< bool >( "DUMP_ID_MAP" ) )
    {
        mSegmenterState.hostVectors.Get< int >( "ScratchpadMap" ) = mSegmenterState.deviceVectors.Get< int >( "IdMap" );

        std::ofstream file( "IdMap.raw", std::ios::binary );
        file.write( (char*)( &mSegmenterState.hostVectors.Get< int >( "ScratchpadMap" )[ 0 ] ), numVoxels * sizeof( int ) );
        file.close();
    }

    if ( mSegmenterState.constParameters.Get< bool >( "DUMP_COLOR_MAP" ) )
    {
        mSegmenterState.hostVectors.Get< uchar4 >( "ScratchpadMap" ) = mSegmenterState.deviceVectors.Get< uchar4 >( "ColorMap" );

        std::ofstream file( "ColorMap.raw", std::ios::binary );
        file.write( (char*)( &mSegmenterState.hostVectors.Get< uchar4 >( "ScratchpadMap" )[ 0 ] ), numVoxels * sizeof( uchar4 ) );
        file.close();
    }
}

void Segmenter::DebugInitialize()
{
    ::DebugInitialize( &mSegmenterState );
}

void Segmenter::DebugTerminate()
{
    ::DebugTerminate( &mSegmenterState );
}

void Segmenter::DebugUpdate()
{
    ::DebugUpdate( &mSegmenterState );
}

Core::SegmenterState* Segmenter::GetSegmenterState()
{
    return &mSegmenterState;
}

void Segmenter::UnloadVolumeInternal()
{
    //
    // output memory stats to the console
    //
    unsigned int freeMemory, totalMemory;
    CUresult memInfoResult;

    memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
    RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );
    Core::Printf( "\nUnloading dataset...\n" );
    Core::Printf( "    Before freeing GPU memory:\n",
                  "        Free memory:  ", freeMemory  / ( 1024 * 1024 ), " MBytes.\n",
                  "        Total memory: ", totalMemory / ( 1024 * 1024 ), " MBytes.\n" );

    //
    // clear device state
    //
    mSegmenterState.deviceVectors.Clear();

    MOJO_FOR_EACH_VALUE( Core::ID3D11CudaTexture* d3d11CudaTexture, mSegmenterState.d3d11CudaTextures.GetHashMap() )
    {
        delete d3d11CudaTexture;
    }
    mSegmenterState.d3d11CudaTextures.GetHashMap().clear();

    MOJO_FOR_EACH_VALUE( cudaArray* c, mSegmenterState.cudaArrays.GetHashMap() )
    {
        MOJO_CUDA_SAFE( cudaFreeArray( c ) );
    }
    mSegmenterState.d3d11CudaTextures.GetHashMap().clear();

    //
    // output memory stats to the console
    //
    memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
    RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );
    Core::Printf( "    After freeing GPU memory:\n",
                  "        Free memory:  ", freeMemory  / ( 1024 * 1024 ), " MBytes.\n",
                  "        Total memory: ", totalMemory / ( 1024 * 1024 ), " MBytes.\n" );

    //
    // clear host state
    //
    mSegmenterState.hostVectors.Clear();

    mSegmenterState.slicesWithForegroundConstraints.clear();
    mSegmenterState.slicesWithBackgroundConstraints.clear();

    mSegmenterState.volumeDescription = Core::VolumeDescription();

    mSegmenterState.dynamicParameters.Set( "ConvergenceGap",         0.0f );
    mSegmenterState.dynamicParameters.Set( "ConvergenceGapDelta",    0.0f );
    mSegmenterState.dynamicParameters.Set( "MaxForegroundCostDelta", mSegmenterState.constParameters.Get< float >( "COST_MAP_INITIAL_MAX_FOREGROUND_COST_DELTA" ) );
    mSegmenterState.dynamicParameters.Set( "IsVolumeLoaded",         false );

}

}
}