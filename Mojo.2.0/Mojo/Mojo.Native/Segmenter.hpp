#pragma once

#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/SegmenterState.hpp"
#include "Mojo.Core/PrimitiveMap.hpp"

struct ID3D11Device;
struct ID3D11DeviceContext;

namespace Mojo
{
namespace Native
{

class Segmenter
{
public:
    Segmenter( ID3D11Device* d3d11Device, ID3D11DeviceContext* d3d11DeviceContext, Core::PrimitiveMap constParameters );
    ~Segmenter();

    void LoadVolume( Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions );
    void UnloadVolume();

    void LoadSegmentation( Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions );
    void SaveSegmentationAs( Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions );

    void InitializeEdgeXYMap( Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions );
    void InitializeEdgeXYMapForSplitting( Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions, int segmentationLabelId );

    void  InitializeSegmentation();
    void  InitializeSegmentationAndRemoveFromCommittedSegmentation( int segmentationLabelId );
          
    void  InitializeConstraintMap();
    void  InitializeConstraintMapFromIdMap( int segmentationLabelId );
    void  InitializeConstraintMapFromIdMapForSplitting( int segmentationLabelId );
    void  InitializeConstraintMapFromPrimalMap();
    void  DilateConstraintMap();
          
    void  AddForegroundHardConstraint( int3 p, float radius );
    void  AddBackgroundHardConstraint( int3 p, float radius );
    void  AddForegroundHardConstraint( int3 p1, int3 p2, float radius );
    void  AddBackgroundHardConstraint( int3 p1, int3 p2, float radius );
          
    void  Update2D( int numIterations, int zSlice );
    void  Update3D( int numIterations );
    void  VisualUpdate();
          
    void  UpdateCommittedSegmentation( int segmentationLabelId, int4 segmentationLabelColor );
    void  UpdateCommittedSegmentationDoNotRemove( int segmentationLabelId, int4 segmentationLabelColor );
          
    void  ReplaceSegmentationLabelInCommittedSegmentation2D( int oldId, int newId, int4 newColor, int slice );
    void  ReplaceSegmentationLabelInCommittedSegmentation3D( int oldId, int newId, int4 newColor );
    void  ReplaceSegmentationLabelInCommittedSegmentation2DConnectedComponentOnly( int oldId, int newId, int4 newColor, int slice, int2 seed );
    void  ReplaceSegmentationLabelInCommittedSegmentation3DConnectedComponentOnly( int oldId, int newId, int4 newColor, int3 seed );
          
    void  UndoLastChangeToCommittedSegmentation();
    void  RedoLastChangeToCommittedSegmentation();
    void  VisualUpdateColorMap();
          
    void  InitializeCostMap();
    void  InitializeCostMapFromPrimalMap();
    void  IncrementCostMapFromPrimalMapForward();
    void  IncrementCostMapFromPrimalMapBackward();
    void  FinalizeCostMapFromPrimalMap();

    void  UpdateConstraintMapAndPrimalMapFromCostMap();

    int   GetSegmentationLabelId( int3 p );
    int4  GetSegmentationLabelColor( int3 p );
    float GetPrimalValue( int3 p );

    void  DumpIntermediateData();

    void  DebugInitialize();
    void  DebugUpdate();
    void  DebugTerminate();

    Core::SegmenterState* GetSegmenterState();

private:
    template < typename TCudaType >
    void  LoadVolumeInternal( Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions );
    void  UnloadVolumeInternal();

    bool  AddHardConstraint( int3 p1, int3 p2, float radius, float constraintValue, float primalValue );

    Core::SegmenterState mSegmenterState;
};

template < typename TCudaType >
inline void Segmenter::LoadVolumeInternal( Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions )
{
    int numElements         = volumeDescriptions.Get( "SourceMap" ).numVoxels.x * volumeDescriptions.Get( "SourceMap" ).numVoxels.y * volumeDescriptions.Get( "SourceMap" ).numVoxels.z;
    int numElementsPerSlice = volumeDescriptions.Get( "SourceMap" ).numVoxels.x * volumeDescriptions.Get( "SourceMap" ).numVoxels.y;


    //
    // initialize host state
    //
    mSegmenterState.volumeDescription = volumeDescriptions.Get( "SourceMap" );

    mSegmenterState.dynamicParameters.Set( "ConvergenceGap",         0.0f );
    mSegmenterState.dynamicParameters.Set( "ConvergenceGapDelta",    0.0f );
    mSegmenterState.dynamicParameters.Set( "MaxForegroundCostDelta", mSegmenterState.constParameters.Get< float >( "COST_MAP_INITIAL_MAX_FOREGROUND_COST_DELTA" ) );
    mSegmenterState.dynamicParameters.Set( "IsVolumeLoaded",         true );

    mSegmenterState.slicesWithForegroundConstraints.clear();
    mSegmenterState.slicesWithBackgroundConstraints.clear();

    mSegmenterState.hostVectors.Set( "ScratchpadMap",       thrust::host_vector< uchar4 >( numElements, make_uchar4( 0, 0, 0, 0 ) ) );
    mSegmenterState.hostVectors.Set( "UndoColorMap",        thrust::host_vector< uchar4 >( numElements, make_uchar4( 0, 0, 0, 0 ) ) );
    mSegmenterState.hostVectors.Set( "RedoColorMap",        thrust::host_vector< uchar4 >( numElements, make_uchar4( 0, 0, 0, 0 ) ) );
    mSegmenterState.hostVectors.Set( "FloodfillColorMap",   thrust::host_vector< uchar4 >( numElements, make_uchar4( 0, 0, 0, 0 ) ) );
    mSegmenterState.hostVectors.Set( "ScratchpadMap",       thrust::host_vector< float  >( numElements, 0.0f ) );
    mSegmenterState.hostVectors.Set( "ScratchpadMap",       thrust::host_vector< int    >( numElements, 0 ) );
    mSegmenterState.hostVectors.Set( "UndoIdMap",           thrust::host_vector< int    >( numElements, 0 ) );
    mSegmenterState.hostVectors.Set( "RedoIdMap",           thrust::host_vector< int    >( numElements, 0 ) );
    mSegmenterState.hostVectors.Set( "FloodfillIdMap",      thrust::host_vector< int    >( numElements, 0 ) );
    mSegmenterState.hostVectors.Set( "FloodfillVisitedMap", thrust::host_vector< char   >( numElements, 0 ) );


    //
    // initialize device state
    //
    Core::Printf(
        "\nLoading dataset. Size in bytes = ",
        volumeDescriptions.Get( "SourceMap" ).numVoxels.x, " * ",
        volumeDescriptions.Get( "SourceMap" ).numVoxels.y, " * ",
        volumeDescriptions.Get( "SourceMap" ).numVoxels.z, " * ", 
        volumeDescriptions.Get( "SourceMap" ).numBytesPerVoxel, " = ",
        ( numElements * volumeDescriptions.Get( "SourceMap" ).numBytesPerVoxel ) / ( 1024 * 1024 ), " MBytes." );


    size_t freeMemory, totalMemory;
    CUresult     memInfoResult;
    memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
    RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );
    Core::Printf( "\nBefore allocating GPU memory:\n",
                  "    Free memory:  ", (unsigned int) freeMemory  / ( 1024 * 1024 ), " MBytes.\n",
                  "    Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );


    D3D11_TEXTURE3D_DESC textureDesc3D;
    ZeroMemory( &textureDesc3D, sizeof( D3D11_TEXTURE3D_DESC ) );

    textureDesc3D.Width     = mSegmenterState.volumeDescription.numVoxels.x;
    textureDesc3D.Height    = mSegmenterState.volumeDescription.numVoxels.y;
    textureDesc3D.Depth     = mSegmenterState.volumeDescription.numVoxels.z;
    textureDesc3D.MipLevels = 1;
    textureDesc3D.Usage     = D3D11_USAGE_DEFAULT;
    textureDesc3D.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    textureDesc3D.Format = volumeDescriptions.Get( "SourceMap" ).dxgiFormat;
    mSegmenterState.d3d11CudaTextures.Set( "SourceMap",          new Core::D3D11CudaTexture< ID3D11Texture3D, TCudaType >( mSegmenterState.d3d11Device, mSegmenterState.d3d11DeviceContext, textureDesc3D, volumeDescriptions.Get( "SourceMap" ) ) );

    textureDesc3D.Format = DXGI_FORMAT_R32_FLOAT;
    mSegmenterState.d3d11CudaTextures.Set( "ConstraintMap",      new Core::D3D11CudaTexture< ID3D11Texture3D, float >( mSegmenterState.d3d11Device, mSegmenterState.d3d11DeviceContext, textureDesc3D ) );
    mSegmenterState.d3d11CudaTextures.Set( "PrimalMap",          new Core::D3D11CudaTexture< ID3D11Texture3D, float >( mSegmenterState.d3d11Device, mSegmenterState.d3d11DeviceContext, textureDesc3D ) );

    textureDesc3D.Format = DXGI_FORMAT_R8G8B8A8_UNORM;          
    mSegmenterState.d3d11CudaTextures.Set( "ColorMap",           new Core::D3D11CudaTexture< ID3D11Texture3D, uchar4 >( mSegmenterState.d3d11Device, mSegmenterState.d3d11DeviceContext, textureDesc3D ) );

    mSegmenterState.cudaArrays.Set( "SourceMap",                 Core::Cuda::MallocArray3D< TCudaType >( mSegmenterState.volumeDescription ) );
    mSegmenterState.cudaArrays.Set( "TempScratchpadMap",         Core::Cuda::MallocArray2D< float     >( mSegmenterState.volumeDescription ) );

    mSegmenterState.deviceVectors.Set( "DualMap",                thrust::device_vector< float4 >( numElements, mSegmenterState.constParameters.Get< float4 >( "DUAL_MAP_INITIAL_VALUE"  ) ) );
    mSegmenterState.deviceVectors.Set( "ColorMap",               thrust::device_vector< uchar4 >( numElements, mSegmenterState.constParameters.Get< uchar4 >( "COLOR_MAP_INITIAL_VALUE" ) ) );
    mSegmenterState.deviceVectors.Set( "EdgeXYMap",              thrust::device_vector< float  >( numElements, mSegmenterState.constParameters.Get< float  >( "EDGE_MAP_INITIAL_VALUE"  ) ) );
    
    if ( mSegmenterState.constParameters.Get< bool >( "DIRECT_ANISOTROPIC_TV" ) )
    {
        mSegmenterState.deviceVectors.Set( "EdgeZMap",           thrust::device_vector< float >( numElements, mSegmenterState.constParameters.Get< float >( "EDGE_MAP_INITIAL_VALUE" ) ) );
    }

    mSegmenterState.deviceVectors.Set( "ConstraintMap",          thrust::device_vector< float >( numElements, mSegmenterState.constParameters.Get< float >( "CONSTRAINT_MAP_INITIAL_VALUE" ) ) );
    mSegmenterState.deviceVectors.Set( "OldPrimalMap",           thrust::device_vector< float >( numElements, mSegmenterState.constParameters.Get< float >( "OLD_PRIMAL_MAP_INITIAL_VALUE" ) ) );
    mSegmenterState.deviceVectors.Set( "ScratchpadMap",          thrust::device_vector< float >( numElements, mSegmenterState.constParameters.Get< float >( "SCRATCHPAD_MAP_INITIAL_VALUE" ) ) );
    mSegmenterState.deviceVectors.Set( "CostForwardMap",         thrust::device_vector< float >( numElements, mSegmenterState.constParameters.Get< float >( "COST_MAP_INITIAL_VALUE"       ) ) );
    mSegmenterState.deviceVectors.Set( "CostBackwardMap",        thrust::device_vector< float >( numElements, mSegmenterState.constParameters.Get< float >( "COST_MAP_INITIAL_VALUE"       ) ) );
    mSegmenterState.deviceVectors.Set( "PrimalMap",              thrust::device_vector< float >( numElements, mSegmenterState.constParameters.Get< float >( "PRIMAL_MAP_INITIAL_VALUE"     ) ) );
    mSegmenterState.deviceVectors.Set( "IdMap",                  thrust::device_vector< int   >( numElements, mSegmenterState.constParameters.Get< int >  ( "ID_MAP_INITIAL_VALUE"         ) ) );

    Core::Cuda::MemcpyHostToArray3D( mSegmenterState.cudaArrays.Get( "SourceMap" ), volumeDescriptions.Get( "SourceMap" ) );

    ::InitializeEdgeXYMap( &mSegmenterState, &volumeDescriptions );

    if (  mSegmenterState.volumeDescription.numVoxels.z > 1 )
    {
        mSegmenterState.deviceVectors.Set( "OpticalFlowForwardMap",  thrust::device_vector< float2 >( numElements, mSegmenterState.constParameters.Get< float2 >( "OPTICAL_FLOW_MAP_INITIAL_VALUE" ) ) );
        mSegmenterState.deviceVectors.Set( "OpticalFlowBackwardMap", thrust::device_vector< float2 >( numElements, mSegmenterState.constParameters.Get< float2 >( "OPTICAL_FLOW_MAP_INITIAL_VALUE" ) ) );

        int numElementsOpticalFlow = volumeDescriptions.Get( "OpticalFlowForwardMap" ).numVoxels.x * volumeDescriptions.Get( "OpticalFlowForwardMap" ).numVoxels.y * volumeDescriptions.Get( "OpticalFlowForwardMap" ).numVoxels.z;

        Core::Thrust::MemcpyHostToDevice( mSegmenterState.deviceVectors.Get< float2 >( "OpticalFlowForwardMap"  )[ 0 ],                   volumeDescriptions.Get( "OpticalFlowForwardMap"  ).data, numElementsOpticalFlow );
        Core::Thrust::MemcpyHostToDevice( mSegmenterState.deviceVectors.Get< float2 >( "OpticalFlowBackwardMap" )[ numElementsPerSlice ], volumeDescriptions.Get( "OpticalFlowBackwardMap" ).data, numElementsOpticalFlow );

        if ( mSegmenterState.constParameters.Get< bool >( "DIRECT_ANISOTROPIC_TV" ) )
        {
            ::InitializeEdgeZMap( &mSegmenterState, &volumeDescriptions );
        }
    }

    memInfoResult = cuMemGetInfo( &freeMemory, &totalMemory );
    RELEASE_ASSERT( memInfoResult == CUDA_SUCCESS );
    Core::Printf( "After allocating GPU memory:\n",
                  "    Free memory:  ", (unsigned int) freeMemory  / ( 1024 * 1024 ), " MBytes.\n",
                  "    Total memory: ", (unsigned int) totalMemory / ( 1024 * 1024 ), " MBytes.\n" );
}

}
}