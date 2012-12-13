#pragma once

#include "Mojo.Native/Segmenter.hpp"

#include "ObservableDictionary.hpp"
#include "PrimitiveMap.hpp"
#include "VolumeDescription.hpp"

#using <SlimDX.dll>

using namespace SlimDX;

namespace Mojo
{
namespace Interop
{

#pragma managed
public ref class Segmenter : public NotifyPropertyChanged
{
public:
    Segmenter( SlimDX::Direct3D11::Device^ d3d11Device, SlimDX::Direct3D11::DeviceContext^ d3d11DeviceContext, PrimitiveMap^ constParameters );
    ~Segmenter();

    void LoadVolume( ObservableDictionary< String^, VolumeDescription^ >^ volumeDescriptions );
    void UnloadVolume();

    bool IsVolumeLoaded();

    void LoadSegmentation( ObservableDictionary< String^, VolumeDescription^ >^ volumeDescriptions );
    void SaveSegmentationAs( ObservableDictionary< String^, VolumeDescription^ >^ volumeDescriptions );

    void InitializeEdgeXYMap( ObservableDictionary< String^, VolumeDescription^ >^ volumeDescriptions );
    void InitializeEdgeXYMapForSplitting( ObservableDictionary< String^, VolumeDescription^ >^ volumeDescriptions, int segmentationLabelId );

    void InitializeSegmentation();
    void InitializeSegmentationAndRemoveFromCommittedSegmentation( int segmentationLabelId );

    void InitializeConstraintMap();
    void InitializeConstraintMapFromIdMap( int segmentationLabelId );
    void InitializeConstraintMapFromIdMapForSplitting( int segmentationLabelId );
    void InitializeConstraintMapFromPrimalMap();
    void DilateConstraintMap();

    void AddForegroundHardConstraint( Vector3^ p, float radius );
    void AddBackgroundHardConstraint( Vector3^ p, float radius );

    void AddForegroundHardConstraint( Vector3^ p1, Vector3^ p2, float radius );
    void AddBackgroundHardConstraint( Vector3^ p1, Vector3^ p2, float radius );

    void Update2D( int numIterations, int zSlice );
    void Update3D( int numIterations );
    void VisualUpdate();

    void UpdateCommittedSegmentation( int segmentationLabelId, Vector3^ segmentationLabelColor );
    void UpdateCommittedSegmentationDoNotRemove( int segmentationLabelId, Vector3^ segmentationLabelColor );

    void ReplaceSegmentationLabelInCommittedSegmentation2D( int oldId, int newId, Vector3^ newColor, int slice );
    void ReplaceSegmentationLabelInCommittedSegmentation3D( int oldId, int newId, Vector3^ newColor );
    void ReplaceSegmentationLabelInCommittedSegmentation2DConnectedComponentOnly( int oldId, int newId, Vector3^ newColor, int slice, Vector3^ seed );
    void ReplaceSegmentationLabelInCommittedSegmentation3DConnectedComponentOnly( int oldId, int newId, Vector3^ newColor, Vector3^ seed );

    void UndoLastChangeToCommittedSegmentation();
    void RedoLastChangeToCommittedSegmentation();
    void VisualUpdateColorMap();

    void InitializeCostMap();

    void InitializeCostMapFromPrimalMap();
    void IncrementCostMapFromPrimalMapForward();
    void IncrementCostMapFromPrimalMapBackward();
    void FinalizeCostMapFromPrimalMap();

    void UpdateConstraintMapAndPrimalMapFromCostMap();

    int     GetSegmentationLabelId( Vector3^ p );
    Vector3 GetSegmentationLabelColor( Vector3^ p );
    float   GetPrimalValue( Vector3^ p );

    void DumpIntermediateData();

    void DebugInitialize();
    void DebugUpdate();
    void DebugTerminate();

    //
    // methods for accessing internal segmenter state
    //
    ObservableDictionary< String^, ShaderResourceView^ >^ GetD3D11CudaTextures();

    VolumeDescription^                                    GetVolumeDescription();
                                                          
    float                                                 GetConvergenceGap();
    void                                                  SetConvergenceGap( float convergenceGap );
                                                          
    float                                                 GetConvergenceGapDelta();
    void                                                  SetConvergenceGapDelta( float convergenceGapDelta );
                                                          
    float                                                 GetMaxForegroundCostDelta();
    void                                                  SetMaxForegroundCostDelta( float maxForegroundCostDelta );

private:
    void LoadD3D11CudaTextures();
    void UnloadD3D11CudaTextures();

    Native::Segmenter* mSegmenter;

    ObservableDictionary< String^, ShaderResourceView^ >^ mD3D11CudaTextures;
};

}
}