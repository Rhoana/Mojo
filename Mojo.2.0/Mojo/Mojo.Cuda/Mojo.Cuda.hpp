#pragma once

#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/VolumeDescription.hpp"

namespace Mojo
{
namespace Core
{
class SegmenterState;
}
}

extern "C" void InitializeScratchpad                                    ( Mojo::Core::SegmenterState* segmenterState );
                                                                        
extern "C" void InitializeEdgeXYMap                                     ( Mojo::Core::SegmenterState* segmenterState, Mojo::Core::HashMap< std::string, Mojo::Core::VolumeDescription >* volumeDescriptions );
extern "C" void InitializeEdgeXYMapForSplitting                         ( Mojo::Core::SegmenterState* segmenterState, Mojo::Core::HashMap< std::string, Mojo::Core::VolumeDescription >* volumeDescriptions, int id );

extern "C" void InitializeEdgeZMap                                      ( Mojo::Core::SegmenterState* segmenterState, Mojo::Core::HashMap< std::string, Mojo::Core::VolumeDescription >* volumeDescriptions );
                                                                        
extern "C" void InitializeConstraintMap                                 ( Mojo::Core::SegmenterState* segmenterState );
extern "C" void InitializeConstraintMapFromIdMap                        ( Mojo::Core::SegmenterState* segmenterState, int id );
extern "C" void InitializeConstraintMapFromIdMapForSplitting            ( Mojo::Core::SegmenterState* segmenterState, int id );
extern "C" void InitializeConstraintMapFromPrimalMap                    ( Mojo::Core::SegmenterState* segmenterState );
extern "C" void AddHardConstraint                                       ( Mojo::Core::SegmenterState* segmenterState, int3 p1, int3 p2, float radius, float newConstraintValue, float primalValue );
extern "C" void DilateConstraintMap                                     ( Mojo::Core::SegmenterState* segmenterState );

extern "C" void UpdateConstraintMapAndPrimalMapFromCostMap              ( Mojo::Core::SegmenterState* segmenterState );
                                                                        
extern "C" void InitializeCommittedSegmentation                         ( Mojo::Core::SegmenterState* segmenterState );
extern "C" void InitializeSegmentationAndRemoveFromCommittedSegmentation( Mojo::Core::SegmenterState* segmenterState, int id );
extern "C" void ReplaceSegmentationLabelInCommittedSegmentation2D           ( Mojo::Core::SegmenterState* segmenterState, int oldId, int newId, uchar4 newColor, int slice );
extern "C" void ReplaceSegmentationLabelInCommittedSegmentation3D           ( Mojo::Core::SegmenterState* segmenterState, int oldId, int newId, uchar4 newColor );
extern "C" void UpdateCommittedSegmentation                             ( Mojo::Core::SegmenterState* segmenterState, int id, uchar4 color );
extern "C" void UpdateCommittedSegmentationDoNotRemove                  ( Mojo::Core::SegmenterState* segmenterState, int id, uchar4 color );
                                                                      
extern "C" void InitializeSegmentation                                  ( Mojo::Core::SegmenterState* segmenterState );

extern "C" void UpdateDualMap2D                                         ( Mojo::Core::SegmenterState* segmenterState, float sigma, int zSlice );
extern "C" void UpdatePrimalMap2D                                       ( Mojo::Core::SegmenterState* segmenterState, float lambda, float tau, int zSlice );
extern "C" void CalculatePrimalEnergy2D                                 ( Mojo::Core::SegmenterState* segmenterState, float lambda, int zSlice, float& primalEnergy );
extern "C" void CalculateDualEnergy2D                                   ( Mojo::Core::SegmenterState* segmenterState, float lambda, int zSlice, float& dualEnergy );
                                                                        
extern "C" void UpdateDualMap3D                                         ( Mojo::Core::SegmenterState* segmenterState, float sigma );
extern "C" void UpdatePrimalMap3D                                       ( Mojo::Core::SegmenterState* segmenterState, float lambda, float tau );
extern "C" void CalculatePrimalEnergy3D                                 ( Mojo::Core::SegmenterState* segmenterState, float lambda, float& primalEnergy );
extern "C" void CalculateDualEnergy3D                                   ( Mojo::Core::SegmenterState* segmenterState, float lambda, float& dualEnergy );
                                                                        
extern "C" void InitializeCostMap                                       ( Mojo::Core::SegmenterState* segmenterState );
extern "C" void InitializeCostMapFromPrimalMap                          ( Mojo::Core::SegmenterState* segmenterState );
extern "C" void IncrementCostMapFromPrimalMapForward                    ( Mojo::Core::SegmenterState* segmenterState );
extern "C" void IncrementCostMapFromPrimalMapBackward                   ( Mojo::Core::SegmenterState* segmenterState );
extern "C" void FinalizeCostMapFromPrimalMap                            ( Mojo::Core::SegmenterState* segmenterState );

extern "C" void UpdateConstraintMapAndPrimalMapFromCostMap              ( Mojo::Core::SegmenterState* segmenterState );

extern "C" void DebugInitialize                                         ( Mojo::Core::SegmenterState* segmenterState );
extern "C" void DebugUpdate                                             ( Mojo::Core::SegmenterState* segmenterState );
extern "C" void DebugTerminate                                          ( Mojo::Core::SegmenterState* segmenterState );
