#include "Segmenter.hpp"

#include "Mojo.Core/Stl.hpp"

#include <msclr/marshal_cppstd.h>

#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/ForEach.hpp"
#include "Mojo.Core/ID3D11CudaTexture.hpp"

#include "ObservableDictionary.hpp"
#include "VolumeDescription.hpp"

namespace Mojo
{
namespace Interop
{

Segmenter::Segmenter( SlimDX::Direct3D11::Device^ d3d11Device, SlimDX::Direct3D11::DeviceContext^ d3d11DeviceContext, PrimitiveMap^ constParameters )
{
    mSegmenter = new Native::Segmenter(
        reinterpret_cast< ID3D11Device* >( d3d11Device->ComPointer.ToPointer() ),
        reinterpret_cast< ID3D11DeviceContext* >( d3d11DeviceContext->ComPointer.ToPointer() ),
        constParameters->ToCore() );

    mD3D11CudaTextures = gcnew ObservableDictionary< String^, ShaderResourceView^ >();
}

Segmenter::~Segmenter()
{
    UnloadD3D11CudaTextures();

    if ( mSegmenter != NULL )
    {
        delete mSegmenter;
        mSegmenter = NULL;
    }
}

void Segmenter::LoadVolume( ObservableDictionary< String^, VolumeDescription^ >^ inVolumeDescriptions )
{
    Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

    for each ( Collections::Generic::KeyValuePair< String^, VolumeDescription^ > keyValuePair in inVolumeDescriptions )
    {
        volumeDescriptions.Set( msclr::interop::marshal_as< std::string >( keyValuePair.Key ), keyValuePair.Value->ToCore() );
    }

    mSegmenter->LoadVolume( volumeDescriptions );

    UnloadD3D11CudaTextures();
    LoadD3D11CudaTextures();
}

void Segmenter::UnloadVolume()
{
    mSegmenter->UnloadVolume();

    UnloadD3D11CudaTextures();
    LoadD3D11CudaTextures();
}

void Segmenter::LoadSegmentation( ObservableDictionary< String^, VolumeDescription^ >^ inVolumeDescriptions )
{
    Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

    for each ( Collections::Generic::KeyValuePair< String^, VolumeDescription^ > keyValuePair in inVolumeDescriptions )
    {
        volumeDescriptions.Set( msclr::interop::marshal_as< std::string >( keyValuePair.Key ), keyValuePair.Value->ToCore() );
    }

    mSegmenter->LoadSegmentation( volumeDescriptions );
}

void Segmenter::SaveSegmentationAs( ObservableDictionary< String^, VolumeDescription^ >^ inVolumeDescriptions )
{
    Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

    for each ( Collections::Generic::KeyValuePair< String^, VolumeDescription^ > keyValuePair in inVolumeDescriptions )
    {
        volumeDescriptions.Set( msclr::interop::marshal_as< std::string >( keyValuePair.Key ), keyValuePair.Value->ToCore() );
    }

    mSegmenter->SaveSegmentationAs( volumeDescriptions );
}

void Segmenter::InitializeEdgeXYMap( ObservableDictionary< String^, VolumeDescription^ >^ inVolumeDescriptions )
{
    Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

    for each ( Collections::Generic::KeyValuePair< String^, VolumeDescription^ > keyValuePair in inVolumeDescriptions )
    {
        volumeDescriptions.Set( msclr::interop::marshal_as< std::string >( keyValuePair.Key ), keyValuePair.Value->ToCore() );
    }

    mSegmenter->InitializeEdgeXYMap( volumeDescriptions );
}

void Segmenter::InitializeEdgeXYMapForSplitting( ObservableDictionary< String^, VolumeDescription^ >^ inVolumeDescriptions, int segmentationLabelId )
{
    Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

    for each ( Collections::Generic::KeyValuePair< String^, VolumeDescription^ > keyValuePair in inVolumeDescriptions )
    {
        volumeDescriptions.Set( msclr::interop::marshal_as< std::string >( keyValuePair.Key ), keyValuePair.Value->ToCore() );
    }

    mSegmenter->InitializeEdgeXYMapForSplitting( volumeDescriptions, segmentationLabelId );
}

void Segmenter::InitializeConstraintMap()
{
    mSegmenter->InitializeConstraintMap();
}

void Segmenter::InitializeConstraintMapFromIdMap( int segmentationLabelId )
{
    mSegmenter->InitializeConstraintMapFromIdMap( segmentationLabelId );
}

void Segmenter::InitializeConstraintMapFromIdMapForSplitting( int segmentationLabelId )
{
    mSegmenter->InitializeConstraintMapFromIdMapForSplitting( segmentationLabelId );
}

void Segmenter::InitializeConstraintMapFromPrimalMap()
{
    mSegmenter->InitializeConstraintMapFromPrimalMap();
}

void Segmenter::DilateConstraintMap()
{
    mSegmenter->DilateConstraintMap();
}

void Segmenter::InitializeSegmentation()
{
    mSegmenter->InitializeSegmentation();
}

void Segmenter::InitializeSegmentationAndRemoveFromCommittedSegmentation( int segmentationLabelId )
{
    mSegmenter->InitializeSegmentationAndRemoveFromCommittedSegmentation( segmentationLabelId );
}

void Segmenter::AddForegroundHardConstraint( Vector3^ p, float radius )
{
    int3 pInt3 = make_int3( (int)p->X, (int)p->Y, (int)p->Z );

    mSegmenter->AddForegroundHardConstraint( pInt3, radius );
}

void Segmenter::AddBackgroundHardConstraint( Vector3^ p, float radius )
{
    int3 pInt3 = make_int3( (int)p->X, (int)p->Y, (int)p->Z );

    mSegmenter->AddBackgroundHardConstraint( pInt3, radius );
}

void Segmenter::AddForegroundHardConstraint( Vector3^ p1, Vector3^ p2, float radius )
{
    int3 p1Int3 = make_int3( (int)p1->X, (int)p1->Y, (int)p1->Z );
    int3 p2Int3 = make_int3( (int)p2->X, (int)p2->Y, (int)p2->Z );

    mSegmenter->AddForegroundHardConstraint( p1Int3, p2Int3, radius );
}

void Segmenter::AddBackgroundHardConstraint( Vector3^ p1, Vector3^ p2, float radius )
{
    int3 p1Int3 = make_int3( (int)p1->X, (int)p1->Y, (int)p1->Z );
    int3 p2Int3 = make_int3( (int)p2->X, (int)p2->Y, (int)p2->Z );

    mSegmenter->AddBackgroundHardConstraint( p1Int3, p2Int3, radius );
}

void Segmenter::Update2D( int numIterations, int zSlice )
{
    mSegmenter->Update2D( numIterations, zSlice );
}

void Segmenter::Update3D( int numIterations )
{
    mSegmenter->Update3D( numIterations );
}

void Segmenter::VisualUpdate()
{
    mSegmenter->VisualUpdate();
}

void Segmenter::UpdateCommittedSegmentation( int segmentationLabelId, Vector3^ segmentationLabelColor )
{
    int4 segmentationLabelColorInt4 = make_int4( (int)segmentationLabelColor->X, (int)segmentationLabelColor->Y, (int)segmentationLabelColor->Z, 255 );

    mSegmenter->UpdateCommittedSegmentation( segmentationLabelId, segmentationLabelColorInt4 );
}

void Segmenter::UpdateCommittedSegmentationDoNotRemove( int segmentationLabelId, Vector3^ segmentationLabelColor )
{
    int4 segmentationLabelColorInt4 = make_int4( (int)segmentationLabelColor->X, (int)segmentationLabelColor->Y, (int)segmentationLabelColor->Z, 255 );

    mSegmenter->UpdateCommittedSegmentationDoNotRemove( segmentationLabelId, segmentationLabelColorInt4 );
}

void Segmenter::ReplaceSegmentationLabelInCommittedSegmentation2D( int oldId, int newId, Vector3^ newColor, int slice )
{
    int4 newColorInt4 = make_int4( (int)newColor->X, (int)newColor->Y, (int)newColor->Z, 255 );

    mSegmenter->ReplaceSegmentationLabelInCommittedSegmentation2D( oldId, newId, newColorInt4, slice );
}

void Segmenter::ReplaceSegmentationLabelInCommittedSegmentation3D( int oldId, int newId, Vector3^ newColor )
{
    int4 newColorInt4 = make_int4( (int)newColor->X, (int)newColor->Y, (int)newColor->Z, 255 );

    mSegmenter->ReplaceSegmentationLabelInCommittedSegmentation3D( oldId, newId, newColorInt4 );
}

void Segmenter::ReplaceSegmentationLabelInCommittedSegmentation2DConnectedComponentOnly( int oldId, int newId, Vector3^ newColor, int slice, Vector3^ seed )
{
    int4 newColorInt4 = make_int4( (int)newColor->X, (int)newColor->Y, (int)newColor->Z, 255 );
    int2 seedInt2     = make_int2( (int)seed->X,     (int)seed->Y );

    mSegmenter->ReplaceSegmentationLabelInCommittedSegmentation2DConnectedComponentOnly( oldId, newId, newColorInt4, slice, seedInt2 );
}

void Segmenter::ReplaceSegmentationLabelInCommittedSegmentation3DConnectedComponentOnly( int oldId, int newId, Vector3^ newColor, Vector3^ seed )
{
    int4 newColorInt4 = make_int4( (int)newColor->X, (int)newColor->Y, (int)newColor->Z, 255 );
    int3 seedInt3     = make_int3( (int)seed->X,     (int)seed->Y,     (int)seed->Z );

    mSegmenter->ReplaceSegmentationLabelInCommittedSegmentation3DConnectedComponentOnly( oldId, newId, newColorInt4, seedInt3 );
}

void Segmenter::UndoLastChangeToCommittedSegmentation()
{
    mSegmenter->UndoLastChangeToCommittedSegmentation();
}

void Segmenter::RedoLastChangeToCommittedSegmentation()
{
    mSegmenter->RedoLastChangeToCommittedSegmentation();
}

void Segmenter::VisualUpdateColorMap()
{
    mSegmenter->VisualUpdateColorMap();
}

void Segmenter::InitializeCostMap()
{
    mSegmenter->InitializeCostMap();
}

void Segmenter::InitializeCostMapFromPrimalMap()
{
    mSegmenter->InitializeCostMapFromPrimalMap();
}

void Segmenter::IncrementCostMapFromPrimalMapForward()
{
    mSegmenter->IncrementCostMapFromPrimalMapForward();
}

void Segmenter::IncrementCostMapFromPrimalMapBackward()
{
    mSegmenter->IncrementCostMapFromPrimalMapBackward();
}

void Segmenter::FinalizeCostMapFromPrimalMap()
{
    mSegmenter->FinalizeCostMapFromPrimalMap();
}

void Segmenter::UpdateConstraintMapAndPrimalMapFromCostMap()
{
    mSegmenter->UpdateConstraintMapAndPrimalMapFromCostMap();
}

int Segmenter::GetSegmentationLabelId( Vector3^ p )
{
    int3 pInt3 = make_int3( (int)p->X, (int)p->Y, (int)p->Z );
    return mSegmenter->GetSegmentationLabelId( pInt3 );
}

Vector3 Segmenter::GetSegmentationLabelColor( Vector3^ p )
{
    int3    pInt3                     = make_int3( (int)p->X, (int)p->Y, (int)p->Z );
    int4    segmentationLabelColor        = mSegmenter->GetSegmentationLabelColor( pInt3 );
    Vector3 segmentationLabelColorVector3 = Vector3( (float)segmentationLabelColor.x, (float)segmentationLabelColor.y, (float)segmentationLabelColor.z );

    return segmentationLabelColorVector3;
}

float Segmenter::GetPrimalValue( Vector3^ p )
{
    int3 pInt3 = make_int3( (int)p->X, (int)p->Y, (int)p->Z );

    return mSegmenter->GetPrimalValue( pInt3 );
}

void Segmenter::DumpIntermediateData()
{
    mSegmenter->DumpIntermediateData();
}

void Segmenter::DebugInitialize()
{
    mSegmenter->DebugInitialize();
}

void Segmenter::DebugUpdate()
{
    mSegmenter->DebugUpdate();
}

void Segmenter::DebugTerminate()
{
    mSegmenter->DebugTerminate();
}

ObservableDictionary< String^, ShaderResourceView^ >^ Segmenter::GetD3D11CudaTextures()
{
    return mD3D11CudaTextures;
}

bool Segmenter::IsVolumeLoaded()
{
    return mSegmenter->GetSegmenterState()->dynamicParameters.Get< bool >( "IsVolumeLoaded" );
}

VolumeDescription^ Segmenter::GetVolumeDescription()
{
    return gcnew Mojo::Interop::VolumeDescription( mSegmenter->GetSegmenterState()->volumeDescription );
}

float Segmenter::GetConvergenceGap()
{
    return mSegmenter->GetSegmenterState()->dynamicParameters.Get< float >( "ConvergenceGap" );
}

void Segmenter::SetConvergenceGap( float convergenceGap )
{
    mSegmenter->GetSegmenterState()->dynamicParameters.Set( "ConvergenceGap", convergenceGap );
}

float Segmenter::GetConvergenceGapDelta()
{
    return mSegmenter->GetSegmenterState()->dynamicParameters.Get< float >( "ConvergenceGapDelta" );
}

void Segmenter::SetConvergenceGapDelta( float convergenceGapDelta )
{
    mSegmenter->GetSegmenterState()->dynamicParameters.Set( "ConvergenceGapDelta", convergenceGapDelta );
}

float Segmenter::GetMaxForegroundCostDelta()
{
    return mSegmenter->GetSegmenterState()->dynamicParameters.Get< float >( "GetMaxForegroundCostDelta" );
}

void Segmenter::SetMaxForegroundCostDelta( float maxForegroundCostDelta )
{
    mSegmenter->GetSegmenterState()->dynamicParameters.Set( "MaxForegroundCostDelta", maxForegroundCostDelta );
}

void Segmenter::LoadD3D11CudaTextures()
{
    MOJO_FOR_EACH_KEY_VALUE( std::string key, Core::ID3D11CudaTexture* d3d11CudaTexture, mSegmenter->GetSegmenterState()->d3d11CudaTextures.GetHashMap() )
    {
        RELEASE_ASSERT( d3d11CudaTexture->GetD3D11ShaderResourceView() != NULL );

        String^ string = msclr::interop::marshal_as< String^ >( key );
        ShaderResourceView^ shaderResourceView = ShaderResourceView::FromPointer( IntPtr( d3d11CudaTexture->GetD3D11ShaderResourceView() ) );

        mD3D11CudaTextures->Set( string, shaderResourceView );
    }
}

void Segmenter::UnloadD3D11CudaTextures()
{
    for each ( Collections::Generic::KeyValuePair< String^, ShaderResourceView^ > keyValuePair in mD3D11CudaTextures )
    {
        delete keyValuePair.Value;
    }

    mD3D11CudaTextures->Internal->Clear();
}

}
}