#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/Thrust.hpp"
#include "Mojo.Core/SegmenterState.hpp"
#include "Mojo.Core/ID3D11CudaTexture.hpp"

#include "Index.cuh"

struct UpdateCommittedSegmentationFunction
{
    enum TupleLayout
    {
        ID_MAP,
        COLOR_MAP,
        PRIMAL_MAP
    };

    int    mId,    mIdMapInitialValue;
    uchar4 mColor, mColorMapInitialValue;
    float  mPrimalMapThreshold;

    UpdateCommittedSegmentationFunction( int    id,
                                         uchar4 color,
                                         int    idMapInitialValue,
                                         uchar4 colorMapInitialValue,
                                         float  primalMapThreshold ) :
        mId                         ( id                          ),
        mColor                      ( color                       ),
        mIdMapInitialValue          ( idMapInitialValue           ),
        mColorMapInitialValue       ( colorMapInitialValue        ),
        mPrimalMapThreshold         ( primalMapThreshold          ) {};

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        float  primal             = thrust::get< PRIMAL_MAP >( tuple );
        bool   commitSegmentation = false;
        int    newId;
        uchar4 newColor;


        if ( primal > mPrimalMapThreshold )
        {
            commitSegmentation = true;
            newId              = mId;
            newColor           = mColor;
        }
        else
        {
            int id = thrust::get< ID_MAP >( tuple );

            if ( id == mId )
            {
                commitSegmentation = true;
                newId              = mIdMapInitialValue;
                newColor           = mColorMapInitialValue;
            }
        }

        if ( commitSegmentation )
        {
            thrust::get< ID_MAP    >( tuple ) = newId;
            thrust::get< COLOR_MAP >( tuple ) = newColor;
        }
    }
};

extern "C" void UpdateCommittedSegmentation( Mojo::Core::SegmenterState* segmenterState, int id, uchar4 color )
{
    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int    >( "IdMap"     ).begin(),
                    segmenterState->deviceVectors.Get< uchar4 >( "ColorMap"  ).begin(),
                    segmenterState->deviceVectors.Get< float  >( "PrimalMap" ).begin() ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int    >( "IdMap"     ).end(),
                    segmenterState->deviceVectors.Get< uchar4 >( "ColorMap"  ).end(),
                    segmenterState->deviceVectors.Get< float  >( "PrimalMap" ).end() ) ),

            UpdateCommittedSegmentationFunction(
                id,
                color,
                segmenterState->constParameters.Get< int    >( "ID_MAP_INITIAL_VALUE"    ),
                segmenterState->constParameters.Get< uchar4 >( "COLOR_MAP_INITIAL_VALUE" ),
                segmenterState->constParameters.Get< float  >( "PRIMAL_MAP_THRESHOLD"    ) ) ) );
}


struct UpdateCommittedSegmentationDoNotRemoveFunction
{
    enum TupleLayout
    {
        ID_MAP,
        COLOR_MAP,
        PRIMAL_MAP
    };

    int    mId;
    uchar4 mColor;
    float  mPrimalMapThreshold;

    UpdateCommittedSegmentationDoNotRemoveFunction( int    id,
                                                    uchar4 color,
                                                    float  primalMapThreshold ) :
        mId                         ( id                 ),
        mColor                      ( color              ),
        mPrimalMapThreshold         ( primalMapThreshold ) {};

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        float primal = thrust::get< PRIMAL_MAP >( tuple );

        if ( primal > mPrimalMapThreshold )
        {
            thrust::get< ID_MAP    >( tuple ) = mId;
            thrust::get< COLOR_MAP >( tuple ) = mColor;
        }
    }
};

extern "C" void UpdateCommittedSegmentationDoNotRemove( Mojo::Core::SegmenterState* segmenterState, int id, uchar4 color )
{
    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int    >( "IdMap"     ).begin(),
                    segmenterState->deviceVectors.Get< uchar4 >( "ColorMap"  ).begin(),
                    segmenterState->deviceVectors.Get< float  >( "PrimalMap" ).begin() ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int    >( "IdMap"     ).end(),
                    segmenterState->deviceVectors.Get< uchar4 >( "ColorMap"  ).end(),
                    segmenterState->deviceVectors.Get< float  >( "PrimalMap" ).end() ) ),

            UpdateCommittedSegmentationDoNotRemoveFunction(
                id,
                color,
                segmenterState->constParameters.Get< float  >( "PRIMAL_MAP_THRESHOLD" ) ) ) );
}


struct ReplaceSegmentationLabelInCommittedSegmentationFunction
{
    enum TupleLayout
    {
        ID_MAP,
        COLOR_MAP
    };

    int    mOldId, mNewId;
    uchar4 mNewColor;

    ReplaceSegmentationLabelInCommittedSegmentationFunction( int oldId, int newId, uchar4 newColor ) : mOldId( oldId ), mNewId( newId ), mNewColor( newColor ) {};

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int oldId = thrust::get< ID_MAP >( tuple );

        if ( oldId == mOldId )
        {
            thrust::get< ID_MAP    >( tuple ) = mNewId;
            thrust::get< COLOR_MAP >( tuple ) = mNewColor;
        }
    }
};

extern "C" void ReplaceSegmentationLabelInCommittedSegmentation2D( Mojo::Core::SegmenterState* segmenterState, int oldId, int newId, uchar4 newColor, int slice )
{
    int numElementsPerSlice = segmenterState->volumeDescription.numVoxels.x * segmenterState->volumeDescription.numVoxels.y;

    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int    >( "IdMap"    ).begin() + ( numElementsPerSlice * ( slice + 0 ) ),
                    segmenterState->deviceVectors.Get< uchar4 >( "ColorMap" ).begin() + ( numElementsPerSlice * ( slice + 0 ) ) ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int    >( "IdMap"    ).begin() + ( numElementsPerSlice * ( slice + 1 ) ),
                    segmenterState->deviceVectors.Get< uchar4 >( "ColorMap" ).begin() + ( numElementsPerSlice * ( slice + 1 ) ) ) ),

            ReplaceSegmentationLabelInCommittedSegmentationFunction( oldId, newId, newColor ) ) );
}

extern "C" void ReplaceSegmentationLabelInCommittedSegmentation3D( Mojo::Core::SegmenterState* segmenterState, int oldId, int newId, uchar4 newColor )
{
    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int    >( "IdMap"    ).begin(),
                    segmenterState->deviceVectors.Get< uchar4 >( "ColorMap" ).begin() ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int    >( "IdMap"    ).end(),
                    segmenterState->deviceVectors.Get< uchar4 >( "ColorMap" ).end() ) ),

            ReplaceSegmentationLabelInCommittedSegmentationFunction( oldId, newId, newColor ) ) );
}


struct InitializeSegmentationAndRemoveFromCommittedSegmentationFunction
{
    enum TupleLayout
    {
        ID_MAP,
        COLOR_MAP,
        PRIMAL_MAP
    };

    int    mId, mIdMapInitialValue;
    uchar4 mColorMapInitialValue;
    float  mPrimalMapForegroundValue, mPrimalMapBackgroundValue;

    InitializeSegmentationAndRemoveFromCommittedSegmentationFunction( int    id,
                                                                      int    idMapInitialValue,
                                                                      uchar4 colorMapInitialValue,
                                                                      float  primalMapForegroundValue,
                                                                      float  primalMapBackgroundValue ) :
        mId                      ( id                       ),
        mIdMapInitialValue       ( idMapInitialValue        ),
        mColorMapInitialValue    ( colorMapInitialValue     ),
        mPrimalMapForegroundValue( primalMapForegroundValue ),
        mPrimalMapBackgroundValue( primalMapBackgroundValue ) {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int id = thrust::get< ID_MAP >( tuple );
        float newPrimalValue;

        if ( id == mId )
        {
            thrust::get< ID_MAP    >( tuple ) = mIdMapInitialValue;
            thrust::get< COLOR_MAP >( tuple ) = mColorMapInitialValue;
            newPrimalValue                    = mPrimalMapForegroundValue;
        }
        else
        {
            newPrimalValue                    = mPrimalMapBackgroundValue;
        }

        thrust::get< PRIMAL_MAP >( tuple )    = newPrimalValue;
    }
};

extern "C" void InitializeSegmentationAndRemoveFromCommittedSegmentation( Mojo::Core::SegmenterState* segmenterState, int segmentationLabelId )
{
    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int    >( "IdMap"     ).begin(),
                    segmenterState->deviceVectors.Get< uchar4 >( "ColorMap"  ).begin(),
                    segmenterState->deviceVectors.Get< float  >( "PrimalMap" ).begin() ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int    >( "IdMap"     ).end(),
                    segmenterState->deviceVectors.Get< uchar4 >( "ColorMap"  ).end(),
                    segmenterState->deviceVectors.Get< float  >( "PrimalMap" ).end() ) ),

            InitializeSegmentationAndRemoveFromCommittedSegmentationFunction(
                segmentationLabelId,
                segmenterState->constParameters.Get< int    >( "ID_MAP_INITIAL_VALUE"    ),
                segmenterState->constParameters.Get< uchar4 >( "COLOR_MAP_INITIAL_VALUE" ),
                segmenterState->constParameters.Get< float  >( "PRIMAL_MAP_FOREGROUND"   ),
                segmenterState->constParameters.Get< float  >( "PRIMAL_MAP_BACKGROUND"   ) ) ) );
}
