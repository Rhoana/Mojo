#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/Thrust.hpp"
#include "Mojo.Core/SegmenterState.hpp"
#include "Mojo.Core/ID3D11CudaTexture.hpp"
#include "Mojo.Core/ForEach.hpp"

#include "Index.cuh"
#include "Math.cuh"

struct InitializeConstraintMapFromIdMapFunction
{
    enum TupleLayout
    {
        ID_MAP,
        CONSTRAINT_MAP
    };

    int    mId;
    float  mForegroundDelta;
    float  mBackgroundDelta;

    InitializeConstraintMapFromIdMapFunction( int id, float foregroundDelta, float backgroundDelta ) : mId( id ), mForegroundDelta( foregroundDelta ), mBackgroundDelta( backgroundDelta ) {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int    id                  = thrust::get< ID_MAP >( tuple );
        bool   updateConstraintMap = false;

        float  constraintMapDelta;

        if ( id == mId )
        {
            updateConstraintMap = true;
            constraintMapDelta  = mForegroundDelta;
        }
        else if ( id != 0 )
        {
            updateConstraintMap = true;
            constraintMapDelta = mBackgroundDelta;
        }

        if ( updateConstraintMap )
        {
            thrust::get< CONSTRAINT_MAP >( tuple ) = thrust::get< CONSTRAINT_MAP >( tuple ) + constraintMapDelta;
        }
    }
};

extern "C" void InitializeConstraintMapFromIdMap( Mojo::Core::SegmenterState* segmenterState, int segmentationLabelId )
{
    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int >( "IdMap" ).begin(),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin() ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int >( "IdMap" ).end(),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).end() ) ),

            InitializeConstraintMapFromIdMapFunction(
                segmentationLabelId,
                segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_INITIALIZE_FROM_ID_MAP_DELTA_FOREGROUND" ),
                segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_INITIALIZE_FROM_ID_MAP_DELTA_BACKGROUND" ) ) ) );
}


struct InitializeConstraintMapFromIdMapForSplittingFunction
{
    enum TupleLayout
    {
        ID_MAP,
        CONSTRAINT_MAP
    };

    int    mId;
    float  mBackgroundDelta;

    InitializeConstraintMapFromIdMapForSplittingFunction( int id, float backgroundDelta ) : mId( id ), mBackgroundDelta( backgroundDelta ) {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int    id                  = thrust::get< ID_MAP >( tuple );
        bool   updateConstraintMap = false;
        float  constraintMapDelta;

        if ( id != mId )
        {
            updateConstraintMap = true;
            constraintMapDelta  = mBackgroundDelta;
        }
        else
        {
            updateConstraintMap = true;
            constraintMapDelta = 0;
        }

        if ( updateConstraintMap )
        {
            thrust::get< CONSTRAINT_MAP >( tuple ) = thrust::get< CONSTRAINT_MAP >( tuple ) + constraintMapDelta;
        }
    }
};

struct UpdateConstraintMapFunction
{
    float* mConstraintMapBuffer;
    int3   mNumVoxels;
    float  mOldValue;
    float  mNeighborValue;
    float  mNewValue;

    UpdateConstraintMapFunction( float* constraintMapBuffer,
                                 int3   numVoxels,
                                 float  oldValue,
                                 float  neighborValue,
                                 float  newValue ) :
        mConstraintMapBuffer( constraintMapBuffer ),
        mNumVoxels          ( numVoxels           ),
        mOldValue           ( oldValue            ),
        mNeighborValue      ( neighborValue       ),
        mNewValue           ( newValue            )
    {}

    __device__ void operator() ( int3 index3D )
    {
        int   index1D            = Index3DToIndex1D( index3D, mNumVoxels );
        float constraintMapValue = mConstraintMapBuffer[ index1D ];

        if ( constraintMapValue == mOldValue )
        {
            int3  index3DForwardX          = make_int3( min( mNumVoxels.x - 1, index3D.x + 1 ), index3D.y,                              index3D.z );
            int3  index3DForwardY          = make_int3( index3D.x,                              min( mNumVoxels.y - 1, index3D.y + 1 ), index3D.z );
            int3  index3DBackwardX         = make_int3( max( 0, index3D.x - 1 ),                index3D.y,                              index3D.z );
            int3  index3DBackwardY         = make_int3( index3D.x,                              max( 0, index3D.y - 1 ),                index3D.z );
                                           
            int   index1DForwardX          = Index3DToIndex1D( index3DForwardX,  mNumVoxels );
            int   index1DForwardY          = Index3DToIndex1D( index3DForwardY,  mNumVoxels );
            int   index1DBackwardX         = Index3DToIndex1D( index3DBackwardX, mNumVoxels );
            int   index1DBackwardY         = Index3DToIndex1D( index3DBackwardY, mNumVoxels );

            float stencilMapValueForwardX  = mConstraintMapBuffer[ index1DForwardX  ];
            float stencilMapValueForwardY  = mConstraintMapBuffer[ index1DForwardY  ];
            float stencilMapValueBackwardX = mConstraintMapBuffer[ index1DBackwardX ];
            float stencilMapValueBackwardY = mConstraintMapBuffer[ index1DBackwardY ];

            if ( stencilMapValueForwardX  == mNeighborValue ||
                 stencilMapValueForwardY  == mNeighborValue ||
                 stencilMapValueBackwardX == mNeighborValue ||
                 stencilMapValueBackwardY == mNeighborValue )
            {
                mConstraintMapBuffer[ index1D ] = mNewValue;
            }
        }
    }
};

extern "C" void InitializeConstraintMapFromIdMapForSplitting( Mojo::Core::SegmenterState* segmenterState, int segmentationLabelId )
{
    int numElements = segmenterState->volumeDescription.numVoxels.x * segmenterState->volumeDescription.numVoxels.y * segmenterState->volumeDescription.numVoxels.z;

    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int   >( "IdMap" ).begin(),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin() ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple(
                    segmenterState->deviceVectors.Get< int   >( "IdMap" ).end(),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).end() ) ),

            InitializeConstraintMapFromIdMapForSplittingFunction(
                segmentationLabelId,
                segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_INITIALIZE_FROM_ID_MAP_DELTA_BACKGROUND" ) ) ) );

    for ( int i = 0; i < segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_SPLIT_BACKGROUND_CONTRACT_NUM_PASSES" ); i++ )
    {
        MOJO_THRUST_SAFE(
            thrust::for_each(
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator( 0 ),
                    Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),

                thrust::make_transform_iterator(
                        thrust::make_counting_iterator( numElements ),
                        Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),

                UpdateConstraintMapFunction(
                    thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float >( "ConstraintMap" )[ 0 ] ),
                    segmenterState->volumeDescription.numVoxels,
                    segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_INITIALIZE_FROM_ID_MAP_DELTA_BACKGROUND" ),
                    segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_INITIAL_VALUE" ),
                    segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_INITIAL_VALUE" ) ) ) );
    }
}

struct InitializeConstraintMapFromPrimalMapFunction
{
    enum TupleLayout
    {
        PRIMAL_MAP,
        CONSTRAINT_MAP
    };

    float  mPrimalMapThreshold;
    float  mForeground;
    float  mBackground;

    InitializeConstraintMapFromPrimalMapFunction( float primalMapThreshold, float foreground, float background ) : mPrimalMapThreshold( primalMapThreshold ), mForeground( foreground ), mBackground( background ) {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        float primal = thrust::get< PRIMAL_MAP >( tuple );
        float constraint;

        if ( primal > mPrimalMapThreshold )
        {
            constraint = mForeground;
        }
        else
        {
            constraint = mBackground;
        }

        thrust::get< CONSTRAINT_MAP >( tuple ) = constraint;
    }
};

extern "C" void InitializeConstraintMapFromPrimalMap( Mojo::Core::SegmenterState* segmenterState )
{
    if ( !segmenterState->slicesWithForegroundConstraints.empty() )
    {
        int numElementsPerSlice               = segmenterState->volumeDescription.numVoxels.x * segmenterState->volumeDescription.numVoxels.y;
        int minSliceWithForegroundConstraints = *std::min_element( segmenterState->slicesWithForegroundConstraints.begin(), segmenterState->slicesWithForegroundConstraints.end() );
        int maxSliceWithForegroundConstraints = *std::max_element( segmenterState->slicesWithForegroundConstraints.begin(), segmenterState->slicesWithForegroundConstraints.end() );

        std::vector< int > slicesWithConstraints;
        slicesWithConstraints.resize( segmenterState->slicesWithForegroundConstraints.size() + segmenterState->slicesWithBackgroundConstraints.size() );
        std::set_union( segmenterState->slicesWithForegroundConstraints.begin(),
                        segmenterState->slicesWithForegroundConstraints.end(),
                        segmenterState->slicesWithBackgroundConstraints.begin(),
                        segmenterState->slicesWithBackgroundConstraints.end(),
                        slicesWithConstraints.begin() );

        MOJO_FOR_EACH( int sliceWithConstraints, slicesWithConstraints )
        {
            if ( sliceWithConstraints >= minSliceWithForegroundConstraints && sliceWithConstraints <= maxSliceWithForegroundConstraints )
            {
                MOJO_THRUST_SAFE(
                    thrust::for_each(
                        thrust::make_zip_iterator(
                            thrust::make_tuple(
                                segmenterState->deviceVectors.Get< float >( "PrimalMap" ).begin()     + ( numElementsPerSlice * sliceWithConstraints ),
                                segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin() + ( numElementsPerSlice * sliceWithConstraints ) ) ),
                                    
                        thrust::make_zip_iterator(
                            thrust::make_tuple(
                                segmenterState->deviceVectors.Get< float >( "PrimalMap" ).begin()     + ( numElementsPerSlice * ( sliceWithConstraints + 1 ) ),
                                segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin() + ( numElementsPerSlice * ( sliceWithConstraints + 1 ) ) ) ),
                                    
                        InitializeConstraintMapFromPrimalMapFunction(
                            segmenterState->constParameters.Get< float >( "PRIMAL_MAP_THRESHOLD" ),
                            segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_HARD_FOREGROUND_AUTO" ),
                            segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_HARD_BACKGROUND_AUTO" ) ) ) );
            }
        }

        if ( minSliceWithForegroundConstraints > 0 )
        {
            MOJO_THRUST_SAFE(
                thrust::for_each(
                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            segmenterState->deviceVectors.Get< float >( "PrimalMap" ).begin()     + ( numElementsPerSlice * ( minSliceWithForegroundConstraints - 1 ) ),
                            segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin() + ( numElementsPerSlice * ( minSliceWithForegroundConstraints - 1 ) ) ) ),
                                    
                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            segmenterState->deviceVectors.Get< float >( "PrimalMap" ).begin()     + ( numElementsPerSlice * ( minSliceWithForegroundConstraints ) ),
                            segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin() + ( numElementsPerSlice * ( minSliceWithForegroundConstraints ) ) ) ),
                                    
                    InitializeConstraintMapFromPrimalMapFunction(
                        segmenterState->constParameters.Get< float >( "PRIMAL_MAP_THRESHOLD" ),
                        segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_HARD_FOREGROUND_AUTO" ),
                        segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_HARD_BACKGROUND_AUTO" ) ) ) );
        }

        if ( maxSliceWithForegroundConstraints < segmenterState->volumeDescription.numVoxels.z - 1 )
        {
            MOJO_THRUST_SAFE(
                thrust::for_each(
                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            segmenterState->deviceVectors.Get< float >( "PrimalMap" ).begin()     + ( numElementsPerSlice * ( maxSliceWithForegroundConstraints + 1 ) ),
                            segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin() + ( numElementsPerSlice * ( maxSliceWithForegroundConstraints + 1 ) ) ) ),
                                    
                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            segmenterState->deviceVectors.Get< float >( "PrimalMap" ).begin()     + ( numElementsPerSlice * ( maxSliceWithForegroundConstraints + 2 ) ),
                            segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin() + ( numElementsPerSlice * ( maxSliceWithForegroundConstraints + 2 ) ) ) ),
                                    
                    InitializeConstraintMapFromPrimalMapFunction(
                        segmenterState->constParameters.Get< float >( "PRIMAL_MAP_THRESHOLD" ),
                        segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_HARD_FOREGROUND_AUTO" ),
                        segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_HARD_BACKGROUND_AUTO" ) ) ) );
        }
    }
}


struct AddHardConstraintFunction
{
    float* mPrimalMapBuffer;
    float* mConstraintMapBuffer;
    int3   mNumVoxels;
    int3   mTopLeftFrontIndex3D;
    int2   mP1;
    int2   mP2;
    float  mRadius;
    float  mPrimalValue;
    float  mNewConstraintValue;
    float  mGaussianSigma;
    float  mMinUpdateThreshold;
    float  mForeground;
    float  mBackground;

    AddHardConstraintFunction( float* primalMapBuffer,
                               float* constraintMapBuffer,
                               int3   numVoxels,
                               int3   topLeftFrontIndex3D,
                               int2   p1,
                               int2   p2,
                               float  radius,
                               float  primalValue,
                               float  newConstraintValue,
                               float  gaussianSigma,
                               float  minUpdateThreshold,
                               float  foreground,
                               float  background ) :
        mPrimalMapBuffer     ( primalMapBuffer     ),
        mConstraintMapBuffer ( constraintMapBuffer ),
        mNumVoxels           ( numVoxels           ),
        mTopLeftFrontIndex3D ( topLeftFrontIndex3D ),
        mP1                  ( p1                  ),
        mP2                  ( p2                  ),
        mRadius              ( radius              ),
        mPrimalValue         ( primalValue         ),
        mNewConstraintValue  ( newConstraintValue  ),
        mGaussianSigma       ( gaussianSigma       ),
        mMinUpdateThreshold  ( minUpdateThreshold  ),
        mForeground          ( foreground          ),
        mBackground          ( background          )
    {}

    __device__ void operator() ( int2 index2D )
    {
        int2 currentIndex2D = make_int2( mTopLeftFrontIndex3D.x + index2D.x, mTopLeftFrontIndex3D.y + index2D.y );

        if ( currentIndex2D.x >= 0 && currentIndex2D.x < mNumVoxels.x &&
             currentIndex2D.y >= 0 && currentIndex2D.y < mNumVoxels.y )
        {
            int   currentIndex1D     = ( mNumVoxels.x * mNumVoxels.y * mTopLeftFrontIndex3D.z ) + ( mNumVoxels.x * currentIndex2D.y ) + currentIndex2D.x;
            float oldConstraintValue = mConstraintMapBuffer[ currentIndex1D ];

            if ( oldConstraintValue <= 0.0f || mNewConstraintValue > 0.0f )
            {
                int2   p1ToP2               = mP2 - mP1;
                int2   p1ToCurrent          = currentIndex2D - mP1;
                int2   p2ToP1               = mP1 - mP2;
                int2   p2ToCurrent          = currentIndex2D - mP2;

                float  distance             = 0.0f;
                float  p1P2Length           = sqrt( ( (float)p1ToP2.x * (float)p1ToP2.x ) + ( (float)p1ToP2.y * (float)p1ToP2.y ) );
                float  p1ToCurrentLength    = sqrt( ( (float)p1ToCurrent.x * (float)p1ToCurrent.x ) + ( (float)p1ToCurrent.y * (float)p1ToCurrent.y ) );
                float  p2ToCurrentLength    = sqrt( ( (float)p2ToCurrent.x * (float)p2ToCurrent.x ) + ( (float)p2ToCurrent.y * (float)p2ToCurrent.y ) );

                if ( p1P2Length < 0.1f )
                {
                    distance = min( p1ToCurrentLength, p2ToCurrentLength );
                }
                else
                {
                    float p1ToCurrentDotP1ToP2 = ( p1ToCurrent.x * p1ToP2.x ) + ( p1ToCurrent.y * p1ToP2.y );
                    float p2ToCurrentDotP2ToP1 = ( p2ToCurrent.x * p2ToP1.x ) + ( p2ToCurrent.y * p2ToP1.y );

                    float scalarProjectionP1ToCurrentOntoP1ToP2 = p1ToCurrentDotP1ToP2 / p1P2Length;
                    float scalarProjectionP2ToCurrentOntoP2ToP1 = p2ToCurrentDotP2ToP1 / p1P2Length;

                    if ( scalarProjectionP1ToCurrentOntoP1ToP2 > 0.0f && scalarProjectionP2ToCurrentOntoP2ToP1 > 0.0f )
                    {
                        if ( p1ToCurrentLength > scalarProjectionP1ToCurrentOntoP1ToP2 )
                        {
                            distance = sqrt( ( p1ToCurrentLength * p1ToCurrentLength ) - ( scalarProjectionP1ToCurrentOntoP1ToP2 * scalarProjectionP1ToCurrentOntoP1ToP2 ) );
                        }
                        else
                        {
                            distance = 0;
                        }
                    }
                    else
                    if ( scalarProjectionP1ToCurrentOntoP1ToP2 > 0.0f && scalarProjectionP2ToCurrentOntoP2ToP1 < 0.0f )
                    {
                        distance = p2ToCurrentLength;
                    }
                    else
                    if ( scalarProjectionP1ToCurrentOntoP1ToP2 < 0.0f && scalarProjectionP2ToCurrentOntoP2ToP1 > 0.0f )
                    {
                        distance = p1ToCurrentLength;
                    }
                    else
                    {
                        distance = min( p1ToCurrentLength, p2ToCurrentLength );
                    }
                }

                if ( distance <= mRadius )
                {
                    float normalizedDistance = distance / mRadius;
                    float sigma              = mGaussianSigma;
                    float alpha              = mNewConstraintValue;
                    float beta               = - ( ( normalizedDistance * normalizedDistance ) / ( 2.0f * sqr( sigma ) ) );
                    float gaussian           = alpha * powf( 2.71f, beta );

                    if ( abs( gaussian ) > mMinUpdateThreshold )
                    {
                        mConstraintMapBuffer[ currentIndex1D ] = min( mBackground, max( mForeground, oldConstraintValue + gaussian ) );
                        mPrimalMapBuffer[ currentIndex1D ]     = mPrimalValue;
                    }
                }
            }
        }
    }
};

extern "C" void AddHardConstraint( Mojo::Core::SegmenterState* segmenterState, int3 p1, int3 p2, float radius, float constraintValue, float primalValue, float gaussianSigma )
{
    RELEASE_ASSERT( p1.z == p2.z );

    int minIndexX = min( p1.x + (int)floor( -radius ), p2.x + (int)floor( -radius ) ) - 1;
    int maxIndexX = max( p1.x +   (int)ceil( radius ), p2.x +   (int)ceil( radius ) ) + 1;

    int minIndexY = min( p1.y + (int)floor( -radius ), p2.y + (int)floor( -radius ) ) - 1;
    int maxIndexY = max( p1.y +   (int)ceil( radius ), p2.y +   (int)ceil( radius ) ) + 1;

    int minIndexZ = min( p1.z, p2.z );

    int numPixelsX = maxIndexX - minIndexX;
    int numPixelsY = maxIndexY - minIndexY;

    int3 topLeftFrontIndex3D = make_int3( minIndexX, minIndexY, minIndexZ );

    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_transform_iterator(
                thrust::make_counting_iterator( 0 ),
                Index1DToIndex2DFunction( numPixelsX ) ),

            thrust::make_transform_iterator(
                thrust::make_counting_iterator( numPixelsX * numPixelsY ),
                Index1DToIndex2DFunction( numPixelsX ) ),

            AddHardConstraintFunction(
                thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float >( "PrimalMap" )[ 0 ] ),
                thrust::raw_pointer_cast( &segmenterState->deviceVectors.Get< float >( "ConstraintMap" )[ 0 ] ),
                segmenterState->volumeDescription.numVoxels,
                topLeftFrontIndex3D,
                make_int2( p1.x, p1.y ),
                make_int2( p2.x, p2.y ),
                radius,
                primalValue,
                constraintValue,
                segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_FALLOFF_GAUSSIAN_SIGMA" ),
                segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_MIN_UPDATE_THRESHOLD" ),
                segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_HARD_FOREGROUND_USER" ),
                segmenterState->constParameters.Get< float >( "CONSTRAINT_MAP_HARD_BACKGROUND_USER" ) ) ) );
}


texture< float, 3, cudaReadModeElementType > TEXTURE_CONSTRAINT_MAP_FLOAT_3D_ELEMENT_TYPE;

struct DilateConstraintMapFunction
{
    enum TupleLayout
    {
        INDEX_3D,
        CONSTRAINT_MAP,
    };

    DilateConstraintMapFunction() {}

    template < typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        int3  index3D                          = thrust::get< INDEX_3D >( tuple );

        float imageValue                       = tex3D( TEXTURE_CONSTRAINT_MAP_FLOAT_3D_ELEMENT_TYPE, index3D.x,        index3D.y        , index3D.z );
        float imageXForward                    = tex3D( TEXTURE_CONSTRAINT_MAP_FLOAT_3D_ELEMENT_TYPE, index3D.x + 1.0f, index3D.y        , index3D.z );
        float imageYForward                    = tex3D( TEXTURE_CONSTRAINT_MAP_FLOAT_3D_ELEMENT_TYPE, index3D.x       , index3D.y + 1.0f , index3D.z );
        float imageXBackward                   = tex3D( TEXTURE_CONSTRAINT_MAP_FLOAT_3D_ELEMENT_TYPE, index3D.x - 1.0f, index3D.y        , index3D.z );
        float imageYBackward                   = tex3D( TEXTURE_CONSTRAINT_MAP_FLOAT_3D_ELEMENT_TYPE, index3D.x       , index3D.y - 1.0f , index3D.z );

        thrust::get< CONSTRAINT_MAP >( tuple ) = max( max( max( max( imageValue, imageXForward ), imageYForward ), imageXBackward ), imageYBackward );
    }
};

extern "C" void DilateConstraintMap( Mojo::Core::SegmenterState* segmenterState )
{
    segmenterState->d3d11CudaTextures.MapCudaArrays();

    cudaArray* constraintsArray = segmenterState->d3d11CudaTextures.Get( "ConstraintMap" )->GetMappedCudaArray();

    TEXTURE_CONSTRAINT_MAP_FLOAT_3D_ELEMENT_TYPE.normalized       = false;
    TEXTURE_CONSTRAINT_MAP_FLOAT_3D_ELEMENT_TYPE.filterMode       = cudaFilterModePoint;
    TEXTURE_CONSTRAINT_MAP_FLOAT_3D_ELEMENT_TYPE.addressMode[ 0 ] = cudaAddressModeClamp;
    TEXTURE_CONSTRAINT_MAP_FLOAT_3D_ELEMENT_TYPE.addressMode[ 1 ] = cudaAddressModeClamp;
    TEXTURE_CONSTRAINT_MAP_FLOAT_3D_ELEMENT_TYPE.addressMode[ 2 ] = cudaAddressModeClamp;

    int numElements = segmenterState->volumeDescription.numVoxels.x * segmenterState->volumeDescription.numVoxels.y * segmenterState->volumeDescription.numVoxels.z;

    Mojo::Core::Cuda::BindTextureReferenceToArray( &TEXTURE_CONSTRAINT_MAP_FLOAT_3D_ELEMENT_TYPE, constraintsArray );

    MOJO_THRUST_SAFE(
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( 0 ),
                        Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).begin() ) ),

            thrust::make_zip_iterator(
                thrust::make_tuple( 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator( numElements ),
                        Index1DToIndex3DFunction( segmenterState->volumeDescription.numVoxels ) ),
                    segmenterState->deviceVectors.Get< float >( "ConstraintMap" ).end() ) ),
              
            DilateConstraintMapFunction() ) );

    segmenterState->d3d11CudaTextures.UnmapCudaArrays();
}
