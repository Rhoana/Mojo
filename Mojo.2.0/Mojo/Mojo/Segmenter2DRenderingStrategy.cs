using System.Collections.Generic;
using System.Diagnostics;
using Mojo.Interop;
using SlimDX;
using SlimDX.Direct3D11;
using SlimDX.DXGI;
using TinyText;

namespace Mojo
{
    public class Segmenter2DRenderingStrategy : NotifyPropertyChanged, IRenderingStrategy
    {
        private const int POSITION_SLOT = 0;
        private const int TEXCOORD_SLOT = 1;
        private const int POSITION_NUM_BYTES_PER_COMPONENT = 4;
        private const int POSITION_NUM_COMPONENTS_PER_VERTEX = 2;
        private const int TEXCOORD_NUM_BYTES_PER_COMPONENT = 4;
        private const int TEXCOORD_NUM_COMPONENTS_PER_VERTEX = 2;
        private const int NUM_VERTICES = 4;
        private const int MAX_NUM_TEXT_CHARACTERS = 1024;

        private const Format POSITION_FORMAT = Format.R32G32_Float;
        private const Format TEXCOORD_FORMAT = Format.R32G32_Float;

        private static readonly Color4 CLEAR_COLOR = new Color4( 0.5f, 0.5f, 1.0f );

        private static readonly Vector2 POSITION_TOP_LEFT = new Vector2( -1f, 1f );
        private static readonly Vector2 POSITION_BOTTOM_LEFT = new Vector2( -1f, -1f );
        private static readonly Vector2 POSITION_TOP_RIGHT = new Vector2( 1f, 1f );
        private static readonly Vector2 POSITION_BOTTOM_RIGHT = new Vector2( 1f, -1f );

        private static readonly Vector2 TEXCOORD_TOP_LEFT = new Vector2( 0f, 0f );
        private static readonly Vector2 TEXCOORD_BOTTOM_LEFT = new Vector2( 0f, 1f );
        private static readonly Vector2 TEXCOORD_TOP_RIGHT = new Vector2( 1f, 0f );
        private static readonly Vector2 TEXCOORD_BOTTOM_RIGHT = new Vector2( 1f, 1f );

        private static readonly IDictionary< string, int > D3D11_CUDA_TEXTURE_RENDERING_MODE_MAP = new Dictionary<string, int>
                                                                                                   {
                                                                                                       { "Default", -2 },
                                                                                                       { "SourceMapDoNotShowSegmentation", -1 },
                                                                                                       { "ColorMap", 0 },
                                                                                                       { "ConstraintMap", 1 },
                                                                                                       { "SourceMap", 2 },
                                                                                                       { "CorrespondenceMap", 3 },
                                                                                                   };

        private static readonly IDictionary< ShaderResourceViewDimension, int > TEXTURE_DIMENSIONS_MAP =
            new Dictionary<ShaderResourceViewDimension, int>
            {
                { ShaderResourceViewDimension.Unknown, -1 },
                { ShaderResourceViewDimension.Texture2D, 2 },
                { ShaderResourceViewDimension.Texture3D, 3 }
            };

        private static readonly IDictionary< ShaderResourceViewDimension, string > TEXTURE_DIMENSIONS_NAME_MAP =
            new Dictionary< ShaderResourceViewDimension, string>
            {
                { ShaderResourceViewDimension.Unknown,   "Unknown" },
                { ShaderResourceViewDimension.Texture2D, "gCurrentTexture2D" },
                { ShaderResourceViewDimension.Texture3D, "gCurrentTexture3D" }
            };

        private readonly Stopwatch mStopwatch = new Stopwatch();

        private Effect mEffect;
        private readonly EffectPass mPass;
        private InputLayout mInputLayout;
        private Buffer mPositionVertexBuffer;
        private Buffer mTexCoordVertexBuffer;

        private Context mTinyTextContext;

        private Segmenter mSegmenter;

        private string CurrentTextureName
        {
            get
            {
                return TEXTURE_DIMENSIONS_NAME_MAP.ContainsKey( mSegmenter.D3D11CudaTextureEnumerator.Current.Value.Description.Dimension )
                           ? TEXTURE_DIMENSIONS_NAME_MAP[ mSegmenter.D3D11CudaTextureEnumerator.Current.Value.Description.Dimension ]
                           : TEXTURE_DIMENSIONS_NAME_MAP[ ShaderResourceViewDimension.Unknown ];
            }
        }

        private int CurrentTextureIndex
        {
            get
            {
                if ( !mSegmenter.ShowSegmentation )
                {
                    return D3D11_CUDA_TEXTURE_RENDERING_MODE_MAP[ "SourceMapDoNotShowSegmentation" ];
                }

                return D3D11_CUDA_TEXTURE_RENDERING_MODE_MAP.ContainsKey( mSegmenter.D3D11CudaTextureEnumerator.Current.Key )
                            ? D3D11_CUDA_TEXTURE_RENDERING_MODE_MAP[ mSegmenter.D3D11CudaTextureEnumerator.Current.Key ]
                            : D3D11_CUDA_TEXTURE_RENDERING_MODE_MAP[ "Default" ];
            }
        }

        private int CurrentTextureDimensions
        {
            get
            {
                return TEXTURE_DIMENSIONS_MAP.ContainsKey( mSegmenter.D3D11CudaTextureEnumerator.Current.Value.Description.Dimension )
                           ? TEXTURE_DIMENSIONS_MAP[ mSegmenter.D3D11CudaTextureEnumerator.Current.Value.Description.Dimension ]
                           : TEXTURE_DIMENSIONS_MAP[ ShaderResourceViewDimension.Unknown ];
            }
        }

        private int SplitMode
        {
            get
            {
                return mSegmenter.CurrentSegmenterToolMode == SegmenterToolMode.Split && mSegmenter.SplitSegmentationLabel != null ? 1 : 0;
            }
        }

        private Vector3 CurrentSegmentationLabelColor
        {
            get
            {
                return mSegmenter.CurrentSegmentationLabel != null
                           ? mSegmenter.CurrentSegmentationLabel.Color * ( 1.0f / 255.0f )
                           : Constants.NULL_SEGMENTATION_LABEL.Color;
            }
        }

        private Vector3 SplitSegmentationLabelColor
        {
            get
            {
                return mSegmenter.SplitSegmentationLabel != null
                           ? mSegmenter.SplitSegmentationLabel.Color * ( 1.0f / 255.0f )
                           : Constants.NULL_SEGMENTATION_LABEL.Color;
            }
        }

        private string FrameTimeString
        {
            get
            {
                return mStopwatch.ElapsedMilliseconds == 0 ? "< 1 ms" : mStopwatch.ElapsedMilliseconds + " ms";
            }
        }

        private string CurrentSegmentationLabelString
        {
            get
            {
                return mSegmenter.CurrentSegmentationLabel != null ? (string)mSegmenter.CurrentSegmentationLabel.Name : "null";
            }
        }

        public float CurrentSliceCoordinate
        {
            get
            {
                return mSegmenter.CurrentSlice / (float)( mSegmenter.Internal.GetVolumeDescription().NumVoxelsZ - 1 );
            }
        }

        public Segmenter2DRenderingStrategy( SlimDX.Direct3D11.Device device, DeviceContext deviceContext, Segmenter segmenter )
        {
            mSegmenter = segmenter;

            mStopwatch.Start();

            mEffect = EffectUtil.CompileEffect( device, @"Shaders\Segmenter2D.fx" );

            // create position vertex data, making sure to rewind the stream afterward
            var positionVertexDataStream = new DataStream( NUM_VERTICES * POSITION_NUM_COMPONENTS_PER_VERTEX * POSITION_NUM_BYTES_PER_COMPONENT, true, true );

            positionVertexDataStream.Write( POSITION_TOP_LEFT );
            positionVertexDataStream.Write( POSITION_BOTTOM_LEFT );
            positionVertexDataStream.Write( POSITION_TOP_RIGHT );
            positionVertexDataStream.Write( POSITION_BOTTOM_RIGHT );
            positionVertexDataStream.Position = 0;

            // create texcoord vertex data, making sure to rewind the stream afterward
            var texCoordVertexDataStream = new DataStream( NUM_VERTICES * TEXCOORD_NUM_COMPONENTS_PER_VERTEX * TEXCOORD_NUM_BYTES_PER_COMPONENT, true, true );

            texCoordVertexDataStream.Write( TEXCOORD_TOP_LEFT );
            texCoordVertexDataStream.Write( TEXCOORD_BOTTOM_LEFT );
            texCoordVertexDataStream.Write( TEXCOORD_TOP_RIGHT );
            texCoordVertexDataStream.Write( TEXCOORD_BOTTOM_RIGHT );
            texCoordVertexDataStream.Position = 0;

            // create the input layout
            var inputElements = new[]
                                {
                                    new InputElement( "POSITION", 0, POSITION_FORMAT, POSITION_SLOT ),
                                    new InputElement( "TEXCOORD", 0, TEXCOORD_FORMAT, TEXCOORD_SLOT )
                                };

            var technique = mEffect.GetTechniqueByName( "Segmenter2D" );
            mPass = technique.GetPassByName( "Segmenter2D" );

            mInputLayout = new InputLayout( device, mPass.Description.Signature, inputElements );

            // create the vertex buffers
            mPositionVertexBuffer = new Buffer( device,
                                                positionVertexDataStream,
                                                NUM_VERTICES * POSITION_NUM_COMPONENTS_PER_VERTEX * POSITION_NUM_BYTES_PER_COMPONENT,
                                                ResourceUsage.Default,
                                                BindFlags.VertexBuffer,
                                                CpuAccessFlags.None,
                                                ResourceOptionFlags.None,
                                                0 );

            mTexCoordVertexBuffer = new Buffer( device,
                                                texCoordVertexDataStream,
                                                NUM_VERTICES * TEXCOORD_NUM_COMPONENTS_PER_VERTEX * TEXCOORD_NUM_BYTES_PER_COMPONENT,
                                                ResourceUsage.Default,
                                                BindFlags.VertexBuffer,
                                                CpuAccessFlags.None,
                                                ResourceOptionFlags.None,
                                                0 );

            System.Threading.Thread.Sleep( 1000 );
            Console.WriteLine( "\nMojo initializing TinyText.Context (NOTE: TinyText.Context generates D3D11 warnings)...\n" );

            bool result;
            mTinyTextContext = new Context( device, deviceContext, MAX_NUM_TEXT_CHARACTERS, out result );
            Release.Assert( result );

            Console.WriteLine( "\nMojo finished Initializing TinyText.Context...\n" );
        }

        public void Dispose()
        {
            if ( mTexCoordVertexBuffer != null )
            {
                mTexCoordVertexBuffer.Dispose();
                mTexCoordVertexBuffer = null;
            }

            if ( mPositionVertexBuffer != null )
            {
                mPositionVertexBuffer.Dispose();
                mPositionVertexBuffer = null;
            }

            if ( mInputLayout != null )
            {
                mInputLayout.Dispose();
                mInputLayout = null;
            }

            if ( mEffect != null )
            {
                mEffect.Dispose();
                mEffect = null;
            }

            if ( mTinyTextContext != null )
            {
                mTinyTextContext.Dispose();
                mTinyTextContext = null;
            }

            if ( mSegmenter != null )
            {
                mSegmenter.Dispose();
                mSegmenter = null;
            }
        }

        public void Render( DeviceContext deviceContext, Viewport viewport, RenderTargetView renderTargetView, DepthStencilView depthStencilView )
        {
            deviceContext.ClearRenderTargetView( renderTargetView, CLEAR_COLOR );

            if ( mSegmenter.DatasetLoaded )
            {
                deviceContext.InputAssembler.InputLayout = mInputLayout;
                deviceContext.InputAssembler.PrimitiveTopology = PrimitiveTopology.TriangleStrip;

                deviceContext.InputAssembler.SetVertexBuffers( POSITION_SLOT,
                                                                new VertexBufferBinding( mPositionVertexBuffer,
                                                                                        POSITION_NUM_COMPONENTS_PER_VERTEX *
                                                                                        POSITION_NUM_BYTES_PER_COMPONENT,
                                                                                        0 ) );

                deviceContext.InputAssembler.SetVertexBuffers( TEXCOORD_SLOT,
                                                                new VertexBufferBinding( mTexCoordVertexBuffer,
                                                                                        TEXCOORD_NUM_COMPONENTS_PER_VERTEX *
                                                                                        TEXCOORD_NUM_BYTES_PER_COMPONENT,
                                                                                        0 ) );

                mEffect.GetVariableByName( CurrentTextureName ).AsResource().SetResource( mSegmenter.D3D11CudaTextureEnumerator.Current.Value );
                mEffect.GetVariableByName( "gSourceMapTexture3D" ).AsResource().SetResource( mSegmenter.Internal.GetD3D11CudaTextures().Get( "SourceMap" ) );
                mEffect.GetVariableByName( "gPrimalMapTexture3D" ).AsResource().SetResource( mSegmenter.Internal.GetD3D11CudaTextures().Get( "PrimalMap" ) );
                mEffect.GetVariableByName( "gColorMapTexture3D" ).AsResource().SetResource( mSegmenter.Internal.GetD3D11CudaTextures().Get( "ColorMap" ) );
                mEffect.GetVariableByName( "gConstraintMapTexture3D" ).AsResource().SetResource( mSegmenter.Internal.GetD3D11CudaTextures().Get( "ConstraintMap" ) );
                mEffect.GetVariableByName( "gPrimalMapThreshold" ).AsScalar().Set( (float)Constants.ConstParameters.GetFloat( "PRIMAL_MAP_THRESHOLD" ) );
                mEffect.GetVariableByName( "gRecordingMode" ).AsScalar().Set( (int)Constants.RECORDING_MODE );
                mEffect.GetVariableByName( "gSplitMode" ).AsScalar().Set( SplitMode );
                mEffect.GetVariableByName( "gCurrentSliceCoordinate" ).AsScalar().Set( CurrentSliceCoordinate );
                mEffect.GetVariableByName( "gCurrentTextureIndex" ).AsScalar().Set( CurrentTextureIndex );
                mEffect.GetVariableByName( "gCurrentTextureDimensions" ).AsScalar().Set( CurrentTextureDimensions );
                mEffect.GetVariableByName( "gCurrentSegmentationLabelColor" ).AsVector().Set( CurrentSegmentationLabelColor );
                mEffect.GetVariableByName( "gSplitSegmentationLabelColor" ).AsVector().Set( SplitSegmentationLabelColor );

                mPass.Apply( deviceContext );
                deviceContext.Draw( NUM_VERTICES, 0 );

                if ( Constants.RECORDING_MODE == RecordingMode.NotRecording )
                {
                    mTinyTextContext.Print( viewport, "Current Z Slice Coordinate: " + CurrentSliceCoordinate + " (slice " + mSegmenter.CurrentSlice + ")", 10, 10 );
                    mTinyTextContext.Print( viewport, "Current Texture: " + mSegmenter.D3D11CudaTextureEnumerator.Current.Key, 10, 30 );
                    mTinyTextContext.Print( viewport, "Frame Time: " + FrameTimeString, 10, 50 );
                    mTinyTextContext.Print( viewport, "Primal Dual Energy Gap: " + mSegmenter.Internal.GetConvergenceGap(), 10, 70 );
                    mTinyTextContext.Print( viewport, "Primal Dual Energy Gap Delta: " + mSegmenter.Internal.GetConvergenceGapDelta(), 10, 90 );
                    mTinyTextContext.Print( viewport, "Current Segmentation Label: " + CurrentSegmentationLabelString, 10, 110 );
                    mTinyTextContext.Print( viewport, "Segmenter Dimension Mode: " + mSegmenter.DimensionMode, 10, 130 );
                    mTinyTextContext.Print( viewport, "Segmenter Max Foreground Cost Delta: " + mSegmenter.Internal.GetMaxForegroundCostDelta(), 10, 150 );
                    mTinyTextContext.Render();
                }

                mStopwatch.Reset();
                mStopwatch.Start();
            }
            else
            {
                mTinyTextContext.Print( viewport, "No dataset loaded.", 10, 10 );
                mTinyTextContext.Render();
            }
        }
    }
}
