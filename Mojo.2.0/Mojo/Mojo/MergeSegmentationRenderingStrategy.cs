using System.Diagnostics;
using System.Linq;
using Mojo.Interop;
using SlimDX;
using SlimDX.DXGI;
using SlimDX.Direct3D11;
using TinyText;

namespace Mojo
{
    public class MergeSegmentationRenderingStrategy : NotifyPropertyChanged, IRenderingStrategy
    {
        private const int POSITION_SLOT = 0;
        private const int POSITION_NUM_BYTES_PER_COMPONENT = 4;
        private const int POSITION_NUM_COMPONENTS_PER_VERTEX = 3;

        private const int TEXCOORD_SLOT = 1;
        private const int TEXCOORD_NUM_BYTES_PER_COMPONENT = 4;
        private const int TEXCOORD_NUM_COMPONENTS_PER_VERTEX = 3;

        private const int QUAD_NUM_VERTICES = 4;

        private const Format POSITION_FORMAT = Format.R32G32B32_Float;
        private const Format TEXCOORD_FORMAT = Format.R32G32B32_Float;

        private readonly Stopwatch mStopwatch = new Stopwatch();

        private Effect mEffect;
        private readonly EffectPass mPass;
        private InputLayout mInputLayout;
        private Buffer mPositionVertexBuffer;
        private Buffer mTexCoordVertexBuffer;

        private Context mTinyTextContext;

        private DebugRenderer mDebugRenderer;
        private TileManager mTileManager;

        private string FrameTimeString
        {
            get
            {
                return mStopwatch.ElapsedMilliseconds == 0 ? "< 1 ms" : mStopwatch.ElapsedMilliseconds + " ms";
            }
        }

        public MergeSegmentationRenderingStrategy( SlimDX.Direct3D11.Device device, DeviceContext deviceContext, TileManager tileManager )
        {
            mTileManager = tileManager;
            mDebugRenderer = new DebugRenderer( device );

            mEffect = EffectUtil.CompileEffect( device, @"Shaders\MergeRenderer2D.fx" );

            var positionTexcoordInputElements = new[]
                                                {
                                                    new InputElement( "POSITION", 0, POSITION_FORMAT, POSITION_SLOT ),
                                                    new InputElement( "TEXCOORD", 0, TEXCOORD_FORMAT, TEXCOORD_SLOT )
                                                };

            EffectTechnique effectTechnique = mEffect.GetTechniqueByName( "TileManager2D" );
            mPass = effectTechnique.GetPassByName( "TileManager2D" );

            mInputLayout = new InputLayout( device, mPass.Description.Signature, positionTexcoordInputElements );

            mPositionVertexBuffer = new Buffer( device,
                                                null,
                                                QUAD_NUM_VERTICES * POSITION_NUM_COMPONENTS_PER_VERTEX * POSITION_NUM_BYTES_PER_COMPONENT,
                                                ResourceUsage.Dynamic,
                                                BindFlags.VertexBuffer,
                                                CpuAccessFlags.Write,
                                                ResourceOptionFlags.None,
                                                0 );

            mTexCoordVertexBuffer = new Buffer( device,
                                                null,
                                                QUAD_NUM_VERTICES * TEXCOORD_NUM_COMPONENTS_PER_VERTEX * TEXCOORD_NUM_BYTES_PER_COMPONENT,
                                                ResourceUsage.Dynamic,
                                                BindFlags.VertexBuffer,
                                                CpuAccessFlags.Write,
                                                ResourceOptionFlags.None,
                                                0 );

            bool result;
            mTinyTextContext = new Context( device, deviceContext, Constants.MAX_NUM_TINY_TEXT_CHARACTERS, out result );
            Release.Assert( result );

            mStopwatch.Start();
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

            if ( mTileManager != null )
            {
                mTileManager.Dispose();
                mTileManager = null;
            }

            if ( mDebugRenderer != null )
            {
                mDebugRenderer.Dispose();
                mDebugRenderer = null;
            }
        }

        public void Render( DeviceContext deviceContext, Viewport viewport, RenderTargetView renderTargetView, DepthStencilView depthStencilView )
        {

            deviceContext.ClearRenderTargetView( renderTargetView, Constants.CLEAR_COLOR );
            deviceContext.ClearDepthStencilView( depthStencilView, DepthStencilClearFlags.Depth | DepthStencilClearFlags.Stencil, 1.0f, 0x00 );

            var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
            var extentDataSpace = mTileManager.TiledDatasetView.ExtentDataSpace;

            var camera = new Camera(
                new Vector3( 0, 0, 0 ),
                new Vector3( 0, 0, 1 ),
                new Vector3( 0, 1, 0 ),
                Matrix.OrthoOffCenterLH(
                    centerDataSpace.X - ( extentDataSpace.X / 2f ),
                    centerDataSpace.X + ( extentDataSpace.X / 2f ),
                    centerDataSpace.Y + ( extentDataSpace.Y / 2f ),
                    centerDataSpace.Y - ( extentDataSpace.Y / 2f ),
                    0.1f,
                    100f ) );

            var datasetExtentDataSpaceX = mTileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumTilesX * Constants.ConstParameters.GetInt( "TILE_SIZE_X" );
            var datasetExtentDataSpaceY = mTileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" ).NumTilesY * Constants.ConstParameters.GetInt( "TILE_SIZE_Y" );

            mTileManager.GetTileCache().ToList().ForEach( tileCacheEntry => RenderTileCacheEntry( deviceContext, camera, datasetExtentDataSpaceX, datasetExtentDataSpaceY, tileCacheEntry ) );

            mTinyTextContext.Print( viewport, "Frame Time: " + FrameTimeString, 10, 10 );
            mTinyTextContext.Print( viewport, "Number of Active Cache Entries: " + mTileManager.GetTileCache().Count, 10, 30 );
            mTinyTextContext.Render();

            mStopwatch.Reset();
            mStopwatch.Start();
        }

        private void RenderTileCacheEntry( DeviceContext deviceContext, Camera camera, int datasetExtentDataSpaceX, int datasetExtentDataSpaceY, TileCacheEntry tileCacheEntry )
        {
            //Check if this tile is over the edge of the image
            var tileMinExtentX = tileCacheEntry.CenterDataSpace.X - ( tileCacheEntry.ExtentDataSpace.X / 2f );
            var tileMinExtentY = tileCacheEntry.CenterDataSpace.Y - ( tileCacheEntry.ExtentDataSpace.Y / 2f );
            var tileMaxExtentX = tileCacheEntry.CenterDataSpace.X + ( tileCacheEntry.ExtentDataSpace.X / 2f );
            var tileMaxExtentY = tileCacheEntry.CenterDataSpace.Y + ( tileCacheEntry.ExtentDataSpace.Y / 2f );
            var tileProportionClipX = 1f;
            var tileProportionClipY = 1f;

            if ( datasetExtentDataSpaceX > 0 && tileMaxExtentX > datasetExtentDataSpaceX )
            {
                tileProportionClipX = 1 - ( ( tileMaxExtentX - datasetExtentDataSpaceX ) / ( tileMaxExtentX - tileMinExtentX ) );
                tileMaxExtentX = datasetExtentDataSpaceX;
            }
            if ( datasetExtentDataSpaceY > 0 && tileMaxExtentY > datasetExtentDataSpaceY )
            {
                tileProportionClipY = 1 - ( ( tileMaxExtentY - datasetExtentDataSpaceY ) / ( tileMaxExtentY - tileMinExtentY ) );
                tileMaxExtentY = datasetExtentDataSpaceY;
            }

            var p1 = new Vector3( tileMinExtentX, tileMinExtentY, 0.5f );
            var p2 = new Vector3( tileMinExtentX, tileMaxExtentY, 0.5f );
            var p3 = new Vector3( tileMaxExtentX, tileMaxExtentY, 0.5f );
            var p4 = new Vector3( tileMaxExtentX, tileMinExtentY, 0.5f );

            var t1 = new Vector3( 0f, 0f, 0f );
            var t2 = new Vector3( 0f, tileProportionClipY, 0f );
            var t3 = new Vector3( tileProportionClipX, tileProportionClipY, 0f );
            var t4 = new Vector3( tileProportionClipX, 0f, 0f );

            DataBox databox;

            databox = deviceContext.MapSubresource( mPositionVertexBuffer,
                                                    0,
                                                    QUAD_NUM_VERTICES *
                                                    POSITION_NUM_COMPONENTS_PER_VERTEX *
                                                    POSITION_NUM_BYTES_PER_COMPONENT,
                                                    MapMode.WriteDiscard,
                                                    SlimDX.Direct3D11.MapFlags.None );

            databox.Data.Write( p1 );
            databox.Data.Write( p4 );
            databox.Data.Write( p2 );
            databox.Data.Write( p3 );

            deviceContext.UnmapSubresource( mPositionVertexBuffer, 0 );

            databox = deviceContext.MapSubresource( mTexCoordVertexBuffer,
                                                    0,
                                                    QUAD_NUM_VERTICES *
                                                    TEXCOORD_NUM_COMPONENTS_PER_VERTEX *
                                                    TEXCOORD_NUM_BYTES_PER_COMPONENT,
                                                    MapMode.WriteDiscard,
                                                    SlimDX.Direct3D11.MapFlags.None );

            databox.Data.Write( t1 );
            databox.Data.Write( t4 );
            databox.Data.Write( t2 );
            databox.Data.Write( t3 );

            deviceContext.UnmapSubresource( mTexCoordVertexBuffer, 0 );

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

            mEffect.GetVariableByName( "gSourceTexture3D" ).AsResource().SetResource( tileCacheEntry.D3D11CudaTextures.Get( "SourceMap" ) );
            if ( mTileManager.SegmentationLoaded )
            {
                //if ( tileCacheEntry.D3D11CudaTextures.Internal.ContainsKey( "IdMap" ) )
                //{
                //    mEffect.GetVariableByName( "gIdTexture3D" ).AsResource().SetResource( tileCacheEntry.D3D11CudaTextures.Get( "IdMap" ) );
                //}
                //else
                //{
                //    System.Console.WriteLine("Warning: expected IdMap not found.");
                //}
                mEffect.GetVariableByName( "gIdTexture3D" ).AsResource().SetResource( tileCacheEntry.D3D11CudaTextures.Get( "IdMap" ) );
                mEffect.GetVariableByName( "gIdColorMapBuffer" ).AsResource().SetResource( mTileManager.Internal.GetIdColorMap() );
            }
            mEffect.GetVariableByName( "gTransform" ).AsMatrix().SetMatrix( camera.GetLookAtMatrix() * camera.GetProjectionMatrix() );
            mEffect.GetVariableByName( "gSegmentationRatio" ).AsScalar().Set( mTileManager.SegmentationVisibilityRatio );
            mEffect.GetVariableByName( "gBoundaryLinesVisible" ).AsScalar().Set( mTileManager.ShowBoundaryLines );
            mEffect.GetVariableByName( "gSelectedSegmentId" ).AsScalar().Set( mTileManager.SelectedSegmentId );
            mEffect.GetVariableByName( "gMouseOverSegmentId" ).AsScalar().Set( mTileManager.MouseOverSegmentId );

            mPass.Apply( deviceContext );
            deviceContext.Draw( QUAD_NUM_VERTICES, 0 );
            
            mDebugRenderer.RenderQuadWireframeOnly( deviceContext, p1, p2, p3, p4, new Vector3( 1, 0, 0 ), camera );

        }
    }
}
