using System;
using SlimDX;
using SlimDX.Direct3D11;
using SlimDX.DXGI;

namespace Mojo
{
    class DebugRenderer : IDisposable
    {
        private const int POSITION_SLOT = 0;
        private const int POSITION_NUM_BYTES_PER_COMPONENT = 4;
        private const int POSITION_NUM_COMPONENTS_PER_VERTEX = 3;

        private const int TEXCOORD_SLOT = 1;
        private const int TEXCOORD_NUM_BYTES_PER_COMPONENT = 4;
        private const int TEXCOORD_NUM_COMPONENTS_PER_VERTEX = 3;

        private const Format POSITION_FORMAT = Format.R32G32B32_Float;
        private const Format TEXCOORD_FORMAT = Format.R32G32B32_Float;

        private const int NUM_VERTICES = 1024;
        private const int POINT_NUM_VERTICES = 1;
        private const int LINE_NUM_VERTICES = 2;
        private const int QUAD_NUM_VERTICES = 4;

        const int NUM_LATITUDE_LINES = 4;
        const int NUM_LONGITUDE_LINES = 4;

        const float LATITUDE_STEP = -(float)Math.PI / NUM_LATITUDE_LINES;
        const float LONGITUDE_STEP = ( 2 * (float)Math.PI ) / NUM_LONGITUDE_LINES;

        private readonly Effect mEffect;

        private readonly EffectPass mRenderWireframePass;
        private readonly EffectPass mRenderSolidPass;
        private readonly EffectPass mRenderTexture3DPass;
        private readonly EffectPass mRenderGreyScaleTexture3DPass;

        private InputLayout mRenderWireframeInputLayout;
        private InputLayout mRenderSolidInputLayout;
        private InputLayout mRenderTexture3DInputLayout;
        private InputLayout mRenderGreyScaleTexture3DInputLayout;

        private SlimDX.Direct3D11.Buffer mPositionVertexBuffer;
        private SlimDX.Direct3D11.Buffer mTexCoordVertexBuffer;

        public DebugRenderer( SlimDX.Direct3D11.Device device )
        {
            mEffect = EffectUtil.CompileEffect( device, @"Shaders\DebugRenderer.fx" );

            var positionInputElements = new[]
                                        {
                                            new InputElement( "POSITION", 0, POSITION_FORMAT, POSITION_SLOT )
                                        };

            var positionTexcoordInputElements = new[]
                                                {
                                                    new InputElement( "POSITION", 0, POSITION_FORMAT, POSITION_SLOT ),
                                                    new InputElement( "TEXCOORD", 0, TEXCOORD_FORMAT, TEXCOORD_SLOT )
                                                };

            EffectTechnique effectTechnique;

            effectTechnique = mEffect.GetTechniqueByName( "RenderWireframe" );
            mRenderWireframePass = effectTechnique.GetPassByName( "RenderWireframe" );

            effectTechnique = mEffect.GetTechniqueByName( "RenderSolid" );
            mRenderSolidPass = effectTechnique.GetPassByName( "RenderSolid" );

            effectTechnique = mEffect.GetTechniqueByName( "RenderTexture3D" );
            mRenderTexture3DPass = effectTechnique.GetPassByName( "RenderTexture3D" );

            effectTechnique = mEffect.GetTechniqueByName( "RenderGreyScaleTexture3D" );
            mRenderGreyScaleTexture3DPass = effectTechnique.GetPassByName( "RenderGreyScaleTexture3D" );

            mRenderWireframeInputLayout = new InputLayout( device, mRenderWireframePass.Description.Signature, positionInputElements );
            mRenderSolidInputLayout = new InputLayout( device, mRenderSolidPass.Description.Signature, positionInputElements );
            mRenderTexture3DInputLayout = new InputLayout( device, mRenderTexture3DPass.Description.Signature, positionTexcoordInputElements );
            mRenderGreyScaleTexture3DInputLayout = new InputLayout( device, mRenderGreyScaleTexture3DPass.Description.Signature, positionTexcoordInputElements );

            mPositionVertexBuffer = new SlimDX.Direct3D11.Buffer( device,
                                                                  null,
                                                                  NUM_VERTICES * POSITION_NUM_COMPONENTS_PER_VERTEX * POSITION_NUM_BYTES_PER_COMPONENT,
                                                                  ResourceUsage.Dynamic,
                                                                  BindFlags.VertexBuffer,
                                                                  CpuAccessFlags.Write,
                                                                  ResourceOptionFlags.None,
                                                                  0 );

            mTexCoordVertexBuffer = new SlimDX.Direct3D11.Buffer( device,
                                                                  null,
                                                                  NUM_VERTICES * TEXCOORD_NUM_COMPONENTS_PER_VERTEX * TEXCOORD_NUM_BYTES_PER_COMPONENT,
                                                                  ResourceUsage.Dynamic,
                                                                  BindFlags.VertexBuffer,
                                                                  CpuAccessFlags.Write,
                                                                  ResourceOptionFlags.None,
                                                                  0 );
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

            if ( mRenderGreyScaleTexture3DInputLayout != null )
            {
                mRenderGreyScaleTexture3DInputLayout.Dispose();
                mRenderGreyScaleTexture3DInputLayout = null;
            }

            if ( mRenderTexture3DInputLayout != null )
            {
                mRenderTexture3DInputLayout.Dispose();
                mRenderTexture3DInputLayout = null;
            }

            if ( mRenderSolidInputLayout != null )
            {
                mRenderSolidInputLayout.Dispose();
                mRenderSolidInputLayout = null;
            }

            if ( mRenderWireframeInputLayout != null )
            {
                mRenderWireframeInputLayout.Dispose();
                mRenderWireframeInputLayout = null;
            }

            mEffect.Dispose();
        }

        public void RenderPoint( DeviceContext deviceContext, Vector3 p, Vector3 color, Camera camera )
        {
            var databox = deviceContext.MapSubresource( mPositionVertexBuffer,
                                                        0,
                                                        POINT_NUM_VERTICES *
                                                        POSITION_NUM_COMPONENTS_PER_VERTEX *
                                                        POSITION_NUM_BYTES_PER_COMPONENT,
                                                        MapMode.WriteDiscard,
                                                        SlimDX.Direct3D11.MapFlags.None );

            databox.Data.Write( p );

            deviceContext.UnmapSubresource( mPositionVertexBuffer, 0 );

            deviceContext.InputAssembler.InputLayout = mRenderWireframeInputLayout;
            deviceContext.InputAssembler.PrimitiveTopology = PrimitiveTopology.PointList;
            deviceContext.InputAssembler.SetVertexBuffers( POSITION_SLOT,
                                                           new VertexBufferBinding( mPositionVertexBuffer,
                                                                                    POSITION_NUM_COMPONENTS_PER_VERTEX *
                                                                                    POSITION_NUM_BYTES_PER_COMPONENT,
                                                                                    0 ) );

            mEffect.GetVariableByName( "gColor" ).AsVector().Set( color );
            mEffect.GetVariableByName( "gTransform" ).AsMatrix().SetMatrix( camera.GetLookAtMatrix() * camera.GetProjectionMatrix() );
            mRenderWireframePass.Apply( deviceContext );

            deviceContext.Draw( POINT_NUM_VERTICES, 0 );
        }

        public void RenderLine( DeviceContext deviceContext, Vector3 p1, Vector3 p2, Vector3 color, Camera camera )
        {
            var databox = deviceContext.MapSubresource( mPositionVertexBuffer,
                                                        0,
                                                        LINE_NUM_VERTICES *
                                                        POSITION_NUM_COMPONENTS_PER_VERTEX *
                                                        POSITION_NUM_BYTES_PER_COMPONENT,
                                                        MapMode.WriteDiscard,
                                                        SlimDX.Direct3D11.MapFlags.None );

            databox.Data.Write( p1 );
            databox.Data.Write( p2 );

            deviceContext.UnmapSubresource( mPositionVertexBuffer, 0 );

            deviceContext.InputAssembler.InputLayout = mRenderWireframeInputLayout;
            deviceContext.InputAssembler.PrimitiveTopology = PrimitiveTopology.LineStrip;
            deviceContext.InputAssembler.SetVertexBuffers( POSITION_SLOT,
                                                           new VertexBufferBinding( mPositionVertexBuffer,
                                                                                    POSITION_NUM_COMPONENTS_PER_VERTEX *
                                                                                    POSITION_NUM_BYTES_PER_COMPONENT,
                                                                                    0 ) );

            mEffect.GetVariableByName( "gColor" ).AsVector().Set( color );
            mEffect.GetVariableByName( "gTransform" ).AsMatrix().SetMatrix( camera.GetLookAtMatrix() * camera.GetProjectionMatrix() );
            mRenderWireframePass.Apply( deviceContext );

            deviceContext.Draw( LINE_NUM_VERTICES, 0 );
        }

        public void RenderQuadWireframeOnly( DeviceContext deviceContext, Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4, Vector3 color, Camera camera )
        {
            RenderLine( deviceContext, p1, p2, color, camera );
            RenderLine( deviceContext, p2, p3, color, camera );
            RenderLine( deviceContext, p3, p4, color, camera );
            RenderLine( deviceContext, p4, p1, color, camera );
        }

        public void RenderQuadSolidOnly( DeviceContext deviceContext, Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4, Vector3 color, Camera camera )
        {
            DataBox databox = deviceContext.MapSubresource( mPositionVertexBuffer,
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

            deviceContext.InputAssembler.InputLayout = mRenderSolidInputLayout;
            deviceContext.InputAssembler.PrimitiveTopology = PrimitiveTopology.TriangleStrip;
            deviceContext.InputAssembler.SetVertexBuffers( POSITION_SLOT,
                                                           new VertexBufferBinding( mPositionVertexBuffer,
                                                                                    POSITION_NUM_COMPONENTS_PER_VERTEX *
                                                                                    POSITION_NUM_BYTES_PER_COMPONENT,
                                                                                    0 ) );

            mEffect.GetVariableByName( "gTransform" ).AsMatrix().SetMatrix( camera.GetLookAtMatrix() * camera.GetProjectionMatrix() );
            mEffect.GetVariableByName( "gColor" ).AsVector().Set( color );

            mRenderSolidPass.Apply( deviceContext );
            deviceContext.Draw( QUAD_NUM_VERTICES, 0 );
        }

        public void RenderQuadTexture3DOnly( DeviceContext deviceContext, Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4, Vector3 t1, Vector3 t2, Vector3 t3, Vector3 t4, ShaderResourceView texture, Camera camera )
        {
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

            deviceContext.InputAssembler.InputLayout = mRenderTexture3DInputLayout;
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


            mEffect.GetVariableByName( "gTexture3D" ).AsResource().SetResource( texture );
            mEffect.GetVariableByName( "gTransform" ).AsMatrix().SetMatrix( camera.GetLookAtMatrix() * camera.GetProjectionMatrix() );

            mRenderTexture3DPass.Apply( deviceContext );
            deviceContext.Draw( QUAD_NUM_VERTICES, 0 );
        }

        public void RenderQuadGreyScaleTexture3DOnly( DeviceContext deviceContext, Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4, Vector3 t1, Vector3 t2, Vector3 t3, Vector3 t4, ShaderResourceView texture, Camera camera )
        {
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

            deviceContext.InputAssembler.InputLayout = mRenderGreyScaleTexture3DInputLayout;
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


            mEffect.GetVariableByName( "gTexture3D" ).AsResource().SetResource( texture );
            mEffect.GetVariableByName( "gTransform" ).AsMatrix().SetMatrix( camera.GetLookAtMatrix() * camera.GetProjectionMatrix() );

            mRenderGreyScaleTexture3DPass.Apply( deviceContext );
            deviceContext.Draw( QUAD_NUM_VERTICES, 0 );
        }

        public void RenderQuadSolidWireframe( DeviceContext deviceContext, Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4, Vector3 backgroundColor, Vector3 foregroundColor, Camera camera )
        {
            RenderQuadSolidOnly( deviceContext, p1, p2, p3, p4, backgroundColor, camera );
            RenderQuadWireframeOnly( deviceContext, p1, p2, p3, p4, foregroundColor, camera );
        }

        public void RenderQuadTexture3DWireframe( DeviceContext deviceContext, Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4, Vector3 t1, Vector3 t2, Vector3 t3, Vector3 t4, ShaderResourceView texture, Vector3 color, Camera camera )
        {
            RenderQuadTexture3DOnly( deviceContext, p1, p2, p3, p4, t1, t2, t3, t4, texture, camera );
            RenderQuadWireframeOnly( deviceContext, p1, p2, p3, p4, color, camera );
        }

        public void RenderQuadGreyScaleTexture3DWireframe( DeviceContext deviceContext, Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4, Vector3 t1, Vector3 t2, Vector3 t3, Vector3 t4, ShaderResourceView texture, Vector3 color, Camera camera )
        {
            RenderQuadGreyScaleTexture3DOnly( deviceContext, p1, p2, p3, p4, t1, t2, t3, t4, texture, camera );
            RenderQuadWireframeOnly( deviceContext, p1, p2, p3, p4, color, camera );
        }

        public void RenderBoxWireframeOnly( DeviceContext deviceContext, Vector3 p1, Vector3 p2, Vector3 color, Camera camera )
        {
            var base1 = new Vector3( p1.X, p1.Y, p1.Z );
            var base2 = new Vector3( p1.X, p1.Y, p2.Z );
            var base3 = new Vector3( p2.X, p1.Y, p2.Z );
            var base4 = new Vector3( p2.X, p1.Y, p1.Z );

            var lid1 = new Vector3( p1.X, p2.Y, p1.Z );
            var lid2 = new Vector3( p1.X, p2.Y, p2.Z );
            var lid3 = new Vector3( p2.X, p2.Y, p2.Z );
            var lid4 = new Vector3( p2.X, p2.Y, p1.Z );

            RenderQuadWireframeOnly( deviceContext, base1, base2, base3, base4, color, camera );
            RenderQuadWireframeOnly( deviceContext, lid1, lid2, lid3, lid4, color, camera );
            RenderQuadWireframeOnly( deviceContext, base1, base2, lid2, lid1, color, camera );
            RenderQuadWireframeOnly( deviceContext, base2, base3, lid3, lid2, color, camera );
            RenderQuadWireframeOnly( deviceContext, base3, base4, lid4, lid3, color, camera );
            RenderQuadWireframeOnly( deviceContext, base4, base1, lid1, lid4, color, camera );
        }

        public void RenderBoxSolidOnly( DeviceContext deviceContext, Vector3 p1, Vector3 p2, Vector3 color, Camera camera )
        {
            var base1 = new Vector3( p1.X, p1.Y, p1.Z );
            var base2 = new Vector3( p1.X, p1.Y, p2.Z );
            var base3 = new Vector3( p2.X, p1.Y, p2.Z );
            var base4 = new Vector3( p2.X, p1.Y, p1.Z );

            var lid1 = new Vector3( p1.X, p2.Y, p1.Z );
            var lid2 = new Vector3( p1.X, p2.Y, p2.Z );
            var lid3 = new Vector3( p2.X, p2.Y, p2.Z );
            var lid4 = new Vector3( p2.X, p2.Y, p1.Z );

            RenderQuadSolidOnly( deviceContext, base1, base2, base3, base4, color, camera );
            RenderQuadSolidOnly( deviceContext, lid1, lid2, lid3, lid4, color, camera );
            RenderQuadSolidOnly( deviceContext, base1, base2, lid2, lid1, color, camera );
            RenderQuadSolidOnly( deviceContext, base2, base3, lid3, lid2, color, camera );
            RenderQuadSolidOnly( deviceContext, base3, base4, lid4, lid3, color, camera );
            RenderQuadSolidOnly( deviceContext, base4, base1, lid1, lid4, color, camera );
        }

        public void RenderBoxTexture3DOnly( DeviceContext deviceContext, Vector3 p1, Vector3 p2, Vector3 t1, Vector3 t2, ShaderResourceView texture, Camera camera )
        {
            var pBase1 = new Vector3( p1.X, p1.Y, p1.Z );
            var pBase2 = new Vector3( p1.X, p1.Y, p2.Z );
            var pBase3 = new Vector3( p2.X, p1.Y, p2.Z );
            var pBase4 = new Vector3( p2.X, p1.Y, p1.Z );

            var pLid1 = new Vector3( p1.X, p2.Y, p1.Z );
            var pLid2 = new Vector3( p1.X, p2.Y, p2.Z );
            var pLid3 = new Vector3( p2.X, p2.Y, p2.Z );
            var pLid4 = new Vector3( p2.X, p2.Y, p1.Z );

            var tBase1 = new Vector3( t1.X, t1.Y, t1.Z );
            var tBase2 = new Vector3( t1.X, t1.Y, t2.Z );
            var tBase3 = new Vector3( t2.X, t1.Y, t2.Z );
            var tBase4 = new Vector3( t2.X, t1.Y, t1.Z );

            var tLid1 = new Vector3( t1.X, t2.Y, t1.Z );
            var tLid2 = new Vector3( t1.X, t2.Y, t2.Z );
            var tLid3 = new Vector3( t2.X, t2.Y, t2.Z );
            var tLid4 = new Vector3( t2.X, t2.Y, t1.Z );

            RenderQuadTexture3DOnly( deviceContext, pBase1, pBase2, pBase3, pBase4, tBase1, tBase2, tBase3, tBase4, texture, camera );
            RenderQuadTexture3DOnly( deviceContext, pLid1, pLid2, pLid3, pLid4, tLid1, tLid2, tLid3, tLid4, texture, camera );
            RenderQuadTexture3DOnly( deviceContext, pBase1, pBase2, pLid2, pLid1, tBase1, tBase2, tLid2, tLid1, texture, camera );
            RenderQuadTexture3DOnly( deviceContext, pBase2, pBase3, pLid3, pLid2, tBase2, tBase3, tLid3, tLid2, texture, camera );
            RenderQuadTexture3DOnly( deviceContext, pBase3, pBase4, pLid4, pLid3, tBase3, tBase4, tLid4, tLid3, texture, camera );
            RenderQuadTexture3DOnly( deviceContext, pBase4, pBase1, pLid1, pLid4, tBase4, tBase1, tLid1, tLid4, texture, camera );
        }

        public void RenderBoxGreyScaleTexture3DOnly( DeviceContext deviceContext, Vector3 p1, Vector3 p2, Vector3 t1, Vector3 t2, ShaderResourceView texture, Camera camera )
        {
            var pBase1 = new Vector3( p1.X, p1.Y, p1.Z );
            var pBase2 = new Vector3( p1.X, p1.Y, p2.Z );
            var pBase3 = new Vector3( p2.X, p1.Y, p2.Z );
            var pBase4 = new Vector3( p2.X, p1.Y, p1.Z );

            var pLid1 = new Vector3( p1.X, p2.Y, p1.Z );
            var pLid2 = new Vector3( p1.X, p2.Y, p2.Z );
            var pLid3 = new Vector3( p2.X, p2.Y, p2.Z );
            var pLid4 = new Vector3( p2.X, p2.Y, p1.Z );

            var tBase1 = new Vector3( t1.X, t1.Y, t1.Z );
            var tBase2 = new Vector3( t1.X, t1.Y, t2.Z );
            var tBase3 = new Vector3( t2.X, t1.Y, t2.Z );
            var tBase4 = new Vector3( t2.X, t1.Y, t1.Z );

            var tLid1 = new Vector3( t1.X, t2.Y, t1.Z );
            var tLid2 = new Vector3( t1.X, t2.Y, t2.Z );
            var tLid3 = new Vector3( t2.X, t2.Y, t2.Z );
            var tLid4 = new Vector3( t2.X, t2.Y, t1.Z );

            RenderQuadGreyScaleTexture3DOnly( deviceContext, pBase1, pBase2, pBase3, pBase4, tBase1, tBase2, tBase3, tBase4, texture, camera );
            RenderQuadGreyScaleTexture3DOnly( deviceContext, pLid1, pLid2, pLid3, pLid4, tLid1, tLid2, tLid3, tLid4, texture, camera );
            RenderQuadGreyScaleTexture3DOnly( deviceContext, pBase1, pBase2, pLid2, pLid1, tBase1, tBase2, tLid2, tLid1, texture, camera );
            RenderQuadGreyScaleTexture3DOnly( deviceContext, pBase2, pBase3, pLid3, pLid2, tBase2, tBase3, tLid3, tLid2, texture, camera );
            RenderQuadGreyScaleTexture3DOnly( deviceContext, pBase3, pBase4, pLid4, pLid3, tBase3, tBase4, tLid4, tLid3, texture, camera );
            RenderQuadGreyScaleTexture3DOnly( deviceContext, pBase4, pBase1, pLid1, pLid4, tBase4, tBase1, tLid1, tLid4, texture, camera );
        }

        public void RenderBoxSolidWireframe( DeviceContext deviceContext, Vector3 p1, Vector3 p2, Vector3 backgroundColor, Vector3 foregroundColor, Camera camera )
        {
            RenderBoxSolidOnly( deviceContext, p1, p2, backgroundColor, camera );
            RenderBoxWireframeOnly( deviceContext, p1, p2, foregroundColor, camera );
        }

        public void RenderBoxTexture3DWireframe( DeviceContext deviceContext, Vector3 p1, Vector3 p2, Vector3 t1, Vector3 t2, ShaderResourceView texture, Vector3 color, Camera camera )
        {
            RenderBoxTexture3DOnly( deviceContext, p1, p2, t1, t2, texture, camera );
            RenderBoxWireframeOnly( deviceContext, p1, p2, color, camera );
        }

        public void RenderBoxGreyScaleTexture3DWireframe( DeviceContext deviceContext, Vector3 p1, Vector3 p2, Vector3 t1, Vector3 t2, ShaderResourceView texture, Vector3 color, Camera camera )
        {
            RenderBoxGreyScaleTexture3DOnly( deviceContext, p1, p2, t1, t2, texture, camera );
            RenderBoxWireframeOnly( deviceContext, p1, p2, color, camera );
        }

        public void RenderSphereWireframeOnly( DeviceContext deviceContext, Vector3 p, float radius, Vector3 color, Camera camera )
        {
            var databox = deviceContext.MapSubresource( mPositionVertexBuffer,
                                                        0,
                                                        NUM_VERTICES *
                                                        POSITION_NUM_COMPONENTS_PER_VERTEX *
                                                        POSITION_NUM_BYTES_PER_COMPONENT,
                                                        MapMode.WriteDiscard,
                                                        SlimDX.Direct3D11.MapFlags.None );

            var currentPoint = new Vector3();
            var furtherSouthPoint = new Vector3();
            var numVertices = 0;

            // northSouthTheta traces from north pole to south pole
            float northSouthTheta = (float)Math.PI / 2;

            for ( int i = 0; i <= NUM_LATITUDE_LINES; i++ )
            {
                float currentLatitudeRadius = (float)Math.Cos( northSouthTheta ) * radius;
                float nextLatitudeRadius = (float)Math.Cos( northSouthTheta + LATITUDE_STEP ) * radius;

                // eastWestTheta traces around each latitude line
                float eastWestTheta = 0;

                for ( int j = 0; j <= NUM_LONGITUDE_LINES; j++ )
                {
                    currentPoint.X = p.X + ( (float)Math.Cos( eastWestTheta ) * currentLatitudeRadius );
                    currentPoint.Y = p.Y + ( (float)Math.Sin( northSouthTheta ) * radius );
                    currentPoint.Z = p.Z + ( (float)Math.Sin( eastWestTheta ) * currentLatitudeRadius );

                    databox.Data.Write( currentPoint );
                    numVertices++;

                    furtherSouthPoint.X = p.X + ( (float)Math.Cos( eastWestTheta ) * nextLatitudeRadius );
                    furtherSouthPoint.Y = p.Y + ( (float)Math.Sin( northSouthTheta + LATITUDE_STEP ) * radius );
                    furtherSouthPoint.Z = p.Z + ( (float)Math.Sin( eastWestTheta ) * nextLatitudeRadius );

                    databox.Data.Write( furtherSouthPoint );
                    numVertices++;

                    eastWestTheta += LONGITUDE_STEP;
                }

                northSouthTheta += LATITUDE_STEP;
            }

            deviceContext.UnmapSubresource( mPositionVertexBuffer, 0 );

            deviceContext.InputAssembler.InputLayout = mRenderWireframeInputLayout;
            deviceContext.InputAssembler.PrimitiveTopology = PrimitiveTopology.TriangleStrip;
            deviceContext.InputAssembler.SetVertexBuffers( POSITION_SLOT,
                                                           new VertexBufferBinding( mPositionVertexBuffer,
                                                                                    POSITION_NUM_COMPONENTS_PER_VERTEX *
                                                                                    POSITION_NUM_BYTES_PER_COMPONENT,
                                                                                    0 ) );

            mEffect.GetVariableByName( "gTransform" ).AsMatrix().SetMatrix( camera.GetLookAtMatrix() * camera.GetProjectionMatrix() );
            mEffect.GetVariableByName( "gColor" ).AsVector().Set( color );

            mRenderWireframePass.Apply( deviceContext );
            deviceContext.Draw( numVertices, 0 );
        }

        public void RenderSphereSolidOnly( DeviceContext deviceContext, Vector3 p, float radius, Vector3 color, Camera camera )
        {
            var databox = deviceContext.MapSubresource( mPositionVertexBuffer,
                                                        0,
                                                        NUM_VERTICES *
                                                        POSITION_NUM_COMPONENTS_PER_VERTEX *
                                                        POSITION_NUM_BYTES_PER_COMPONENT,
                                                        MapMode.WriteDiscard,
                                                        SlimDX.Direct3D11.MapFlags.None );

            var currentPoint = new Vector3();
            var furtherSouthPoint = new Vector3();
            var numVertices = 0;

            // northSouthTheta traces from north pole to south pole
            float northSouthTheta = (float)Math.PI / 2;

            for ( int i = 0; i <= NUM_LATITUDE_LINES; i++ )
            {
                float currentLatitudeRadius = (float)Math.Cos( northSouthTheta ) * radius;
                float nextLatitudeRadius = (float)Math.Cos( northSouthTheta + LATITUDE_STEP ) * radius;

                // eastWestTheta traces around each latitude line
                float eastWestTheta = 0;

                for ( int j = 0; j <= NUM_LONGITUDE_LINES; j++ )
                {
                    currentPoint.X = p.X + ( (float)Math.Cos( eastWestTheta ) * currentLatitudeRadius );
                    currentPoint.Y = p.Y + ( (float)Math.Sin( northSouthTheta ) * radius );
                    currentPoint.Z = p.Z + ( (float)Math.Sin( eastWestTheta ) * currentLatitudeRadius );

                    databox.Data.Write( currentPoint );
                    numVertices++;

                    furtherSouthPoint.X = p.X + ( (float)Math.Cos( eastWestTheta ) * nextLatitudeRadius );
                    furtherSouthPoint.Y = p.Y + ( (float)Math.Sin( northSouthTheta + LATITUDE_STEP ) * radius );
                    furtherSouthPoint.Z = p.Z + ( (float)Math.Sin( eastWestTheta ) * nextLatitudeRadius );

                    databox.Data.Write( furtherSouthPoint );
                    numVertices++;

                    eastWestTheta += LONGITUDE_STEP;
                }

                northSouthTheta += LATITUDE_STEP;
            }

            deviceContext.UnmapSubresource( mPositionVertexBuffer, 0 );

            deviceContext.InputAssembler.InputLayout = mRenderSolidInputLayout;
            deviceContext.InputAssembler.PrimitiveTopology = PrimitiveTopology.TriangleStrip;
            deviceContext.InputAssembler.SetVertexBuffers( POSITION_SLOT,
                                                           new VertexBufferBinding( mPositionVertexBuffer,
                                                                                    POSITION_NUM_COMPONENTS_PER_VERTEX *
                                                                                    POSITION_NUM_BYTES_PER_COMPONENT,
                                                                                    0 ) );

            mEffect.GetVariableByName( "gTransform" ).AsMatrix().SetMatrix( camera.GetLookAtMatrix() * camera.GetProjectionMatrix() );
            mEffect.GetVariableByName( "gColor" ).AsVector().Set( color );

            mRenderSolidPass.Apply( deviceContext );
            deviceContext.Draw( numVertices, 0 );
        }

        public void RenderSphereSolidWireframe( DeviceContext deviceContext, Vector3 p, float radius, Vector3 backgroundColor, Vector3 foregroundColor, Camera camera )
        {
            RenderSphereSolidOnly( deviceContext, p, radius, backgroundColor, camera );
            RenderSphereWireframeOnly( deviceContext, p, radius, foregroundColor, camera );
        }
    }
}
